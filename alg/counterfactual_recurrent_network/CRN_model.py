'''
Title: Estimating counterfactual treatment outcomes over time through adversarially balanced representations
Authors: Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
International Conference on Learning Representations (ICLR) 2020

Last Updated Date: January 15th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''


import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.python.ops import rnn

from utils.flip_gradient import flip_gradient
import numpy as np
import os

import logging


class CRN_Model:
    def __init__(self, params, hyperparams, b_train_decoder=False):
        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        self.num_outputs = params['num_outputs']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']

        self.br_size = hyperparams['br_size']
        self.rnn_hidden_units = hyperparams['rnn_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.batch_size = hyperparams['batch_size']
        self.rnn_keep_prob = hyperparams['rnn_keep_prob']
        self.learning_rate = hyperparams['learning_rate']

        self.b_train_decoder = b_train_decoder

        tf.reset_default_graph()

        self.current_covariates = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_covariates])

        # Initial previous treatment needs to consist of zeros (this is done when building the feed dictionary)
        self.previous_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_treatments])
        self.current_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_treatments])
        self.outputs = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_outputs])
        self.active_entries = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_outputs])

        self.init_state = None
        if (self.b_train_decoder):
            self.init_state = tf.placeholder(tf.float32, [None, self.rnn_hidden_units])

        self.alpha = tf.placeholder(tf.float32, [])  # Gradient reversal scalar

    def build_balancing_representation(self):
        self.rnn_input = tf.concat([self.current_covariates, self.previous_treatments], axis=-1)
        self.sequence_length = self.compute_sequence_length(self.rnn_input)

        rnn_cell = DropoutWrapper(LSTMCell(self.rnn_hidden_units, state_is_tuple=False),
                                  output_keep_prob=self.rnn_keep_prob,
                                  state_keep_prob=self.rnn_keep_prob,
                                  variational_recurrent=True,
                                  dtype=tf.float32)

        decoder_init_state = None
        if (self.b_train_decoder):
            decoder_init_state = tf.concat([self.init_state, self.init_state], axis=-1)

        rnn_output, _ = rnn.dynamic_rnn(
            rnn_cell,
            self.rnn_input,
            initial_state=decoder_init_state,
            dtype=tf.float32,
            sequence_length=self.sequence_length)

        # Flatten to apply same weights to all time steps.
        rnn_output = tf.reshape(rnn_output, [-1, self.rnn_hidden_units])
        balancing_representation = tf.layers.dense(rnn_output, self.br_size, activation=tf.nn.elu)

        return balancing_representation

    def build_treatment_assignments_one_hot(self, balancing_representation):
        balancing_representation_gr = flip_gradient(balancing_representation, self.alpha)

        treatments_network_layer = tf.layers.dense(balancing_representation_gr, self.fc_hidden_units,
                                                   activation=tf.nn.elu)
        treatment_logit_predictions = tf.layers.dense(treatments_network_layer, self.num_treatments, activation=None)
        treatment_prob_predictions = tf.nn.softmax(treatment_logit_predictions)

        return treatment_prob_predictions

    def build_outcomes(self, balancing_representation):
        current_treatments_reshape = tf.reshape(self.current_treatments, [-1, self.num_treatments])

        outcome_network_input = tf.concat([balancing_representation, current_treatments_reshape], axis=-1)
        outcome_network_layer = tf.layers.dense(outcome_network_input, self.fc_hidden_units,
                                                activation=tf.nn.elu)
        outcome_predictions = tf.layers.dense(outcome_network_layer, self.num_outputs, activation=None)

        return outcome_predictions

    def train(self, dataset_train, dataset_val, model_name, model_folder):
        self.balancing_representation = self.build_balancing_representation()
        self.treatment_prob_predictions = self.build_treatment_assignments_one_hot(self.balancing_representation)
        self.predictions = self.build_outcomes(self.balancing_representation)

        self.loss_treatments = self.compute_loss_treatments_one_hot(target_treatments=self.current_treatments,
                                                                    treatment_predictions=self.treatment_prob_predictions,
                                                                    active_entries=self.active_entries)
        self.loss_outcomes = self.compute_loss_predictions(self.outputs, self.predictions, self.active_entries)
        self.loss = self.loss_outcomes + self.loss_treatments
        optimizer = self.get_optimizer()

        # Setup tensorflow
        tf_device = 'gpu'
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        for epoch in range(self.num_epochs):
            p = float(epoch) / float(self.num_epochs)
            alpha_current = 2. / (1. + np.exp(-10. * p)) - 1

            iteration = 0
            for (batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state,
                 batch_outputs, batch_active_entries) in self.gen_epoch(dataset_train, batch_size=self.batch_size):
                feed_dict = self.build_feed_dictionary(batch_current_covariates, batch_previous_treatments,
                                                       batch_current_treatments, batch_init_state, batch_outputs,
                                                       batch_active_entries,
                                                       alpha_current)

                _, training_loss, training_loss_outcomes, training_loss_treatments = self.sess.run(
                    [optimizer, self.loss, self.loss_outcomes, self.loss_treatments],
                    feed_dict=feed_dict)

                iteration += 1

            logging.info(
                "Epoch {} out of {} | total loss = {} | outcome loss = {} | "
                "treatment loss = {} | current alpha = {} ".format(epoch + 1, self.num_epochs, training_loss,
                                                                   training_loss_outcomes,
                                                                   training_loss_treatments, alpha_current))

        # Validation loss
        validation_loss, validation_loss_outcomes, \
        validation_loss_treatments = self.compute_validation_loss(dataset_val)

        validation_mse, _ = self.evaluate_predictions(dataset_val)

        logging.info(
            "Epoch {} Summary| Validation total loss = {} | Validation outcome loss = {} | Validation treatment loss {} | Validation mse = {}".format(
                epoch, validation_loss, validation_loss_outcomes, validation_loss_treatments, validation_mse))

        checkpoint_name = model_name + "_final"
        self.save_network(self.sess, model_folder, checkpoint_name)

    def load_model(self, model_name, model_folder):
        self.balancing_representation = self.build_balancing_representation()
        self.treatment_prob_predictions = self.build_treatment_assignments_one_hot(self.balancing_representation)
        self.predictions = self.build_outcomes(self.balancing_representation)

        tf_device = 'gpu'
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        checkpoint_name = model_name + "_final"
        self.load_network(self.sess, model_folder, checkpoint_name)

    def build_feed_dictionary(self, batch_current_covariates, batch_previous_treatments,
                              batch_current_treatments, batch_init_state,
                              batch_outputs=None, batch_active_entries=None,
                              alpha_current=1.0, lr_current=0.01, training_mode=True):
        batch_size = batch_previous_treatments.shape[0]
        zero_init_treatment = np.zeros(shape=[batch_size, 1, self.num_treatments])
        new_batch_previous_treatments = np.concatenate([zero_init_treatment, batch_previous_treatments], axis=1)

        if training_mode:
            if self.b_train_decoder:
                feed_dict = {self.current_covariates: batch_current_covariates,
                             self.previous_treatments: batch_previous_treatments,
                             self.current_treatments: batch_current_treatments,
                             self.init_state: batch_init_state,
                             self.outputs: batch_outputs,
                             self.active_entries: batch_active_entries,
                             self.alpha: alpha_current}

            else:
                feed_dict = {self.current_covariates: batch_current_covariates,
                             self.previous_treatments: new_batch_previous_treatments,
                             self.current_treatments: batch_current_treatments,
                             self.outputs: batch_outputs,
                             self.active_entries: batch_active_entries,
                             self.alpha: alpha_current}
        else:
            if self.b_train_decoder:
                feed_dict = {self.current_covariates: batch_current_covariates,
                             self.previous_treatments: batch_previous_treatments,
                             self.current_treatments: batch_current_treatments,
                             self.init_state: batch_init_state,
                             self.alpha: alpha_current}
            else:
                feed_dict = {self.current_covariates: batch_current_covariates,
                             self.previous_treatments: new_batch_previous_treatments,
                             self.current_treatments: batch_current_treatments,
                             self.alpha: alpha_current}

        return feed_dict

    def gen_epoch(self, dataset, batch_size, training_mode=True):
        dataset_size = dataset['current_covariates'].shape[0]
        num_batches = int(dataset_size / batch_size) + 1

        for i in range(num_batches):
            if (i == num_batches - 1):
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(i * batch_size, (i + 1) * batch_size)

            if training_mode:
                batch_current_covariates = dataset['current_covariates'][batch_samples, :, :]
                batch_previous_treatments = dataset['previous_treatments'][batch_samples, :, :]
                batch_current_treatments = dataset['current_treatments'][batch_samples, :, :]
                batch_outputs = dataset['outputs'][batch_samples, :, :]
                batch_active_entries = dataset['active_entries'][batch_samples, :, :]

                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset['init_state'][batch_samples, :]

                yield (batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state,
                       batch_outputs, batch_active_entries)
            else:
                batch_current_covariates = dataset['current_covariates'][batch_samples, :, :]
                batch_previous_treatments = dataset['previous_treatments'][batch_samples, :, :]
                batch_current_treatments = dataset['current_treatments'][batch_samples, :, :]

                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset['init_state'][batch_samples, :]

                yield (batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state)

    def compute_validation_loss(self, dataset):
        validation_losses = []
        validation_losses_outcomes = []
        validation_losses_treatments = []

        dataset_size = dataset['current_covariates'].shape[0]
        if (dataset_size > 10000):
            batch_size = 10000
        else:
            batch_size = dataset_size

        for (batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state,
             batch_outputs, batch_active_entries) in self.gen_epoch(dataset, batch_size=batch_size):
            feed_dict = self.build_feed_dictionary(batch_current_covariates, batch_previous_treatments,
                                                   batch_current_treatments, batch_init_state, batch_outputs,
                                                   batch_active_entries)

            validation_loss, validation_loss_outcomes, validation_loss_treatments = self.sess.run(
                [self.loss, self.loss_outcomes, self.loss_treatments],
                feed_dict=feed_dict)

            validation_losses.append(validation_loss)
            validation_losses_outcomes.append(validation_loss_outcomes)
            validation_losses_treatments.append(validation_loss_treatments)

        validation_loss = np.mean(np.array(validation_losses))
        validation_loss_outcomes = np.mean(np.array(validation_losses_outcomes))
        validation_loss_treatments = np.mean(np.array(validation_losses_treatments))

        return validation_loss, validation_loss_outcomes, validation_loss_treatments

    def get_balancing_reps(self, dataset):
        logging.info("Computing balancing representations.")

        dataset_size = dataset['current_covariates'].shape[0]
        balancing_reps = np.zeros(
            shape=(dataset_size, self.max_sequence_length, self.br_size))

        dataset_size = dataset['current_covariates'].shape[0]
        if (dataset_size > 10000): # Does not fit into memory
            batch_size = 10000
        else:
            batch_size = dataset_size

        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 50
        for (batch_current_covariates, batch_previous_treatments,
             batch_current_treatments, batch_init_state) in self.gen_epoch(dataset, batch_size=batch_size,
                                                                           training_mode=False):
            feed_dict = self.build_feed_dictionary(batch_current_covariates, batch_previous_treatments,
                                                   batch_current_treatments, batch_init_state, training_mode=False)

            # Dropout samples
            total_predictions = np.zeros(
                shape=(batch_size, self.max_sequence_length, self.br_size))

            for sample in range(num_samples):
                br_outputs = self.sess.run(self.balancing_representation, feed_dict=feed_dict)
                br_outputs = np.reshape(br_outputs,
                                        newshape=(-1, self.max_sequence_length, self.br_size))
                total_predictions += br_outputs

            total_predictions /= num_samples

            if (batch_id == num_batches - 1):
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

            batch_id += 1
            balancing_reps[batch_samples] = total_predictions

        return balancing_reps

    def get_predictions(self, dataset):
        logging.info("Performing one-step-ahed prediction.")
        dataset_size = dataset['current_covariates'].shape[0]

        predictions = np.zeros(
            shape=(dataset_size, self.max_sequence_length, self.num_outputs))

        dataset_size = dataset['current_covariates'].shape[0]
        if (dataset_size > 10000):
            batch_size = 10000
        else:
            batch_size = dataset_size

        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 50
        for (batch_current_covariates, batch_previous_treatments,
             batch_current_treatments, batch_init_state) in self.gen_epoch(dataset, batch_size=batch_size,
                                                                           training_mode=False):
            feed_dict = self.build_feed_dictionary(batch_current_covariates, batch_previous_treatments,
                                                   batch_current_treatments, batch_init_state, training_mode=False)

            # Dropout samples
            total_predictions = np.zeros(
                shape=(batch_size, self.max_sequence_length, self.num_outputs))

            for sample in range(num_samples):
                predicted_outputs = self.sess.run(self.predictions, feed_dict=feed_dict)
                predicted_outputs = np.reshape(predicted_outputs,
                                               newshape=(-1, self.max_sequence_length, self.num_outputs))
                total_predictions += predicted_outputs

            total_predictions /= num_samples

            if (batch_id == num_batches - 1):
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

            batch_id += 1
            predictions[batch_samples] = total_predictions

        return predictions

    def get_autoregressive_sequence_predictions(self, test_data, data_map, encoder_states, encoder_outputs,
                                                projection_horizon):
        logging.info("Performing multi-step ahead prediction.")
        current_treatments = data_map['current_treatments']
        previous_treatments = data_map['previous_treatments']

        sequence_lengths = test_data['sequence_lengths'] - 1
        num_patient_points = current_treatments.shape[0]

        current_dataset = dict()
        current_dataset['current_covariates'] = np.zeros(shape=(num_patient_points, projection_horizon,
                                                                test_data['current_covariates'].shape[-1]))
        current_dataset['previous_treatments'] = np.zeros(shape=(num_patient_points, projection_horizon,
                                                                 test_data['previous_treatments'].shape[-1]))
        current_dataset['current_treatments'] = np.zeros(shape=(num_patient_points, projection_horizon,
                                                                test_data['current_treatments'].shape[-1]))
        current_dataset['init_state'] = np.zeros((num_patient_points, encoder_states.shape[-1]))

        predicted_outputs = np.zeros(shape=(num_patient_points, projection_horizon,
                                            test_data['outputs'].shape[-1]))

        for i in range(num_patient_points):
            seq_length = int(sequence_lengths[i])
            current_dataset['init_state'][i] = encoder_states[i, seq_length - 1]
            current_dataset['current_covariates'][i, 0, 0] = encoder_outputs[i, seq_length - 1]
            current_dataset['previous_treatments'][i] = previous_treatments[i,
                                                        seq_length - 1:seq_length + projection_horizon - 1, :]
            current_dataset['current_treatments'][i] = current_treatments[i, seq_length:seq_length + projection_horizon,
                                                       :]

        for t in range(0, projection_horizon):
            print(t)
            predictions = self.get_predictions(current_dataset)
            for i in range(num_patient_points):
                predicted_outputs[i, t] = predictions[i, t]
                if (t < projection_horizon - 1):
                    current_dataset['current_covariates'][i, t + 1, 0] = predictions[i, t, 0]

        test_data['predicted_outcomes'] = predicted_outputs

        return predicted_outputs

    def compute_loss_treatments_one_hot(self, target_treatments, treatment_predictions, active_entries):
        treatment_predictions = tf.reshape(treatment_predictions, [-1, self.max_sequence_length, self.num_treatments])
        cross_entropy_loss = tf.reduce_sum(
            (- target_treatments * tf.log(treatment_predictions + 1e-8)) * active_entries) \
                             / tf.reduce_sum(active_entries)
        return cross_entropy_loss

    def compute_loss_predictions(self, outputs, predictions, active_entries):
        predictions = tf.reshape(predictions, [-1, self.max_sequence_length, self.num_outputs])
        mse_loss = tf.reduce_sum(tf.square(outputs - predictions) * active_entries) \
                   / tf.reduce_sum(active_entries)

        return mse_loss

    def evaluate_predictions(self, dataset):
        predictions = self.get_predictions(dataset)
        unscaled_predictions = predictions * dataset['output_stds'] \
                               + dataset['output_means']
        unscaled_predictions = np.reshape(unscaled_predictions,
                                          newshape=(-1, self.max_sequence_length, self.num_outputs))
        unscaled_outputs = dataset['unscaled_outputs']
        active_entries = dataset['active_entries']

        mse = self.get_mse_at_follow_up_time(unscaled_predictions, unscaled_outputs, active_entries)
        mean_mse = np.mean(mse)
        return mean_mse, mse

    def get_mse_at_follow_up_time(self, prediction, output, active_entires):
        mses = np.sum(np.sum((prediction - output) ** 2 * active_entires, axis=-1), axis=0) \
               / active_entires.sum(axis=0).sum(axis=-1)
        return mses

    def get_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return optimizer

    def compute_sequence_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)

        return length

    def save_network(self, tf_session, model_dir, checkpoint_name):
        saver = tf.train.Saver(max_to_keep=100000)
        vars = 0
        for v in tf.global_variables():
            vars += np.prod(v.get_shape().as_list())

        save_path = saver.save(tf_session, os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name)))
        logging.info("Model saved to: {0}".format(save_path))

    def load_network(self, tf_session, model_dir, checkpoint_name):
        load_path = os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name))
        logging.info('Restoring model from {0}'.format(load_path))

        saver = tf.train.Saver()
        saver.restore(tf_session, load_path)
