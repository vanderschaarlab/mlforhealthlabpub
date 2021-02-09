'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.python.ops import rnn

from utils.predictive_checks_utils import compute_test_statistic_all_timesteps
from utils.rnn_utils import AutoregressiveLSTMCell, compute_sequence_length

class FactorModel:
    def __init__(self, params, hyperparams):
        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        self.num_confounders = params['num_confounders']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']

        self.rnn_hidden_units = hyperparams['rnn_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.rnn_keep_prob = hyperparams['rnn_keep_prob']

        tf.reset_default_graph()
        self.previous_covariates = tf.placeholder(tf.float32, [None, self.max_sequence_length - 1, self.num_covariates])
        self.previous_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length - 1, self.num_treatments])
        self.trainable_init_input = tf.get_variable(name='trainable_init_input',
                                                    shape=[self.batch_size, 1,
                                                           self.num_covariates + self.num_treatments], trainable=True)

        self.current_covariates = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_covariates])
        self.target_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_treatments])

    def build_confounders(self, trainable_state=True):
        previous_covariates_and_treatments = tf.concat([self.previous_covariates, self.previous_treatments],
                                                       axis=-1)
        self.rnn_input = tf.concat([self.trainable_init_input, previous_covariates_and_treatments], axis=1)
        self.sequence_length = compute_sequence_length(self.rnn_input)

        rnn_cell = DropoutWrapper(LSTMCell(self.rnn_hidden_units, state_is_tuple=False),
                                  output_keep_prob=self.rnn_keep_prob,
                                  state_keep_prob=self.rnn_keep_prob, variational_recurrent=True,
                                  dtype=tf.float32)

        autoregressive_cell = AutoregressiveLSTMCell(rnn_cell, self.num_confounders)

        if trainable_state:
            init_state = tf.get_variable(name='init_cell',
                                         shape=[self.batch_size, autoregressive_cell.state_size],
                                         trainable=True)
        else:
            init_state = autoregressive_cell.zero_state(self.batch_size, dtype=tf.float32)

        rnn_output, _ = rnn.dynamic_rnn(
            autoregressive_cell,
            self.rnn_input,
            initial_state=init_state,
            dtype=tf.float32,
            sequence_length=self.sequence_length)

        # Flatten to apply same weights to all time steps.
        rnn_output = tf.reshape(rnn_output, [-1, self.num_confounders])

        hidden_confounders = rnn_output
        covariates = tf.reshape(self.current_covariates, [-1, self.num_covariates])
        self.multitask_input = tf.concat([covariates, hidden_confounders], axis=-1)

        self.hidden_confounders = tf.reshape(hidden_confounders,
                                             [-1, self.max_sequence_length, self.num_confounders])

    def build_treatment_assignments(self):
        self.treatment_prob_predictions = dict()
        for treatment in range(self.num_treatments):
            treatment_network_layer = tf.layers.dense(self.multitask_input, self.fc_hidden_units,
                                                      name='treatment_network_%s' % str(treatment),
                                                      activation=tf.nn.leaky_relu)

            treatment_output = tf.layers.dense(treatment_network_layer, 1, activation=tf.nn.sigmoid,
                                               name='treatment_output_%s' % str(treatment))

            self.treatment_prob_predictions[treatment] = treatment_output

        self.treatment_prob_predictions = tf.concat(list(self.treatment_prob_predictions.values()), axis=-1)

        return self.treatment_prob_predictions

    def build_network(self):
        self.build_confounders()
        self.treatment_prob_predictions = self.build_treatment_assignments()
        return self.treatment_prob_predictions

    def gen_epoch(self, dataset):
        dataset_size = dataset['previous_covariates'].shape[0]
        num_batches = int(dataset_size / self.batch_size) + 1

        for i in range(num_batches):
            if (i == num_batches - 1):
                batch_samples = range(dataset_size - self.batch_size, dataset_size)
            else:
                batch_samples = range(i * self.batch_size, (i + 1) * self.batch_size)

            batch_previous_covariates = dataset['previous_covariates'][batch_samples, :, :]
            batch_previous_treatments = dataset['previous_treatments'][batch_samples, :, :]
            batch_current_covariates = dataset['covariates'][batch_samples, :, :]
            batch_target_treatments = dataset['treatments'][batch_samples, :, :].astype(np.int32)

            yield (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
                   batch_target_treatments)

    def eval_network(self, dataset):
        validation_losses = []

        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
             batch_target_treatments) in self.gen_epoch(dataset):
            feed_dict = self.build_feed_dictionary(batch_previous_covariates, batch_previous_treatments,
                                                   batch_current_covariates, batch_target_treatments)
            validation_loss= self.sess.run([self.loss], feed_dict=feed_dict)
            validation_losses.append(validation_loss)

        validation_loss = np.mean(np.array(validation_losses))

        return validation_loss

    def compute_test_statistic(self, num_samples, target_treatments, feed_dict, predicted_mask):
        test_statistic = np.zeros(shape=(self.max_sequence_length,))

        for sample_idx in range(num_samples):
            [treatment_probability] = self.sess.run(
                [self.treatment_prob_predictions], feed_dict=feed_dict)

            treatment_probability = np.reshape(treatment_probability, newshape=(
                self.batch_size, self.max_sequence_length, self.num_treatments))

            test_statistic_sequence = compute_test_statistic_all_timesteps(target_treatments,
                                                                           treatment_probability,
                                                                           self.max_sequence_length, predicted_mask)
            test_statistic += test_statistic_sequence

        test_statistic = test_statistic / num_samples

        return test_statistic

    def eval_predictive_checks(self, dataset):
        num_replications = 50
        num_samples = 50

        p_values_over_time = np.zeros(shape=(self.max_sequence_length,))

        steps = 0

        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
             batch_target_treatments) in self.gen_epoch(dataset):
            feed_dict = self.build_feed_dictionary(batch_previous_covariates, batch_previous_treatments,
                                                   batch_current_covariates, batch_target_treatments)

            mask = tf.sign(tf.reduce_max(tf.abs(self.rnn_input), axis=2))
            [seq_lenghts, predicted_mask] = self.sess.run([self.sequence_length, mask], feed_dict=feed_dict)
            steps = steps + 1

            """ Compute test statistics for replicas """
            test_statistic_replicas = np.zeros(shape=(num_replications, self.max_sequence_length))
            for replication_idx in range(num_replications):
                [treatment_replica, treatment_prob_pred] = self.sess.run(
                    [self.treatment_realizations, self.treatment_prob_predictions], feed_dict=feed_dict)

                treatment_replica = np.reshape(treatment_replica, newshape=(
                    self.batch_size, self.max_sequence_length, self.num_treatments))

                test_statistic_replicas[replication_idx] = self.compute_test_statistic(num_samples, treatment_replica,
                                                                                       feed_dict, predicted_mask)

            """ Compute test statistic for target """
            test_statistic_target = self.compute_test_statistic(num_samples, batch_target_treatments, feed_dict,
                                                                predicted_mask)

            probability = np.mean(np.less(test_statistic_replicas, test_statistic_target).astype(np.int32), axis=0)
            p_values_over_time += probability

        p_values_over_time = p_values_over_time / steps
        return p_values_over_time

    def train(self, dataset_train, dataset_val, verbose=False):
        self.treatment_prob_predictions = self.build_network()
        self.treatment_realizations = tf.distributions.Bernoulli(probs=self.treatment_prob_predictions).sample()

        self.loss = self.compute_loss(self.target_treatments, self.treatment_prob_predictions)
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

        for epoch in tqdm(range(self.num_epochs)):
            for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
                 batch_target_treatments) in self.gen_epoch(dataset_train):
                feed_dict = self.build_feed_dictionary(batch_previous_covariates, batch_previous_treatments,
                                                       batch_current_covariates, batch_target_treatments)

                _, training_loss = self.sess.run([optimizer, self.loss], feed_dict=feed_dict)

            if (verbose):
                logging.info(
                    "Epoch {} out of {}: Summary| Training loss = {}".format(
                        (epoch + 1), self.num_epochs, training_loss))

                if ((epoch + 1) % 100 == 0):
                    validation_loss = self.eval_network(dataset_val)
                    logging.info(
                        "Epoch {} out of {}: Summary| Validation loss = {}".format(epoch, self.num_epochs, validation_loss))

    def build_feed_dictionary(self, batch_previous_covariates, batch_previous_treatments,
                              batch_current_covariates, batch_target_treatments):
        feed_dict = {self.previous_covariates: batch_previous_covariates,
                     self.previous_treatments: batch_previous_treatments,
                     self.current_covariates: batch_current_covariates,
                     self.target_treatments: batch_target_treatments}
        return feed_dict

    def compute_loss(self, target_treatments, treatment_predictions):
        target_treatments_reshape = tf.reshape(target_treatments, [-1, self.num_treatments])

        mask = tf.sign(tf.reduce_max(tf.abs(self.rnn_input), axis=2))
        flat_mask = tf.reshape(mask, [-1, 1])

        cross_entropy = - tf.reduce_sum((target_treatments_reshape * tf.log(
            tf.clip_by_value(treatment_predictions, 1e-10, 1.0)) + (1 - target_treatments_reshape) * (tf.log(
            tf.clip_by_value(1 - treatment_predictions, 1e-10, 1.0)))) * flat_mask, axis=0)

        self.mask = mask
        cross_entropy /= tf.reduce_sum(tf.cast(self.sequence_length, tf.float32), axis=0)

        return tf.reduce_mean(cross_entropy)

    def get_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return optimizer

    def compute_hidden_confounders(self, dataset):
        dataset_size = dataset['covariates'].shape[0]

        hidden_confounders = np.zeros(
            shape=(dataset_size, self.max_sequence_length, self.num_confounders))

        num_batches = int(dataset_size / self.batch_size) + 1

        batch_id = 0
        num_samples = 50
        for (batch_previous_covariates, batch_previous_treatments, batch_current_covariates,
             batch_target_treatments) in self.gen_epoch(dataset):
            feed_dict = self.build_feed_dictionary(batch_previous_covariates, batch_previous_treatments,
                                                   batch_current_covariates, batch_target_treatments)
            total_predicted_hidden_confounders = np.zeros(
                shape=(self.batch_size, self.max_sequence_length, self.num_confounders))

            for sample in range(num_samples):
                predicted_hidden_confounders, predicted_treatment_probs = self.sess.run(
                    [self.hidden_confounders, self.treatment_prob_predictions], feed_dict=feed_dict)

                total_predicted_hidden_confounders += predicted_hidden_confounders
            total_predicted_hidden_confounders /= num_samples

            if (batch_id == num_batches - 1):
                batch_samples = range(dataset_size - self.batch_size, dataset_size)
            else:
                batch_samples = range(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)

            batch_id += 1
            hidden_confounders[batch_samples] = total_predicted_hidden_confounders

        return hidden_confounders
