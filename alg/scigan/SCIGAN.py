# Copyright (c) 2020, Ioana Bica

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.model_utils import equivariant_layer, invariant_layer, sample_dosages, sample_X, sample_Z


class SCIGAN_Model:
    def __init__(self, params):
        self.num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        self.export_dir = params['export_dir']

        self.h_dim = params['h_dim']
        self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        self.alpha = params['alpha']
        self.num_dosage_samples = params['num_dosage_samples']

        self.size_z = self.num_treatments * self.num_dosage_samples
        self.num_outcomes = self.num_treatments * self.num_dosage_samples

        tf.reset_default_graph()
        tf.random.set_random_seed(10)

        # Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name='input_features')
        # Treatment (T) - one-hot encoding for the treatment
        self.T = tf.placeholder(tf.float32, shape=[None, self.num_treatments], name='input_treatment')
        # Dosage (D)
        self.D = tf.placeholder(tf.float32, shape=[None, 1], name='input_dosage')
        # Dosage samples (D)
        self.Treatment_Dosage_Samples = tf.placeholder(tf.float32,
                                                       shape=[None, self.num_treatments, self.num_dosage_samples],
                                                       name='input_treatment_dosage_samples')
        # Treatment dosage mask to indicate the factual outcome
        self.Treatment_Dosage_Mask = tf.placeholder(tf.float32,
                                                    shape=[None, self.num_treatments, self.num_dosage_samples],
                                                    name='input_treatment_dosage_mask')
        # Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
        # Random Noise (G)
        self.Z_G = tf.placeholder(tf.float32, shape=[None, self.size_z], name='input_noise')

    def generator(self, x, y, t, d, z, treatment_dosage_samples):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            inputs = tf.concat(axis=1, values=[x, y, t, d, z])
            G_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')
            G_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                treatment_dosages = treatment_dosage_samples[:, treatment]
                treatment_dosages = tf.reshape(treatment_dosages, shape=(-1, 1))
                G_shared_expand = tf.reshape(tf.tile(G_shared, multiples=[1, self.num_dosage_samples]),
                                             shape=(-1, self.h_dim))
                input_counterfactual_dosage = tf.concat(axis=1, values=[G_shared_expand, treatment_dosages])

                treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_1_%s' % str(treatment), reuse=tf.AUTO_REUSE)

                treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_2_%s' % str(treatment), reuse=tf.AUTO_REUSE)

                treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                          name='treatment_output_%s' % str(treatment),
                                                          reuse=tf.AUTO_REUSE)

                dosage_counterfactuals = tf.reshape(treatment_dosage_output, shape=(-1, self.num_dosage_samples))

                G_treatment_dosage_outcomes[treatment] = dosage_counterfactuals

            G_logits = tf.concat(list(G_treatment_dosage_outcomes.values()), axis=1)
            G_logits = tf.reshape(G_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return G_logits, G_treatment_dosage_outcomes

    def dosage_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask,
                             G_treatment_dosage_outcomes):

        with tf.variable_scope('dosage_discriminator', reuse=tf.AUTO_REUSE):
            patient_features_representation = tf.expand_dims(tf.layers.dense(x, self.h_dim, activation=tf.nn.elu),
                                                             axis=1)
            D_dosage_outcomes = dict()
            for treatment in range(self.num_treatments):
                treatment_mask = treatment_dosage_mask[:, treatment]
                treatment_dosages = treatment_dosage_samples[:, treatment]
                G_treatment_dosage_outcomes[treatment] = treatment_mask * y + (1 - treatment_mask) * \
                                                         G_treatment_dosage_outcomes[treatment]

                dosage_samples = tf.expand_dims(treatment_dosages, axis=-1)
                dosage_potential_outcomes = tf.expand_dims(G_treatment_dosage_outcomes[treatment], axis=-1)

                inputs = tf.concat(axis=-1, values=[dosage_samples, dosage_potential_outcomes])
                D_h1 = tf.nn.elu(equivariant_layer(inputs, self.h_inv_eqv_dim, layer_id=1,
                                                   treatment_id=treatment) + patient_features_representation)
                D_h2 = tf.nn.elu(equivariant_layer(D_h1, self.h_inv_eqv_dim, layer_id=2, treatment_id=treatment))
                D_logits_treatment = tf.layers.dense(D_h2, 1, activation=None,
                                                     name='treatment_output_%s' % str(treatment))

                D_dosage_outcomes[treatment] = tf.squeeze(D_logits_treatment, axis=-1)

            D_dosage_logits = tf.concat(list(D_dosage_outcomes.values()), axis=-1)
            D_dosage_logits = tf.reshape(D_dosage_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return D_dosage_logits, D_dosage_outcomes

    def treatment_discriminator(self, x, y, treatment_dosage_samples, treatment_dosage_mask,
                                G_treatment_dosage_outcomes):
        with tf.variable_scope('treatment_discriminator', reuse=tf.AUTO_REUSE):
            patient_features_representation = tf.layers.dense(x, self.h_dim, activation=tf.nn.elu)

            D_treatment_outcomes = dict()
            for treatment in range(self.num_treatments):
                treatment_mask = treatment_dosage_mask[:, treatment]
                treatment_dosages = treatment_dosage_samples[:, treatment]
                G_treatment_dosage_outcomes[treatment] = treatment_mask * y + (1 - treatment_mask) * \
                                                         G_treatment_dosage_outcomes[treatment]

                dosage_samples = tf.expand_dims(treatment_dosages, axis=-1)
                dosage_potential_outcomes = tf.expand_dims(G_treatment_dosage_outcomes[treatment], axis=-1)

                inputs = tf.concat(axis=-1, values=[dosage_samples, dosage_potential_outcomes])
                D_treatment_rep = invariant_layer(x=inputs, h_dim=self.h_inv_eqv_dim, treatment_id=treatment)

                D_treatment_outcomes[treatment] = D_treatment_rep

            D_treatment_representations = tf.concat(list(D_treatment_outcomes.values()), axis=-1)
            D_shared_representation = tf.concat([D_treatment_representations, patient_features_representation], axis=-1)

            D_treatment_rep_hidden = tf.layers.dense(D_shared_representation, self.h_dim, activation=tf.nn.elu,
                                                     name='rep_all',
                                                     reuse=tf.AUTO_REUSE)

            D_treatment_logits = tf.layers.dense(D_treatment_rep_hidden, self.num_treatments, activation=None,
                                                 name='output_all',
                                                 reuse=tf.AUTO_REUSE)

        return D_treatment_logits

    def inference(self, x, treatment_dosage_samples):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            inputs = x
            I_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')

            I_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                dosage_counterfactuals = dict()
                treatment_dosages = treatment_dosage_samples[:, treatment]

                for index in range(self.num_dosage_samples):
                    dosage_sample = tf.expand_dims(treatment_dosages[:, index], axis=-1)
                    input_counterfactual_dosage = tf.concat(axis=1, values=[I_shared, dosage_sample])

                    treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_1_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_2_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                              name='treatment_output_%s' % str(treatment),
                                                              reuse=tf.AUTO_REUSE)

                    dosage_counterfactuals[index] = treatment_dosage_output

                I_treatment_dosage_outcomes[treatment] = tf.concat(list(dosage_counterfactuals.values()), axis=-1)

            I_logits = tf.concat(list(I_treatment_dosage_outcomes.values()), axis=1)
            I_logits = tf.reshape(I_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return I_logits, I_treatment_dosage_outcomes

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):
        # 1. Counterfactual generator
        G_logits, G_treatment_dosage_outcomes = self.generator(x=self.X, y=self.Y, t=self.T, d=self.D,
                                                               z=self.Z_G,
                                                               treatment_dosage_samples=self.Treatment_Dosage_Samples)

        # 2. Dosage discriminator
        D_dosage_logits, D_dosage_outcomes = self.dosage_discriminator(x=self.X, y=self.Y,
                                                                       treatment_dosage_samples=self.Treatment_Dosage_Samples,
                                                                       treatment_dosage_mask=self.Treatment_Dosage_Mask,
                                                                       G_treatment_dosage_outcomes=G_treatment_dosage_outcomes)
        # 3. Treatment discriminator
        D_treatment_logits = self.treatment_discriminator(x=self.X, y=self.Y,
                                                          treatment_dosage_samples=self.Treatment_Dosage_Samples,
                                                          treatment_dosage_mask=self.Treatment_Dosage_Mask,
                                                          G_treatment_dosage_outcomes=G_treatment_dosage_outcomes)

        # 4. Inference network
        I_logits, I_treatment_dosage_outcomes = self.inference(self.X, self.Treatment_Dosage_Samples)

        G_outcomes = tf.identity(G_logits, name='generator_outcomes')
        I_outcomes = tf.identity(I_logits, name="inference_outcomes")

        # 1. Dosage discriminator loss
        num_examples = tf.cast(self.batch_size, dtype=tf.int64)
        factual_treatment_idx = tf.argmax(self.T, axis=1)
        idx = tf.stack([tf.range(num_examples), factual_treatment_idx], axis=-1)

        D_dosage_logits_factual_treatment = tf.gather_nd(D_dosage_logits, idx)
        Dosage_Mask = tf.gather_nd(self.Treatment_Dosage_Mask, idx)

        D_dosage_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=Dosage_Mask, logits=D_dosage_logits_factual_treatment))

        # 2. Treatment discriminator loss
        D_treatment_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reduce_max(self.Treatment_Dosage_Mask, axis=-1),
                                                    logits=D_treatment_logits))

        # 3. Overall discriminator loss
        D_combined_prob = tf.nn.sigmoid(D_dosage_logits) * tf.nn.sigmoid(
            tf.tile(tf.expand_dims(D_treatment_logits, axis=-1), multiples=[1, 1, self.num_dosage_samples]))

        D_combined_loss = tf.reduce_mean(
            self.Treatment_Dosage_Mask * -tf.log(D_combined_prob + 1e-7) + (1.0 - self.Treatment_Dosage_Mask) * -tf.log(
                1.0 - D_combined_prob + 1e-7))

        # 4. Generator loss
        G_loss_GAN = -D_combined_loss
        G_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * G_logits, axis=[1, 2]), axis=-1)
        G_loss_R = tf.reduce_mean((self.Y - G_logit_factual) ** 2)
        G_loss = self.alpha * tf.sqrt(G_loss_R) + G_loss_GAN

        # 4. Inference loss
        I_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * I_logits, axis=[1, 2]), axis=-1)
        I_loss1 = tf.reduce_mean((G_logits - I_logits) ** 2)
        I_loss2 = tf.reduce_mean((self.Y - I_logit_factual) ** 2)
        I_loss = tf.sqrt(I_loss1) + tf.sqrt(I_loss2)

        theta_G = tf.trainable_variables(scope='generator')
        theta_D_dosage = tf.trainable_variables(scope='dosage_discriminator')
        theta_D_treatment = tf.trainable_variables(scope='treatment_discriminator')
        theta_I = tf.trainable_variables(scope='inference')

        # %% Solver
        G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=theta_G)
        D_dosage_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_dosage_loss, var_list=theta_D_dosage)
        D_treatment_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_treatment_loss,
                                                                                  var_list=theta_D_treatment)
        I_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(I_loss, var_list=theta_I)

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

        # Iterations
        print("Training SCIGAN generator and discriminator.")
        for it in tqdm(range(5000)):
            for kk in range(2):
                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, G_loss_curr, G_logits_curr, G_logit_factual_curr = self.sess.run(
                    [G_solver, G_loss, G_logits, G_logit_factual],
                    feed_dict={self.X: X_mb, self.T: treatment_one_hot, self.D: D_mb[:, np.newaxis],
                               self.Treatment_Dosage_Samples: treatment_dosage_samples,
                               self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb,
                               self.Z_G: Z_G_mb})

            for kk in range(1):
                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, D_dosage_loss_curr = self.sess.run([D_dosage_solver, D_dosage_loss],
                                                      feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                                 self.D: D_mb[:, np.newaxis],
                                                                 self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                                 self.Treatment_Dosage_Mask: treatment_dosage_mask,
                                                                 self.Y: Y_mb, self.Z_G: Z_G_mb})

                idx_mb = sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
                D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
                Z_G_mb = sample_Z(self.batch_size, self.size_z)

                treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                          self.num_dosage_samples)
                factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
                treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

                treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                        self.num_dosage_samples])
                treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
                treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

                _, D_treatment_loss_curr = self.sess.run([D_treatment_solver, D_treatment_loss],
                                                         feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                                    self.D: D_mb[:, np.newaxis],
                                                                    self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                                    self.Treatment_Dosage_Mask: treatment_dosage_mask,
                                                                    self.Y: Y_mb, self.Z_G: Z_G_mb})

            # %% Debugging
            if it % 1000 == 0 and verbose:
                D_treatment_loss_curr, D_dosage_loss_curr, G_loss_curr, = self.sess.run(
                    [D_treatment_loss, D_dosage_loss, G_loss],
                    feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                               self.D: D_mb[:, np.newaxis],
                               self.Treatment_Dosage_Samples: treatment_dosage_samples,
                               self.Treatment_Dosage_Mask: treatment_dosage_mask,
                               self.Y: Y_mb, self.Z_G: Z_G_mb})

                print('Iter: {}'.format(it))
                print('D_loss_treatments: {:.4}'.format((D_treatment_loss_curr)))
                print('D_loss_dosages: {:.4}'.format((D_dosage_loss_curr)))
                print('G_loss: {:.4}'.format((G_loss_curr)))
                print()

        # Train Inference Network
        print("Training inference network.")
        for it in tqdm(range(10000)):
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
            D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
            Z_G_mb = sample_Z(self.batch_size, self.size_z)

            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments,
                                                      self.num_dosage_samples)
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

            treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                    self.num_dosage_samples])
            treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
            treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

            _, I_loss_curr = self.sess.run([I_solver, I_loss],
                                           feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                      self.D: D_mb[:, np.newaxis],
                                                      self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                      self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb,
                                                      self.Z_G: Z_G_mb})

            # %% Debugging
            if it % 1000 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('I_loss: {:.4}'.format((I_loss_curr)))
                print()

        tf.compat.v1.saved_model.simple_save(self.sess, export_dir=self.export_dir,
                                             inputs={'input_features': self.X,
                                                     'input_treatment_dosage_samples': self.Treatment_Dosage_Samples},
                                             outputs={'inference_outcome': I_logits})
