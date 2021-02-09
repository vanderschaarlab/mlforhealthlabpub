import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time


import utils_network as utils

_EPSILON = 1e-08



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


class DeepTPC:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name
        
        # INPUT DIMENSIONS
        self.num_Event       = input_dims['num_Event']
        self.max_length      = input_dims['max_length']
        self.num_Feature     = input_dims['num_Feature'] #static features


        # RNN
        self.h_dim1          = network_settings['h_dim1'] #RNN hidden nodes
        self.num_layers_RNN  = network_settings['num_layers_RNN']
        self.RNN_type        = network_settings['RNN_type']
        self.RNN_active_fn   = network_settings['RNN_active_fn']

        # FC-NET
        self.h_dim2          = network_settings['h_dim2'] #FC hidden nodes

        # CLUSTERING
        self.z_dim           = self.h_dim1 * self.num_layers_RNN
        self.L               = network_settings['L'] # points for trapazoid approx. on the distance in the output space
        self.delta_range_    = network_settings['delta_range']

        # OTHERS
        self.initial_W       = network_settings['initial_W']

        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')
            self.lr_rate     = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob   = tf.placeholder(tf.float32, name='keep_probability')

            self.K           = tf.placeholder(tf.int32, [], name='num_Cluster')

            self.M           = tf.placeholder(tf.float32, shape=[None, self.max_length, self.num_Event], name='M_onehot')  
            self.D           = tf.placeholder(tf.float32, shape=[None, self.max_length, 1], name='delta')
            self.X           = tf.placeholder(tf.float32, shape=[None, self.num_Feature], name='X')
            self.Mask        = tf.placeholder(tf.float32, shape=[None, self.max_length], name='rnn_mask')

            self.MU          = tf.placeholder(tf.float32, [None, self.z_dim], name='MU') #this will become [K, z_dim]
            self.S           = tf.placeholder(tf.int64, [None], name='S')
            S_one_hot   = tf.one_hot(self.S, self.K, name='S_one_hot')
            
            self.delta_range = tf.placeholder(tf.float32, [self.L], name='delta_range')

            # LOSS PARAMETERS
            self.alpha      = tf.placeholder(tf.float32, name = 'alpha')
            self.beta       = tf.placeholder(tf.float32, name = 'beta')
            self.beta_ms    = tf.placeholder(tf.float32, name = 'beta_ms', shape=[self.num_Event - 1]) #(set [1, ..., 1] as a default)
            self.gamma      = tf.placeholder(tf.float32, name = 'gamma')


            # DECLARE FUNCTIONS FOR NETWORK CONSTRUCTION
            def prediction_network_softplus(h, delta, reuse=tf.AUTO_REUSE): #version 0
                with tf.variable_scope('prediction_net', reuse=reuse):
                    tmp = tf.contrib.layers.fully_connected(inputs=tf.concat([h, delta], axis=1), num_outputs=self.h_dim2, activation_fn=None) #layer1
                    tmp = tf.nn.dropout(tmp, keep_prob=self.keep_prob)
                    tmp = tf.contrib.layers.fully_connected(inputs=tmp, num_outputs=self.h_dim2, activation_fn=tf.nn.relu) #layer2
                    tmp = tf.nn.dropout(tmp, keep_prob=self.keep_prob)
                    tmp = tf.contrib.layers.fully_connected(inputs=tmp, num_outputs=self.num_Event, activation_fn=None)                          #layer2
                    out = tf.nn.softplus(tmp)
                return out

            ### DEFINE LOOP FUNCTION FOR RAW_RNN w/ TEMPORAL ATTENTION
            def loop_fn_MPP(time, cell_output, cell_state, loop_state):   
                emit_output = cell_output 

                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                    next_loop_state = (tf.TensorArray(size=self.max_length, dtype=tf.float32),  #lambda(t_{j})
                                       tf.TensorArray(size=self.max_length, dtype=tf.float32),  #lambda(t_{j-1})
                                       tf.TensorArray(size=self.max_length, dtype=tf.float32))  #hidden states (h_{j})

                else:
                    next_cell_state = cell_state
                    tmp_h = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type, BiRNN=None)

                    def fn_time_last(): #the last lambda_curr will not be included in the loss function (thus, time-1 is applied to remove the error)
                        d_next = tf.reshape(inputs_ta.read(time-1)[:,0], shape=[-1, 1]) #to prevent indexing error
                        l_next  = prediction_network_softplus(tmp_h, d_next)
                        l_curr  = prediction_network_softplus(tmp_h, tf.zeros_like(d_next))            
                        return l_curr, l_next

                    def fn_time_others(): #the last lambda_curr will not be included in the loss function (thus, time-1 is applied to remove the error)
                        d_next = tf.reshape(inputs_ta.read(time)[:,0], shape=[-1, 1]) #to prevent indexing error
                        l_next  = prediction_network_softplus(tmp_h, d_next)
                        l_curr  = prediction_network_softplus(tmp_h, tf.zeros_like(d_next))            
                        return l_curr, l_next

                    l_curr, l_next = tf.cond(
                        tf.equal(time, self.max_length), lambda: fn_time_last(), lambda: fn_time_others()
                    )

                    next_loop_state = (loop_state[0].write(time-1, l_next),   # save lambda(t_{j})
                                       loop_state[1].write(time-1, l_curr),      # save lambda(t_{j-1})
                                       loop_state[2].write(time-1, tmp_h))            # save all the h_ins

                elements_finished = (time >= seq_length)

                #this gives the break-point (no more recurrence after the max_length)
                finished = tf.reduce_all(elements_finished)


                def fn_input_embedding():
                    embedding = tf.concat([inputs_ta.read(time), self.X], axis=1)
            #         embedding = tf.nn.dropout(embedding, keep_prob=keep_prob)
                    embedding = tf.contrib.layers.fully_connected(inputs=embedding, num_outputs=self.h_dim2, activation_fn=tf.nn.relu)

                    return embedding

                next_input = tf.cond(
                    finished, lambda: tf.zeros([self.mb_size, self.h_dim2], dtype=tf.float32), lambda: fn_input_embedding()
                )

                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)


        ### INPUTS
        inputs = tf.concat([self.D, self.M], axis=2, name='inputs')

        inputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=self.max_length,
            clear_after_read=False
        ).unstack(_transpose_batch_time(inputs), name='inputs_ta')

        seq_length = get_seq_length(inputs)


        ### RNNS
        cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, self.RNN_type, self.RNN_active_fn)
        _, rnn_final_state, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn_MPP)


        next_lambdas    = _transpose_batch_time(loop_state_ta[0].stack())
        curr_lambdas    = _transpose_batch_time(loop_state_ta[1].stack())
        rnn_states      = _transpose_batch_time(loop_state_ta[2].stack())


        self.Z = tf.reduce_sum(rnn_states * tf.tile(tf.expand_dims(self.Mask, axis=2), [1,1, self.z_dim]), axis=1)


        '''
            AFTER PUTTING (m_{0}, t_{0})
                - m_{0} = [1,0,0,...] (auxilary event type)
                - t_{0} = 0
                - Thus, no need to consider the first event issue i.e., putting an additional loss function for t=1
                - Instead, m=0, t=0 (i.e., the first index of event and timing) is not considered.
        '''
        tmp_MLE1 = tf.reduce_sum(tf.reduce_sum(tf.log(next_lambdas[:, :-1, 1:] + 1e-8) * self.M[:,1:,1:], axis=2), axis=1)

        #do not consider m=0 (this is indicator for BOS)
        for m in range(1, self.num_Event):
            if m == 1:
                tmp_MLE2 =  tf.reduce_sum(1/2 * (next_lambdas[:, :-1, m] + curr_lambdas[:, :-1, m]) * self.D[:, 1:, 0], axis=1)
            else:
                tmp_MLE2 += tf.reduce_sum(1/2 * (next_lambdas[:, :-1, m] + curr_lambdas[:, :-1, m]) * self.D[:, 1:, 0], axis=1)

        self.loss_MLE = - tf.reduce_mean(tmp_MLE1 - tmp_MLE2)


        ### LOSS - CLUSTERING

        # DISTANCE IN THE LATENT SPACE
        Z_expanded      = tf.tile(tf.expand_dims(self.Z, axis=1), [1, self.K, 1])     #[None, num_Cluster, 2]
        MU_expanded     = tf.tile(tf.expand_dims(self.MU, axis=0), [self.mb_size, 1, 1])        #[None, num_Cluster, 2]
        dist_z_expanded = tf.reduce_sum((Z_expanded - MU_expanded)**2, axis=2) #[None, num_Cluster]


        dist_z_homo     = tf.reduce_sum(dist_z_expanded * S_one_hot, axis=1) #[None]
        dist_z_hetero   = tf.reduce_sum(dist_z_expanded * (1. - S_one_hot), axis=1) #[None]

        self.dist_z_homo     = tf.reduce_mean(dist_z_homo, axis=0)
        self.dist_z_hetero   = tf.reduce_mean(dist_z_hetero, axis=0)


        # DISTANCE IN THE OUTPUT SPACE (LAMBDA)
        Y    = []
        PSI  = []
        for l in range(self.L):
            tmp_d1 = self.delta_range[l] * tf.ones_like(tf.reshape(self.Z[:,0], shape=[-1, 1]))
            tmp_d2 = self.delta_range[l] * tf.ones_like(tf.reshape(self.MU[:,0], shape=[-1, 1]))

            with tf.variable_scope('rnn', reuse=True):
                Y.append(prediction_network_softplus(self.Z, tmp_d1))
                PSI.append(prediction_network_softplus(self.MU, tmp_d2))


        self.Y_stacked = tf.stack(Y, axis=2)
        self.PSI_stacked = tf.stack(PSI, axis=2)

        Y_stacked_expanded     = tf.tile(tf.expand_dims(self.Y_stacked, axis=1), [1, self.K, 1, 1])     #[None, num_Cluster, num_Event, L]
        PSI_stacked_expanded   = tf.tile(tf.expand_dims(self.PSI_stacked, axis=0), [self.mb_size, 1, 1, 1])       #[None, num_Cluster, num_Event, L]

        tmp = ( Y_stacked_expanded - PSI_stacked_expanded )**2

        # tripazoidal approximation
        dist_y_expanded_ms = self.delta_range[-1]/(self.L-1) * (tf.reduce_sum(tmp, axis=3) - tmp[:, :, :, 0] - tmp[:, :, :, -1])
        dist_y_expanded    = tf.reduce_sum(dist_y_expanded_ms[:, :, 1:] * self.beta_ms, axis=2)

        dist_y_homo     = tf.reduce_sum(dist_y_expanded * S_one_hot, axis=1) #[None]
        dist_y_hetero   = tf.reduce_sum(dist_y_expanded * (1. - S_one_hot), axis=1) #[None]

        self.dist_y_homo     = tf.reduce_mean(dist_y_homo, axis=0)
        self.dist_y_hetero   = tf.reduce_mean(dist_y_hetero, axis=0)


        ### FOR USER-DEFINED DISTANCE MEASURE
        self.ZZ = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        YY    = []
        for l in range(self.L):
            tmp_d1 = self.delta_range[l] * tf.ones_like(tf.reshape(self.ZZ[:,0], shape=[-1, 1]))
            
            with tf.variable_scope('rnn', reuse=True):
                YY.append(prediction_network_softplus(self.ZZ, tmp_d1))
        self.YY_stacked = tf.stack(YY, axis=2)


        ### FOR THINNING-ALGORITHM
        self.D_IN = tf.placeholder(tf.float32, shape=[None], name='delta_in')  
        self.Z_IN = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        tmp_d_in = tf.reshape(self.D_IN * tf.ones_like(self.Z_IN[:,0]), shape=[-1, 1])

        with tf.variable_scope('rnn', reuse=True):
            self.Y_pred = prediction_network_softplus(self.Z_IN, tmp_d_in)


        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        pred_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn/prediction_net')
        enc_vars    = [tmp_var for tmp_var in global_vars if tmp_var not in pred_vars]

        self.loss_CLU = self.alpha*self.dist_z_homo
        self.loss_COM = self.beta*self.dist_y_homo - self.gamma*self.dist_y_hetero

        self.loss_CLU_COM = self.loss_CLU+self.loss_COM

        self.loss_TOTAL = self.loss_MLE + self.loss_CLU + self.loss_COM

        self.solver_MLE     = tf.train.AdamOptimizer(self.lr_rate, beta1=0.9, beta2=0.999).minimize(self.loss_MLE, var_list=global_vars)
        self.solver_CLUSTER = tf.train.AdamOptimizer(self.lr_rate, beta1=0.9, beta2=0.999).minimize(self.loss_CLU_COM, var_list=enc_vars)
        self.solver_TOTAL   = tf.train.AdamOptimizer(self.lr_rate, beta1=0.9, beta2=0.999).minimize(self.loss_TOTAL, var_list=global_vars)


    def train_mle(self, M_, D_, X_, lr_train, k_prob):
        return self.sess.run([self.solver_MLE, self.loss_MLE], 
                        feed_dict={self.M: M_, self.D: D_, self.X: X_,
                                   self.mb_size:np.shape(M_)[0], self.lr_rate: lr_train, self.keep_prob: k_prob})

    def train_cluster(self, M_, D_, X_, Mask_, S_, MU_, K_, delta_range_, alpha_, beta_, beta_ms_, gamma_, lr_train, k_prob):
        return self.sess.run([self.solver_CLUSTER, self.loss_CLU_COM, self.dist_z_homo, self.dist_y_homo, self.dist_y_hetero], 
                        feed_dict={self.M: M_, self.D: D_, self.X:X_, self.Mask: Mask_, self.S: S_, self.MU: MU_, 
                                   self.K: K_, self.delta_range: delta_range_,
                                   self.alpha:alpha_, self.beta:beta_, self.beta_ms:beta_ms_, self.gamma:gamma_,
                                   self.mb_size:np.shape(M_)[0], self.lr_rate: lr_train, self.keep_prob: k_prob})

    def train_total(self, M_, D_, X_, Mask_, S_, MU_, K_, delta_range_, alpha_, beta_, beta_ms_, gamma_, lr_train, k_prob):
        return self.sess.run([self.solver_MLE, self.solver_CLUSTER, self.loss_TOTAL, self.loss_MLE, self.loss_CLU_COM, 
                         self.dist_z_homo, self.dist_y_homo, self.dist_y_hetero], 
                        feed_dict={self.M: M_, self.D: D_, self.X:X_, self.Mask:Mask_, self.S: S_, self.MU: MU_, 
                                   self.K: K_, self.delta_range: delta_range_,
                                   self.alpha:alpha_, self.beta:beta_, self.beta_ms:beta_ms_, self.gamma:gamma_,
                                   self.mb_size:np.shape(M_)[0], self.lr_rate: lr_train, self.keep_prob: k_prob})

    def get_loss_mle(self, M_, D_, X_):
        return self.sess.run(self.loss_MLE, 
                        feed_dict={self.M: M_, self.D: D_, self.X: X_,
                                   self.mb_size:np.shape(M_)[0], self.keep_prob: 1.0})

    def get_loss_total(self, M_, D_, X_, Mask_, S_, MU_, K_, delta_range_, alpha_, beta_, beta_ms_, gamma_):
        return self.sess.run(self.loss_TOTAL, 
                        feed_dict={self.M: M_, self.D: D_, self.X:X_, self.Mask:Mask_, self.S: S_, self.MU: MU_, 
                                   self.K: K_, self.delta_range: delta_range_,
                                   self.alpha:alpha_, self.beta:beta_, self.beta_ms:beta_ms_, self.gamma:gamma_,
                                   self.mb_size:np.shape(M_)[0], self.keep_prob: 1.0})

    ### GET VALUES - LATENT SPACE
    def f_get_prediction_Z(self, M_, D_, X_, Mask_):
        return self.sess.run(self.Z, 
                        feed_dict={self.M:M_, self.D:D_, self.X:X_, self.Mask:Mask_, 
                                   self.mb_size:np.shape(M_)[0], self.keep_prob:1.0})

    def f_get_distance_Z(self, M_, D_, X_, Mask_, S_, MU_, K_):
        return self.sess.run([self.dist_z_homo, self.dist_z_hetero],
                        feed_dict={self.M: M_, self.D: D_, self.X:X_, self.Mask:Mask_, self.S: S_, self.MU: MU_, 
                                   self.K:K_, self.mb_size:np.shape(M_)[0], self.keep_prob: 1.0})


    ### GET VALUES - OUTPUT SPACE
    def f_get_prediction_Y_stacked(self, M_, D_, X_, Mask_, delta_range_):
        return self.sess.run(self.Y_stacked, 
                        feed_dict={self.M:M_, self.D:D_, self.X:X_, self.Mask:Mask_, self.delta_range:delta_range_, 
                                   self.mb_size:np.shape(M_)[0], self.keep_prob:1.0})

    def f_get_prediction_PSI_stacked(self, MU_, K_, delta_range_):
        return self.sess.run(self.PSI_stacked, 
                        feed_dict={self.MU:MU_, self.K:K_, self.delta_range:delta_range_, self.keep_prob:1.0})
							        # self.mb_size:np.shape(M_)[0], self.keep_prob:1.0})

    def f_get_distance_Y(self, M_, D_, X_, Mask_, S_, MU_, beta_ms_, K_, delta_range_):
        return self.sess.run([self.dist_y_homo, self.dist_y_hetero], 
                        feed_dict={self.M:M_, self.D:D_, self.X:X_, self.Mask:Mask_, self.S:S_, self.MU:MU_, self.K:K_, 
                                   self.beta_ms:beta_ms_, self.delta_range:delta_range_,
                                   self.mb_size:np.shape(M_)[0], self.keep_prob: 1.0})

    def f_get_prediction_YY_stacked(self, Z_, delta_range_):
        return self.sess.run(self.YY_stacked, 
                        feed_dict={self.ZZ: Z_, self.delta_range:delta_range_, 
                                   self.mb_size:np.shape(Z_)[0], self.keep_prob:1.0})

    def f_get_prediction_Y_pred(self, Z_, D_):
        return self.sess.run(self.Y_pred, 
                        feed_dict={self.Z_IN:Z_, self.D_IN: D_, self.mb_size:np.shape(Z_)[0], self.keep_prob:1.0})
