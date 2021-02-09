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


class Model_Longitudinal_Attention:
    # def __init__(self, sess, name, mb_size, input_dims, network_settings):
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.x_dim_cont         = input_dims['x_dim_cont']
        self.x_dim_bin          = input_dims['x_dim_bin']

        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.max_length         = input_dims['max_length']

        # NETWORK HYPER-PARMETERS
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_ATT     = network_settings['num_layers_ATT']
        self.num_layers_CS      = network_settings['num_layers_CS']

        self.RNN_type           = network_settings['RNN_type']

        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']
        
        self.reg_W              = tf.contrib.layers.l1_regularizer(scale=network_settings['reg_W'])
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=network_settings['reg_W_out'])

        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')

            self.lr_rate     = tf.placeholder(tf.float32)
            self.keep_prob   = tf.placeholder(tf.float32)                                                      #keeping rate
            self.a           = tf.placeholder(tf.float32)
            self.b           = tf.placeholder(tf.float32)
            self.c           = tf.placeholder(tf.float32)

            self.x           = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim])
            self.x_mi        = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim])           #this is the missing indicator (including for cont. & binary) (includes delta)
            self.k           = tf.placeholder(tf.float32, shape=[None, 1])                                     #event/censoring label (censoring:0)
            self.t           = tf.placeholder(tf.float32, shape=[None, 1])


            self.fc_mask1    = tf.placeholder(tf.float32, shape=[None, self.num_Event, self.num_Category])     #for denominator
            self.fc_mask2    = tf.placeholder(tf.float32, shape=[None, self.num_Event, self.num_Category])     #for Loss 1
            self.fc_mask3    = tf.placeholder(tf.float32, shape=[None, self.num_Category])                     #for Loss 2

            
            seq_length     = get_seq_length(self.x)
            tmp_range      = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            
            self.rnn_mask1 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)            
            self.rnn_mask2 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32) 
            
            
            ### DEFINE LOOP FUNCTION FOR RAW_RNN w/ TEMPORAL ATTENTION
            def loop_fn_att(time, cell_output, cell_state, loop_state):

                emit_output = cell_output 

                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                    next_loop_state = loop_state_ta
                else:
                    next_cell_state = cell_state
                    tmp_h = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type)

                    e = utils.create_FCNet(tf.concat([tmp_h, all_last], axis=1), self.num_layers_ATT, self.h_dim2, 
                                           tf.nn.tanh, 1, None, self.initial_W, keep_prob=self.keep_prob)
                    e = tf.exp(e)

                    next_loop_state = (loop_state[0].write(time-1, e),                # save att power (e_{j})
                                       loop_state[1].write(time-1, tmp_h))  # save all the hidden states

                # elements_finished = (time >= seq_length)
                elements_finished = (time >= self.max_length-1)

                #this gives the break-point (no more recurrence after the max_length)
                finished = tf.reduce_all(elements_finished)    
                next_input = tf.cond(finished, lambda: tf.zeros([self.mb_size, 2*self.x_dim], dtype=tf.float32),  # [x_hist, mi_hist]
                                               lambda: inputs_ta.read(time))

                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)


            
            # divide into the last x and previous x's
            x_last = tf.slice(self.x, [0,(self.max_length-1), 1], [-1,-1,-1])      #current measurement
            x_last = tf.reshape(x_last, [-1, (self.x_dim_cont+self.x_dim_bin)])    #remove the delta of the last measurement

            x_last = tf.reduce_sum(tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim]) * self.x, reduction_indices=1)    #sum over time since all others time stamps are 0
            x_last = tf.slice(x_last, [0,1], [-1,-1])                               #remove the delta of the last measurement
            x_hist = self.x * (1.-tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim]))                                    #since all others time stamps are 0 and measurements are 0-padded
            x_hist = tf.slice(x_hist, [0, 0, 0], [-1,(self.max_length-1),-1])  

            # do same thing for missing indicator
            mi_last = tf.slice(self.x_mi, [0,(self.max_length-1), 1], [-1,-1,-1])      #current measurement
            mi_last = tf.reshape(mi_last, [-1, (self.x_dim_cont+self.x_dim_bin)])    #remove the delta of the last measurement

            mi_last = tf.reduce_sum(tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim]) * self.x_mi, reduction_indices=1)    #sum over time since all others time stamps are 0
            mi_last = tf.slice(mi_last, [0,1], [-1,-1])                               #remove the delta of the last measurement
            mi_hist = self.x_mi * (1.-tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim]))                                    #since all others time stamps are 0 and measurements are 0-padded
            mi_hist = tf.slice(mi_hist, [0, 0, 0], [-1,(self.max_length-1),-1])  

            all_hist = tf.concat([x_hist, mi_hist], axis=2)
            all_last = tf.concat([x_last, mi_last], axis=1)


            #extract inputs for the temporal attention: mask (to incorporate only the measured time) and x_{M}
            seq_length     = get_seq_length(x_hist)
            rnn_mask_att   = tf.cast(tf.not_equal(tf.reduce_sum(x_hist, reduction_indices=2), 0), dtype=tf.float32)  #[mb_size, max_length-1], 1:measurements 0:no measurements
            

            ##### SHARED SUBNETWORK: RNN w/ TEMPORAL ATTENTION
            #change the input tensor to TensorArray format with [max_length, mb_size, x_dim]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_length-1).unstack(_transpose_batch_time(all_hist), name = 'Shared_Input')


            #create a cell with RNN hyper-parameters (RNN types, #layers, #nodes, activation functions, keep proability)
            cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, 
                                         self.RNN_type, self.RNN_active_fn)

            #define the loop_state TensorArray for information from rnn time steps
            loop_state_ta = (tf.TensorArray(size=self.max_length-1, dtype=tf.float32),  #e values (e_{j})
                             tf.TensorArray(size=self.max_length-1, dtype=tf.float32))  #hidden states (h_{j})
            
            rnn_outputs_ta, self.rnn_final_state, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn_att)
            #rnn_outputs_ta  : TensorArray
            #rnn_final_state : Tensor
            #rnn_states_ta   : (TensorArray, TensorArray)

            rnn_outputs = _transpose_batch_time(rnn_outputs_ta.stack())
            # rnn_outputs =  tf.reshape(rnn_outputs, [-1, self.max_length-1, self.h_dim1])

            rnn_states  = _transpose_batch_time(loop_state_ta[1].stack())

            att_weight  = _transpose_batch_time(loop_state_ta[0].stack()) #e_{j}
            att_weight  = tf.reshape(att_weight, [-1, self.max_length-1]) * rnn_mask_att # masking to set 0 for the unmeasured e_{j}

            #get a_{j} = e_{j}/sum_{l=1}^{M-1}e_{l}
            self.att_weight  = div(att_weight,(tf.reduce_sum(att_weight, axis=1, keepdims=True) + _EPSILON)) #softmax (tf.exp is done, previously)

            # 1) expand att_weight to hidden state dimension, 2) c = \sum_{j=1}^{M} a_{j} x h_{j}
            self.context_vec = tf.reduce_sum(tf.tile(tf.reshape(self.att_weight, [-1, self.max_length-1, 1]), [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_states, axis=1)


            self.z_mean      = FC_Net(rnn_outputs, self.x_dim, activation_fn=None, weights_initializer=self.initial_W, scope="RNN_out_mean1")
            self.z_std       = tf.exp(FC_Net(rnn_outputs, self.x_dim, activation_fn=None, weights_initializer=self.initial_W, scope="RNN_out_std1"))

            epsilon          = tf.random_normal([self.mb_size, self.max_length-1, self.x_dim], mean=0.0, stddev=1.0, dtype=tf.float32)
            self.z           = self.z_mean + self.z_std * epsilon

            
            ##### CS-SPECIFIC SUBNETWORK w/ FCNETS 
            inputs = tf.concat([x_last, self.context_vec], axis=1)


            #1 layer for combining inputs
            h = FC_Net(inputs, self.h_dim2, activation_fn=self.FC_active_fn, weights_initializer=self.initial_W, scope="Layer1")
            h = tf.nn.dropout(h, keep_prob=self.keep_prob)

            # (num_layers_CS-1) layers for cause-specific (num_Event subNets)
            out = []
            for _ in range(self.num_Event):
                cs_out = utils.create_FCNet(h, (self.num_layers_CS), self.h_dim2, self.FC_active_fn, self.h_dim2, self.FC_active_fn, self.initial_W, self.reg_W, self.keep_prob)
                out.append(cs_out)
            out = tf.stack(out, axis=1) # stack referenced on subject
            out = tf.reshape(out, [-1, self.num_Event*self.h_dim2])
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = FC_Net(out, self.num_Event * self.num_Category, activation_fn=tf.nn.softmax, 
                         weights_initializer=self.initial_W, weights_regularizer=self.reg_W_out, scope="Output")
            self.out = tf.reshape(out, [-1, self.num_Event, self.num_Category])


            ##### GET LOSS FUNCTIONS
            self.loss_Log_Likelihood()      #get loss1: Log-Likelihood loss
            self.loss_Ranking()             #get loss2: Ranking loss
            self.loss_RNN_Prediction()      #get loss3: RNN prediction loss

            self.LOSS_TOTAL     = self.a*self.LOSS_1 + self.b*self.LOSS_2 + self.c*self.LOSS_3 + tf.losses.get_regularization_loss()
            self.LOSS_BURNIN    = self.LOSS_3 + tf.losses.get_regularization_loss()

            self.solver         = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_TOTAL)
            self.solver_burn_in = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_BURNIN)


    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self):
        sigma3 = tf.constant(1.0, dtype=tf.float32)

        I_1 = tf.sign(self.k)
        denom = 1 - tf.reduce_sum(tf.reduce_sum(self.fc_mask1 * self.out, reduction_indices=2), reduction_indices=1, keepdims=True) # make subject specific denom.
        denom = tf.clip_by_value(denom, tf.cast(_EPSILON, dtype=tf.float32), tf.cast(1.-_EPSILON, dtype=tf.float32))

        #for uncenosred: log P(T=t,K=k|x,Y,t>t_M)
        tmp1 = tf.reduce_sum(tf.reduce_sum(self.fc_mask2 * self.out, reduction_indices=2), reduction_indices=1, keepdims=True)
        tmp1 = I_1 * log(div(tmp1,denom))

        #for censored: log \sum P(T>t|x,Y,t>t_M)
        tmp2 = tf.reduce_sum(tf.reduce_sum(self.fc_mask2 * self.out, reduction_indices=2), reduction_indices=1, keepdims=True)
        tmp2 = (1. - I_1) * log(div(tmp2,denom))

        self.LOSS_1 = - tf.reduce_mean(tmp1 + sigma3*tmp2)


    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self):
        sigma1 = tf.constant(0.1, dtype=tf.float32)

        eta = []
        for e in range(self.num_Event):
            one_vector = tf.ones_like(self.t, dtype=tf.float32)
            I_2 = tf.cast(tf.equal(self.k, e+1), dtype = tf.float32) #indicator for event
            I_2 = tf.diag(tf.squeeze(I_2))
            tmp_e = tf.reshape(tf.slice(self.out, [0, e, 0], [-1, 1, -1]), [-1, self.num_Category]) #event specific joint prob.

            R = tf.matmul(tmp_e, tf.transpose(self.fc_mask3)) #no need to divide by each individual dominator
            # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

            diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
            R = tf.matmul(one_vector, tf.transpose(diag_R)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
            R = tf.transpose(R)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

            T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
            # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

            T = tf.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

            tmp_eta = tf.reduce_mean(T * tf.exp(-R/sigma1), reduction_indices=1, keepdims=True)

            eta.append(tmp_eta)
        eta = tf.stack(eta, axis=1) #stack referenced on subjects
        eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_Event]), reduction_indices=1, keepdims=True)

        self.LOSS_2 = tf.reduce_sum(eta) #sum over num_Events


    ### LOSS-FUNCTION 3 -- RNN prediction loss
    def loss_RNN_Prediction(self):
        tmp_x  = tf.slice(self.x, [0,1,0], [-1,-1,-1])  # (t=2 ~ M)
        tmp_mi = tf.slice(self.x_mi, [0,1,0], [-1,-1,-1])  # (t=2 ~ M)

        tmp_mask1  = tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.x_dim]) #for hisotry (1...J-1)
        tmp_mask1  = tmp_mask1[:, :(self.max_length-1), :] 

        zeta = tf.reduce_mean(tf.reduce_sum(tmp_mask1 * (1. - tmp_mi) * tf.pow(self.z - tmp_x, 2), reduction_indices=1))  #loss calculated for selected features.

        self.LOSS_3 = zeta

 
    def get_cost(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma)      = PARAMETERS
        return self.sess.run(self.LOSS_TOTAL, 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb, 
                                        self.fc_mask1: m1_mb, self.fc_mask2:m2_mb, self.fc_mask3: m3_mb, 
                                        self.a:alpha, self.b:beta, self.c:gamma,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})

    def train(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma)      = PARAMETERS
        return self.sess.run([self.solver, self.LOSS_TOTAL], 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb,
                                        self.fc_mask1: m1_mb, self.fc_mask2:m2_mb, self.fc_mask3: m3_mb, 
                                        self.a:alpha, self.b:beta, self.c:gamma,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
    
    def train_burn_in(self, DATA, MISSING, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (x_mi_mb)                 = MISSING

        return self.sess.run([self.solver_burn_in, self.LOSS_3], 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb, 
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
    
    def predict(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.out, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_z(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.z, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_rnnstate(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.rnn_final_state, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_att(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.att_weight, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_context_vec(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.context_vec, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def get_z_mean_and_std(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run([self.z_mean, self.z_std], feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})