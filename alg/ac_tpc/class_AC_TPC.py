import numpy as np
import tensorflow as tf

import random
import os,sys

from tensorflow.python.ops.rnn import _transpose_batch_time

#user defined
import utils_network as utils



def log(x): 
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length



class AC_TPC:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name
        
        # INPUT/OUTPUT DIMENSIONS
        self.x_dim           = input_dims['x_dim'] #features + delta
        self.y_dim           = input_dims['y_dim']
        self.y_type          = input_dims['y_type']
        self.K               = input_dims['max_cluster']
        self.max_length      = input_dims['max_length']

        # Encoder
        self.h_dim_f         = network_settings['h_dim_encoder'] #encoder nodes
        self.num_layers_f    = network_settings['num_layers_encoder'] #encoder layers
        self.rnn_type        = network_settings['rnn_type']
        self.rnn_activate_fn   = network_settings['rnn_activate_fn']

        # Selector
        self.h_dim_h         = network_settings['h_dim_selector'] #selector nodes
        self.num_layers_h    = network_settings['num_layers_selector'] #selector layers
        
        # Predictor
        self.h_dim_g         = network_settings['h_dim_predictor'] #predictor nodes
        self.num_layers_g    = network_settings['num_layers_predictor'] #predictor layers
        
        self.fc_activate_fn  = network_settings['fc_activate_fn'] #selector & predictor
        
        # Latent Space
        self.z_dim           = self.h_dim_f * self.num_layers_f

        self._build_net()

        
    def _build_net(self):
        with tf.variable_scope(self.name):
            
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')
            self.lr_rate1    = tf.placeholder(tf.float32, name='learning_rate1')
            self.lr_rate2    = tf.placeholder(tf.float32, name='learning_rate2')
            self.keep_prob   = tf.placeholder(tf.float32, name='keep_probability')

            # Input and Output
            self.x          = tf.placeholder(tf.float32, [None, self.max_length, self.x_dim], name='inputs')
            self.y          = tf.placeholder(tf.float32, [None, self.max_length, self.y_dim], name='labels_onehot')
            
            # Embedding
            self.E          = tf.placeholder(tf.float32, [self.K, self.z_dim], name='embeddings_input')
            self.EE         = tf.Variable(self.E, name='embeddings_var')
            self.embeddings = tf.nn.tanh(self.EE)

            # self.embde         = tf.nn.tanh(self.EE)
            # self.EE         = tf.Variable(self.E, name='embeddings_var')
            
            self.s          = tf.placeholder(tf.int32, [None], name='cluster_label')
            self.s_onehot   = tf.one_hot(self.s, self.K)

            # LOSS PARAMETERS
            self.alpha      = tf.placeholder(tf.float32, name = 'alpha') #For sample-wise entropy
            self.beta       = tf.placeholder(tf.float32, name = 'beta')  #For prediction loss (i.e., mle)
            self.gamma      = tf.placeholder(tf.float32, name = 'gamma') #For batch-wise entropy
            self.delta      = tf.placeholder(tf.float32, name = 'delta') #For embedding

            
            '''
                ### CREATE RNN MASK
                    - This is to flexibly handle sequences with different length
                    - rnn_mask1: last observation; [mb_size, max_length]
                    - rnn_mask2: all available observations; [mb_size, max_length]
            '''
            # CREATE RNN MASK:            
            seq_length     = get_seq_length(self.x)
            tmp_range      = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            self.rnn_mask1 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length-1, axis=1)), tf.float32) #last observation
            self.rnn_mask2 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length-1, axis=1)), tf.float32) #all available observation
            
            
            ### DEFINE SELECTOR
            def selector(x_, o_dim_=self.K, num_layers_=2, h_dim_=self.h_dim_h, activation_fn=self.fc_activate_fn, reuse=tf.AUTO_REUSE):
                out_fn = tf.nn.softmax
                with tf.variable_scope('selector', reuse=reuse):
                    if num_layers_ == 1:
                        out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=out_fn, scope='selector_out')
                    else: #num_layers > 1
                        for tmp_layer in range(num_layers_-1):
                            if tmp_layer == 0:
                                net = x_
                            net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, scope='selector_'+str(tmp_layer))
                            net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                        out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=out_fn, scope='selector_out')  
                return out
            
            
            ### DEFINE PREDICTOR
            def predictor(x_, o_dim_=self.y_dim, o_type_=self.y_type, num_layers_=1, h_dim_=self.h_dim_g, activation_fn=self.fc_activate_fn, reuse=tf.AUTO_REUSE):
                if o_type_ == 'continuous':
                    out_fn = None
                elif o_type_ == 'categorical':
                    out_fn = tf.nn.softmax #for classification task
                elif o_type_ == 'binary':
                    out_fn = tf.nn.sigmoid
                else:
                    raise Exception('Wrong output type. The value {}!!'.format(o_type_))
                    
                with tf.variable_scope('predictor', reuse=reuse):
                    if num_layers_ == 1:
                        out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=out_fn, scope='predictor_out')
                    else: #num_layers > 1
                        for tmp_layer in range(num_layers_-1):
                            if tmp_layer == 0:
                                net = x_
                            net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, scope='predictor_'+str(tmp_layer))
                            net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                        out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=out_fn, scope='predictor_out')  
                return out

            
            ### DEFINE LOOP FUNCTION FOR ENCODRER (f-g, f-h relations are created here)
            def loop_fn(time, cell_output, cell_state, loop_state):
                
                emit_output = cell_output 

                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                    next_loop_state = loop_state_ta
                else:
                    next_cell_state = cell_state
                    tmp_z  = utils.create_concat_state_h(next_cell_state, self.num_layers_f, self.rnn_type)      
                    tmp_y  = predictor(tmp_z, self.y_dim, self.y_type, self.num_layers_g, self.h_dim_g, self.fc_activate_fn)        
                    tmp_pi = selector(tmp_z, self.K, self.num_layers_h, self.h_dim_h, self.fc_activate_fn)

                    next_loop_state = (loop_state[0].write(time-1, tmp_z),  # save all the hidden states
                                       loop_state[1].write(time-1, tmp_y),  # save all the output
                                       loop_state[2].write(time-1, tmp_pi)) # save all the selector_net output (i.e., pi)

                elements_finished = (time >= self.max_length)

                #this gives the break-point (no more recurrence after the max_length)
                finished = tf.reduce_all(elements_finished)    
                next_input = tf.cond(finished, 
                                     lambda: tf.zeros([self.mb_size, self.x_dim], dtype=tf.float32),  
                                     lambda: inputs_ta.read(time))
                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

            
            '''
                ##### CREATE RNN NETWORK
                    - (INPUT)  inputs_ta: TensorArray with [max_length, mb_size, x_dim] #x_dim included delta
                    - (OUTPUT) 
                        . zs     = rnn states (h) in LSTM/GRU             ; [mb_size, max_length z_dim]
                        . y_hats = output of predictor taking zs as inputs; [mb_size, max_length, y_dim]
                        . pis    = output of selector                     ; [mb_size, max_length, K]

            '''
            inputs    = self.x
            inputs_ta = tf.TensorArray(
                dtype=tf.float32, size=self.max_length
            ).unstack(_transpose_batch_time(inputs), name = 'rnn_input')


            cell = utils.create_rnn_cell(
                self.h_dim_f, self.num_layers_f, 
                self.keep_prob, self.rnn_type, self.rnn_activate_fn
            )

            #define the loop_state TensorArray for information from rnn time steps
            loop_state_ta = (
                tf.TensorArray(size=self.max_length, dtype=tf.float32, clear_after_read=False),  #zs (j=1,...,J)
                tf.TensorArray(size=self.max_length, dtype=tf.float32, clear_after_read=False),  #y_hats (j=1,...,J)
                tf.TensorArray(size=self.max_length, dtype=tf.float32, clear_after_read=False)   #pis (j=1,...,J)
            )  

            _, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn) #, parallel_iterations=1)


            self.zs         = _transpose_batch_time(loop_state_ta[0].stack())
            self.y_hats     = _transpose_batch_time(loop_state_ta[1].stack())
            self.pis        = _transpose_batch_time(loop_state_ta[2].stack())

            ### SAMPLING PROCESS
            s_dist          = tf.distributions.Categorical(probs=tf.reshape(self.pis, [-1, self.K])) #define the categorical dist.
            s_sample        = s_dist.sample()

            mask_e          = tf.cast(tf.equal(tf.expand_dims(tf.range(0, self.K, 1), axis=0), tf.expand_dims(s_sample, axis=1)), tf.float32)
            z_bars          = tf.matmul(mask_e, self.embeddings)
            pi_sample       = tf.reduce_sum(mask_e * tf.reshape(log(self.pis), [-1, self.K]), axis=1)

            with tf.variable_scope('rnn', reuse=True):
                y_bars   = predictor(z_bars, self.y_dim, self.y_type, self.num_layers_g, self.h_dim_g, self.fc_activate_fn)

            self.z_bars    = tf.reshape(z_bars, [-1, self.max_length, self.z_dim])
            self.y_bars    = tf.reshape(y_bars, [-1, self.max_length, self.y_dim])
            self.pi_sample = tf.reshape(pi_sample, [-1, self.max_length])
            self.s_sample  = tf.reshape(s_sample, [-1, self.max_length])

            
            ### DEFINE LOSS FUNCTIONS
            #\ell_{1}: KL divergence loss for regression and binary/categorical-classification task
            def loss_1(y_true_, y_pred_, y_type_ = self.y_type):                
                if y_type_ == 'continuous':
                    tmp_loss = tf.reduce_sum((y_true_ - y_pred_)**2, axis=-1)
                elif y_type_ == 'categorical':
                    tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_), axis=-1)
                elif y_type_ == 'binary':
                    tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_) + (1.-y_true_) * log(1.-y_pred_), axis=-1)
                else:
                    raise Exception('Wrong output type. The value {}!!'.format(y_type_))                    
                return tmp_loss

            #batch-wise entropy
            tmp_pis   = tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.K]) * self.pis
            mean_pis  = tf.reduce_sum(tf.reduce_sum(tmp_pis, axis=1), axis=0) / tf.reduce_sum(tf.reduce_sum(self.rnn_mask2, axis=1), axis=0, keepdims=True)

            ## LOSS_MLE: MLE prediction loss (for initalization)
            self.LOSS_MLE   = tf.reduce_mean(tf.reduce_sum(self.rnn_mask2 * loss_1(self.y, self.y_hats, self.y_type), axis=1))
            
            
            ## LOSS1: predictive clustering loss
            self.LOSS_1     = tf.reduce_mean(tf.reduce_sum(self.rnn_mask2 * loss_1(self.y, self.y_bars, self.y_type), axis=1))
            self.LOSS_1_AC  = tf.reduce_mean(tf.reduce_sum(self.rnn_mask2 * self.pi_sample * loss_1(self.y, self.y_bars, self.y_type), axis=1))

            ## LOSS2: sample-wise entropy loss
            self.LOSS_2     = tf.reduce_mean(-tf.reduce_sum(self.rnn_mask2 * tf.reduce_sum(self.pis * log(self.pis), axis=2), axis=1))
            
            
            predictor_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/rnn/predictor')
            selecter_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/rnn/selector')
            embedding_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/embeddings_var')
            encoder_vars    = [vars_ for vars_ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
                               if vars_ not in predictor_vars+selecter_vars+embedding_vars]
            
            ### EMBEDDING TRAINING
            with tf.variable_scope('rnn', reuse=True):
                Ey   = predictor(self.embeddings, self.y_dim, self.y_type, self.num_layers_g, self.h_dim_g, self.fc_activate_fn)
                # Ey   = predictor(self.EE, self.y_dim, self.y_type, self.num_layers_g, self.h_dim_g, self.fc_activate_fn)

            ## LOSS3: embedding separation loss (prevents embedding from collapsing)
            self.LOSS_3 = 0
            for i in range(self.K):
                for j in range(i+1, self.K):
                    self.LOSS_3 += - loss_1(Ey[i, :], Ey[j, :], y_type_ = self.y_type) / ((self.K-1)*(self.K-2)) # negative because we want to increase this;
                    
            
            ### DEFINE OPTIMIZATION SOLVERS
            self.solver_MLE           = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_MLE, var_list=encoder_vars+predictor_vars
            )
            self.solver_L1_critic     = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_1,
                var_list=encoder_vars+predictor_vars
            )
            self.solver_L1_actor      = tf.train.AdamOptimizer(self.lr_rate2).minimize(
                self.LOSS_1_AC + self.alpha*self.LOSS_2, 
                var_list=encoder_vars + selecter_vars
            )            
            self.solver_E             = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_1 + self.beta*self.LOSS_3, 
                var_list=embedding_vars
            )

            ### INITIALIZE SELECTOR
            self.zz     = tf.placeholder(tf.float32, [None, self.z_dim])
            with tf.variable_scope('rnn', reuse=True):
                self.yy    = predictor(self.zz, self.y_dim, self.y_type, self.num_layers_g, self.h_dim_g, self.fc_activate_fn) #to check the predictor output given z
                self.s_out = selector(self.zz, self.K, self.num_layers_h, self.h_dim_h, self.fc_activate_fn)
            
            ## LOSS_S: selector initialization (cross-entropy wrt initialized class)
            self.LOSS_S   = tf.reduce_mean(- tf.reduce_sum(self.s_onehot*log(self.s_out), axis=1))
            self.solver_S = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_S, var_list=selecter_vars
            )
            
    ### TRAINING FUNCTIONS        
    def train_mle(self, x_, y_, lr_train, k_prob):
        return self.sess.run([self.solver_MLE, self.LOSS_MLE],
                             feed_dict={self.x: x_, self.y: y_,
                                        self.mb_size:np.shape(x_)[0], self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_critic(self, x_, y_, lr_train, k_prob):
        return self.sess.run([self.solver_L1_critic, self.LOSS_1],
                             feed_dict={self.x: x_, self.y: y_, 
                                        self.mb_size:np.shape(x_)[0], self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_actor(self, x_, y_, alpha_, lr_train, k_prob):
        return self.sess.run([self.solver_L1_actor, self.LOSS_1, self.LOSS_2],
                             feed_dict={self.x: x_, self.y: y_,
                                        self.alpha: alpha_,
                                        self.mb_size:np.shape(x_)[0], self.lr_rate2: lr_train, self.keep_prob: k_prob})
    
    def train_selector(self, z_, s_, lr_train, k_prob):
        return self.sess.run([self.solver_S, self.LOSS_S],
                             feed_dict={self.zz: z_, self.s: s_, 
                                        self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_embedding(self, x_, y_, beta_, lr_train, k_prob):   
        return self.sess.run([self.solver_E, self.LOSS_1, self.LOSS_3], 
                             feed_dict={self.x:x_, self.y:y_,
                                        self.beta:beta_,
                                        self.mb_size:np.shape(x_)[0], 
                                        self.lr_rate1:lr_train, self.keep_prob:k_prob})

    def get_losses(self, x_, y_):   
        return self.sess.run([self.LOSS_1, self.LOSS_2, self.LOSS_3], 
                             feed_dict={self.x:x_, self.y:y_,
                                        self.mb_size:np.shape(x_)[0], 
                                        self.keep_prob:1.0})
        
    ### PREDICTION FUNCTIONS
    def predict_y_hats(self, x_):
        return self.sess.run([self.y_hats, self.rnn_mask2], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_y_bars(self, x_):
        return self.sess.run([self.y_bars, self.rnn_mask2], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_yy(self, z_):
        return self.sess.run(self.yy,
                             feed_dict={self.zz:z_, self.mb_size:np.shape(z_)[0], self.keep_prob:1.0})
        
    def predict_zs_and_pis_m2(self, x_):
        return self.sess.run([self.zs, self.pis, self.rnn_mask2], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_s_sample(self, x_):
        return self.sess.run([self.s_sample, self.rnn_mask2], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_zbars_and_pis_m1(self, x_):
        return self.sess.run([self.z_bars, self.pis, self.rnn_mask1], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_zs_and_pis_m1(self, x_):
        return self.sess.run([self.zs, self.pis, self.rnn_mask1], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})
    
    def predict_zbars_and_pis_m2(self, x_):
        return self.sess.run([self.z_bars, self.pis, self.rnn_mask2], 
                             feed_dict={self.x:x_, self.mb_size:np.shape(x_)[0], self.keep_prob:1.0})



### INITIALIZE EMBEDDING AND SELECTOR
from sklearn.cluster import MiniBatchKMeans, KMeans

def initialize_embedding(model, x, K):
    tmp_z, _, _     = model.predict_zs_and_pis_m2(x)
    tmp_y, tmp_m    = model.predict_y_hats(x)

    z_dim  = np.shape(tmp_z)[-1]
    y_dim  = np.shape(tmp_y)[-1]

    tmp_z  = (tmp_z * np.tile(np.expand_dims(tmp_m, axis=2), [1,1,z_dim])).reshape([-1, z_dim])
    tmp_z  = tmp_z[np.sum(np.abs(tmp_z), axis=1) != 0]

    tmp_y  = (tmp_y * np.tile(np.expand_dims(tmp_m, axis=2), [1,1,y_dim])).reshape([-1, y_dim])
    tmp_y  = tmp_y[np.sum(np.abs(tmp_y), axis=1) != 0]

    km     = KMeans(n_clusters = K, init='k-means++')
    _      = km.fit(tmp_y)
    tmp_ey = km.cluster_centers_
    tmp_s  = km.predict(tmp_y)

    tmp_e  = np.zeros([K, z_dim])
    for k in range(K):
        # tmp_e[k, :] = np.mean(tmp_z[tmp_s == k])
        tmp_e[k,:] = tmp_z[np.argmin(np.sum(np.abs(tmp_y - tmp_ey[k, :]),axis=1)), :]

    return tmp_e, tmp_s, tmp_z