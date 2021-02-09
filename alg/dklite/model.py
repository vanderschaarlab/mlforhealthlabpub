import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np



class DKLITE(object):
    def __init__(self, input_dim, output_dim, num_hidden=50, num_layers =2, learning_rate=0.001,
                 reg_var=1.0,reg_rec=1.0):


        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.size_z = num_hidden
        self.ml_primal = {}
        self.ker_inv = {}
        self.params = {}
        self.mean = {}
        self.num = {}

        ''' Initialize parameter weight '''
        self.params = self.initialize_weights()

        self.mu = tf.reduce_mean(self.T)
        self.Z_train = self.Encoder(self.X)
        self.Z_test = self.Encoder(self.X_u)

        self.loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.Decoder(self.Z_train)),axis=1))

        Z_0 = tf.gather(self.Z_train, tf.where(self.T < 0.5)[:, 0])
        Y_0 = tf.gather(self.Y,  tf.where(self.T < 0.5)[:, 0])

        Z_1 = tf.gather(self.Z_train, tf.where(self.T > 0.5)[:, 0])
        Y_1 = tf.gather(self.Y, tf.where(self.T > 0.5)[:, 0])

        mean_0 = tf.reduce_mean(Y_0)
        mean_1 = tf.reduce_mean(Y_1)

        Y_0 = (Y_0-mean_0)
        Y_1 = (Y_1-mean_1)


        self.GP_NN(Y_0, Z_0, 0)
        self.GP_NN(Y_1, Z_1,1)


        self.var_0 = tf.reduce_mean(tf.diag_part(tf.matmul(Z_1,tf.matmul(self.ker_inv['0'], tf.transpose(Z_1)))))
        self.var_1 = tf.reduce_mean(tf.diag_part(tf.matmul(Z_0,tf.matmul(self.ker_inv['1'], tf.transpose(Z_0)))))


        self.ele_var_0_tr = tf.diag_part(tf.matmul(self.Z_train,tf.matmul(self.ker_inv['0'], tf.transpose(self.Z_train))))
        self.ele_var_1_tr = tf.diag_part(tf.matmul(self.Z_train,tf.matmul(self.ker_inv['1'], tf.transpose(self.Z_train))))

        self.ele_var_0_te = tf.diag_part(tf.matmul(self.Z_test,tf.matmul(self.ker_inv['0'], tf.transpose(self.Z_test))))
        self.ele_var_1_te = tf.diag_part(tf.matmul(self.Z_test,tf.matmul(self.ker_inv['1'], tf.transpose(self.Z_test))))

        pred_tr_0 = tf.matmul(self.Z_train, self.mean['0'])  + mean_0
        pred_tr_1 = tf.matmul(self.Z_train, self.mean['1']) + mean_1
        pred_te_0 = tf.matmul(self.Z_test, self.mean['0'])  + mean_0
        pred_te_1 = tf.matmul(self.Z_test, self.mean['1']) + mean_1

        self.Y_train = tf.concat([pred_tr_0,pred_tr_1],axis=1)
        self.Y_test = tf.concat([pred_te_0,pred_te_1],axis=1)

        self.loss_0 =  self.ml_primal['0']+ self.ml_primal['1']
        self.prediction_loss = self.ml_primal['0']+ self.ml_primal['1'] + reg_var *(self.var_0 + self.var_1)+ reg_rec * self.loss_1
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.prediction_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def element_var(self, X, Y, T, X_u):

        var_0_tr, var_1_tr,var_0_te, var_1_te = self.sess.run([self.ele_var_0_tr,self.ele_var_1_tr,self.ele_var_0_te,self.ele_var_1_te],
                feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})

        return var_0_tr,var_1_tr,var_0_te,var_1_te


    def embed(self, X, Y, T):

        Z= self.sess.run(self.Z_train, feed_dict={self.X: X, self.Y: Y, self.T: T})

        return Z


    def fit(self, X, Y, T, num_iteration):

        loss_list = []
        for i in range(num_iteration):

            loss, _ = self.sess.run([self.prediction_loss,self.optimizer], feed_dict={self.X: X, self.Y: Y, self.T: T})

            loss_list.append(np.sum(loss))

            diff_list = np.abs(np.diff(loss_list))

            if  i>50 and np.abs(np.mean(diff_list[-10:]) - np.mean(diff_list[-40:-10]) )< np.std(diff_list[-40:-10]):
                break


    def pred(self, X, Y, T, X_u):

        Y_hat_train, Y_hat_test = self.sess.run([self.Y_train, self.Y_test], feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})

        return Y_hat_train, Y_hat_test


    def destroy_graph(self):

        tf.reset_default_graph()


    def Encoder(self, X):

        X_h =tf.nn.elu(tf.matmul(X, self.params['e_w_in']) + self.params['e_b_in'])

        for layer_i in range( self.num_layers):
            X_h = tf.nn.elu(tf.matmul(X_h, self.params['e_w_' + str(layer_i)])+self.params['e_b_' + str(layer_i)])

        Z =  tf.nn.elu(tf.matmul(X_h, self.params['e_w_' + str(self.num_layers)])+ self.params['e_b_' + str(self.num_layers)])

        return Z


    def Decoder(self,Z):

        Z_pred = tf.nn.elu(tf.matmul(Z, self.params['d_w_in']) + self.params['d_b_in'])

        for layer_i in range(self.num_layers):
            Z_pred = tf.nn.elu(tf.matmul(Z_pred, self.params['d_w_' + str(layer_i)])+ self.params['d_b_' + str(layer_i)])

        X_p = tf.matmul(Z_pred, self.params['d_w_' + str(self.num_layers)]+ self.params['d_b_' + str(self.num_layers)])

        return X_p


    def GP_NN(self, Y_f, Z_f,index):

        beta = tf.ones([1,1],tf.float32)
        lam = 1000*tf.ones([1,1],tf.float32)
        r = beta / lam

        self.DD = tf.shape(Z_f)[1]

        phi_phi = tf.matmul(tf.transpose(Z_f), Z_f)

        Ker = r  * phi_phi + tf.eye(tf.shape(Z_f)[1], dtype=tf.float32)

        L_matrix = tf.cholesky(Ker)

        L_inv_reduce = tf.linalg.triangular_solve(L_matrix, rhs=tf.eye(self.DD, dtype=tf.float32))

        L_y = tf.matmul(L_inv_reduce, tf.matmul(tf.transpose(Z_f), Y_f))

        self.ker_inv[str(index)] = tf.matmul(tf.transpose(L_inv_reduce), L_inv_reduce) / lam

        self.mean[str(index)] = r * tf.matmul(tf.transpose(L_inv_reduce), L_y)

        term1 = - tf.reduce_mean(tf.square(L_y))
        #term2 = tf.log(tf.linalg.diag_part(L_matrix)) / ((1-index)*tf.reduce_sum(1 - self.T) + (index)* tf.reduce_sum(self.T))

        self.ml_primal[str(index)] = term1 #+  term2


    def initialize_weights(self):

        self.X = tf.placeholder(tf.float32, [None, self.input_dim])

        self.X_u = tf.placeholder(tf.float32, [None, self.input_dim])

        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.T = tf.placeholder(tf.float32, [None, 1])

        all_weights = {}

        ''' Input layer of the encoder '''
        name_wi = 'e_w_in'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.input_dim, self.num_hidden], trainable=True)
        name_bi = 'e_b_in'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.num_hidden], trainable=True)

        ''' Hidden layer of the encoder '''
        for layer_i in range(self.num_layers):

            name_wi = 'e_w_' + str(layer_i)
            all_weights[name_wi ] = tf.get_variable(name =name_wi,  shape=[self.num_hidden,self.num_hidden],  trainable=True)

            name_bi = 'e_b_' + str(layer_i)
            all_weights[name_bi] = tf.get_variable(name =name_bi, shape = [self.num_hidden], trainable=True)


        ''' Final layer of the encoder '''
        name_wi = 'e_w_' + str(self.num_layers)
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.num_hidden, self.size_z], trainable=True)

        name_bi = 'e_b_' + str(self.num_layers)
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.size_z], trainable=True)

        name_wi = 'e_w_out_0'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.output_dim], trainable=True)

        name_bi = 'e_b_out_0'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.output_dim], trainable=True)

        name_wi = 'e_w_out_1'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.output_dim], trainable=True)

        name_bi = 'e_b_out_1'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.output_dim], trainable=True)


        ''' Input layer of the decoder '''
        name_wi = 'd_w_in'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.num_hidden],trainable=True)

        name_bi = 'd_b_in'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.num_hidden],trainable=True)


        ''' Hidden layer of the decoder '''
        for layer_i in range(self.num_layers):

            name_wi = 'd_w_' + str(layer_i)
            all_weights[name_wi ] = tf.get_variable(name =name_wi,  shape=[self.num_hidden,self.num_hidden],trainable=True)

            name_bi = 'd_b_' + str(layer_i)
            all_weights[name_bi] = tf.get_variable(name =name_bi, shape = [self.num_hidden],trainable=True)


        ''' Final layer of the decoder '''
        name_wi = 'd_w_' + str(self.num_layers)
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.num_hidden, self.input_dim],trainable=True)

        name_bi = 'd_b_' + str(self.num_layers)
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[(self.input_dim)],trainable=True)

        return all_weights
