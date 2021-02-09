import os

from helper import hist_embedding
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops import rnn
from tqdm import tqdm


class GRV_S(object):
    def __init__(self, x_list,a_list,  m_rnn_hidden=32, dtr_rnn_hidden=32, m_hidden=32,
                 dtr_hidden=32, lr=0.001,Lambda =1.0, maximize = True):

        self.dtr_rnn_hidden, self.dtr_hidden = dtr_rnn_hidden, dtr_hidden
        self.m_rnn_hidden, self.m_hidden = m_rnn_hidden, m_hidden
        _, self.max_length, self.n_treatments = np.shape(a_list)
        self.start, self.Lambda,  self.lr = True, Lambda, lr
        self.coff = np.sign(0.5- float(maximize))
        _, _, self.n_features = np.shape(x_list)

        output_g, output_v, output_dtr = self.initial()
        data_slice = self.slice_data()
        self.d_part(output_dtr, data_slice)
        self.v_part(output_g, output_v, data_slice)
        self.losses()


    def d_part(self, output_dtr, data_slice):

        self.d_matrix, self.prob_pred = [], []

        for step in range(self.max_length):

            x_t, _, a_t, _ = data_slice[step]
            embed_dtr_t = hist_embedding(output_dtr, step)
            p_t, d_t = self.dtr_net(embed_dtr_t, x_t)
            self.prob_pred.append(p_t)
            self.d_matrix.append(tf.expand_dims(d_t, axis=2))

        self.d_matrix = tf.transpose(tf.concat(self.d_matrix, axis=2), [0, 2, 1])


    def v_part(self, output_g, output_v, data_slice):


        self.mu_d, self.mu_d_e = {}, {}
        self.L_0, self.R, self.prob_matrix = 0, 0, []


        for step in range(self.max_length)[::-1]:

            x_t, y_t, a_t, a_t_T = data_slice[step]
            d_t_T = tf.slice(self.d_matrix, [0, step, 0], [-1, -1, -1])
            embed_v_t = hist_embedding(output_v, step)
            embed_g_t = hist_embedding(output_g, step)

            g_t = self.g_net(embed_g_t, x_t)
            _, mu_tt = self.mu_net(embed_v_t, x_t, a_t_T, step)
            self.mu_d[str(step)], mu_tt_d = self.mu_net(embed_v_t, x_t, d_t_T, step)
            self.L_0 +=  tf.reduce_mean(tf.square(y_t - mu_tt)) + tf.reduce_mean(tf.reduce_sum(-tf.multiply(a_t, tf.log(g_t)),axis=1))

            delta_tt = tf.squeeze(tf.slice(self.d_matrix, [0, step, 0], [-1, 1, -1]), axis=1)
            delta_tt = tf.reduce_sum(a_t * delta_tt, axis=1)
            self.R += tf.reduce_mean(tf.square(y_t - mu_tt_d - self.m_params['epsilon_' + str(0)] *delta_tt/ tf.reduce_sum(g_t * a_t, axis=1)))
            self.prob_matrix.append(tf.reduce_sum(g_t * a_t, axis=1, keepdims=True))


        self.prob_matrix = tf.concat(self.prob_matrix[::-1], axis=1)

        delta = tf.reduce_sum(self.d_matrix * self.treatments, axis=2)

        prob_cum, delta_cum, step = {}, {}, 0

        for m in range(step + 1, self.max_length + 1):
            prob_cum[str(step) + '_' + str(m - 1)] = tf.reduce_prod(tf.slice(self.prob_matrix, [0, 0], [-1, m - step]), axis=1)
            delta_cum[str(step) + '_' + str(m - 1)] = tf.reduce_prod(tf.slice(delta, [0, 0], [-1, m - step]), axis=1)

        for r in range(step, self.max_length):
            for k in range(step, r + 1):

                delta_g_cum = tf.constant(0.0, shape=(1,))
                for m in range(k, r + 1):
                    delta_g_cum += delta_cum[str(step) + '_' + str(m)] / prob_cum[str(step) + '_' + str(m)]

                mu_k_r = tf.squeeze(tf.slice(self.mu_d[str(k)], [0, r - k], [-1, 1]), axis=1)

                self.mu_d_e[str(step) + '_' + str(k) + '_' + str(r)] = mu_k_r + self.m_params['epsilon_' + str(step)] * delta_g_cum

        for r in range(step, self.max_length):
            for s in range(step, r - 1):
                zeta_s_r_t = tf.reduce_mean(tf.square(self.mu_d_e[str(step) + '_' + str(s + 1) + '_' + str(r)]
                                                      - self.mu_d_e[str(step) + '_' + str(s) + '_' + str(r)]))

                self.R += zeta_s_r_t

        self.loss_v_sum = self.L_0 + self.Lambda * self.R


    def losses(self):

        delta_1_T = tf.cumprod(tf.reduce_sum(self.d_matrix * self.treatments, axis=2), axis=1)

        self.weight__ = self.mu_d[str(0)] + self.m_params['epsilon_' + str(0)] \
                    * tf.cumsum(  tf.cumprod(tf.reduce_sum(self.d_matrix * self.treatments, axis=2), axis=1) / tf.cumprod(self.prob_matrix, axis=1), axis=1)

        #Remove delta_1_T if the treatment rules do not depend on the treatments in the history and the treatment has no influence on the future covariates
        self.loss_dtr_sum = tf.reduce_sum(delta_1_T * self.weight__, axis=1)
        self.loss_dtr_sum = self.coff * tf.reduce_mean(self.loss_dtr_sum)

        var_value = [var for var in tf.trainable_variables() if 'value' in var.name]
        var_dtr = [var for var in tf.trainable_variables() if 'dtr' in var.name]

        self.opt_v_sum = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_v_sum, var_list=var_value)
        self.opt_v_sum_0 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.L_0, var_list=var_value)
        self.opt_dtr_sum = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_dtr_sum, var_list=var_dtr)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def fit(self, rnn_X, treatments, rnn_Y,batch_size=50, num_initial=500, num_d=100, num_v=5):

        train_size = np.shape(rnn_X)[0]
        num_batch = int(train_size / batch_size)
        n, _, d = np.shape(treatments)

        if self.start:
            self.initial_fit(rnn_X, treatments, rnn_Y, batch_size, num_epochs=num_initial)

        for _ in range(num_d):
            perm = np.random.permutation(train_size)
            rnn_X = rnn_X[perm, :, :]
            treatments = treatments[perm, :, :]
            rnn_Y = rnn_Y[perm, :]

            for i in range(num_batch):
                batch_index = range(i * batch_size, (i + 1) * batch_size)
                batch_rnn_x = rnn_X[batch_index, :, :]
                batch_treatments = treatments[batch_index, :, :]
                batch_y = rnn_Y[batch_index, :]
                feed_input_2 = {self.rnn_x: batch_rnn_x,
                                self.treatments: batch_treatments,
                                self.y: batch_y}

                self.sess.run(self.opt_dtr_sum, feed_dict=feed_input_2)

            for _ in range(num_v):
                perm = np.random.permutation(train_size)
                rnn_X = rnn_X[perm, :, :]
                treatments = treatments[perm, :, :]
                rnn_Y = rnn_Y[perm, :]

                for i in range(num_batch):
                    batch_index = range(i * batch_size, (i + 1) * batch_size)
                    batch_rnn_x = rnn_X[batch_index, :, :]
                    batch_treatments = treatments[batch_index, :, :]
                    batch_y = rnn_Y[batch_index, :]
                    feed_input_1 = {self.rnn_x: batch_rnn_x,
                                    self.treatments: batch_treatments,
                                    self.y: batch_y}

                    loss, _ = self.sess.run([self.loss_v_sum, self.opt_v_sum], feed_dict=feed_input_1)


    def initial_fit(self, rnn_X, treatments, rnn_Y, batch_size, num_epochs):
        # minimize the loss term that has nothing to do with d.
        train_size = np.shape(rnn_X)[0]
        num_batch = int(train_size / batch_size)
        n, _, d = np.shape(treatments)

        for _ in tqdm(range(num_epochs)):
            perm = np.random.permutation(train_size)
            rnn_X = rnn_X[perm, :, :]
            treatments = treatments[perm, :, :]
            rnn_Y = rnn_Y[perm, :]

            for i in range(num_batch):
                batch_index = range(i * batch_size, (i + 1) * batch_size)
                batch_rnn_x = rnn_X[batch_index, :, :]
                batch_treatments = treatments[batch_index, :, :]
                batch_y = rnn_Y[batch_index, :]

                feed_input_1 = {self.rnn_x: batch_rnn_x,
                                self.treatments: batch_treatments,
                                self.y: batch_y}

                self.sess.run(self.opt_v_sum_0, feed_dict=feed_input_1)

        self.start = False


    def g_net(self, embed_m_t, x_t):

        Z_in = tf.concat([embed_m_t,x_t],axis=1)
        Z0 = tf.nn.selu(tf.add(tf.matmul(Z_in, self.m_params['w_h0']), self.m_params['b_h0']))
        Z1 = tf.nn.selu(tf.add(tf.matmul(Z0,self.m_params['w_h1']), self.m_params['b_h1']))
        Z2 = tf.add(tf.matmul(Z1, self.m_params['w_propensity']), self.m_params['b_propensity'])
        prob = (tf.nn.softmax(Z2) + 0.01) / 1.02

        return prob


    def mu_net(self, embed_t, x_t, a_t_T, step):

        x_t_ = tf.concat([tf.expand_dims(x_t, axis=2)] * (self.max_length - step), axis=2)
        x_t_ = tf.transpose(x_t_, [0, 2, 1])
        input_decoder = tf.concat([x_t_, a_t_T], axis=2)

        with tf.variable_scope("rnn_value_decoder",reuse=tf.AUTO_REUSE):
            out, _ = rnn.dynamic_rnn(cell=self.lstm_cell_decoder, inputs=input_decoder,initial_state=embed_t,  dtype=tf.float32)

        out_ = []
        for t in range (self.max_length-step):

            out_t = tf.squeeze(tf.slice(out,[0,t,0],[-1,1,-1]),axis=1)
            out_t = tf.nn.selu(tf.add(tf.matmul(out_t, self.m_params['w_outcome_' + str(1)]), self.m_params['b_outcome_' + str(1)]))
            out_t = tf.nn.selu(tf.add(tf.matmul(out_t, self.m_params['w_outcome_' + str(2)]), self.m_params['b_outcome_' + str(2)]))
            out_t = tf.add(tf.matmul(out_t, self.m_params['w_outcome_' + str(3)]), self.m_params['b_outcome_' + str(3)])
            out_.append(out_t)

        out = tf.concat(out_,axis=1)

        return out, out_[0]


    def dtr_net(self, embed_dtr_t, x_t):

        Z_in = tf.concat([embed_dtr_t, x_t], axis=1)
        Z0 =  tf.nn.selu(tf.add(tf.matmul(Z_in, self.dtr_params['w_h0']), self.dtr_params['b_h0']))
        Z1 =  tf.nn.selu(tf.add(tf.matmul(Z0, self.dtr_params['w_h1']), self.dtr_params['b_h1']))
        Z2 = tf.add(tf.matmul(Z1, self.dtr_params['w_prob']), self.dtr_params['b_prob'])
        prob = (tf.nn.softmax(Z2) + 0.01) / 1.02

        c_sample = prob

        return prob, c_sample


    def slice_data(self):

        data_slice = []

        for step in range(self.max_length):

            x_t = tf.squeeze(tf.slice(self.rnn_x, [0, step, 0], [-1, 1, -1]), axis=1)
            a_t = tf.squeeze(tf.slice(self.treatments, [0, step, 0], [-1, 1, -1]), axis=1)
            a_t_T = tf.slice(self.treatments, [0, step, 0], [-1, -1, -1])
            y_t = tf.slice(self.y, [0, step], [-1, 1])
            data_slice.append([x_t, y_t, a_t, a_t_T])

        return data_slice


    def predict(self, rnn_X, treatments):

        prob_,_ = self.sess.run([self.prob_pred,self.d_matrix], feed_dict={self.rnn_x: rnn_X, self.treatments: treatments})
        prob_ = np.swapaxes(np.array(prob_),0,1)
        d_matrix = np.zeros_like(prob_)

        for i in range(np.shape(rnn_X)[0]):
            for k in range(self.max_length):
                index = np.argmax(prob_[i, k, :])
                d_matrix[i, k, index] = 1

        return prob_,d_matrix


    def initial(self):

        self.rnn_x = tf.placeholder(tf.float32, [None, self.max_length, self.n_features])
        self.treatments = tf.placeholder(tf.float32, [None, self.max_length, self.n_treatments])
        self.y = tf.placeholder(tf.float32, [None, self.max_length])
        rnn_input = tf.concat([self.treatments, self.rnn_x], axis=2)

        self.m_params = {
            'w_h0': tf.Variable(tf.random_normal([self.m_rnn_hidden + self.n_features, self.m_hidden], stddev=0.1),name='value_who'),
            'b_h0': tf.Variable(tf.random_normal([self.m_hidden], stddev=0.1),name='value_bho'),
            'w_h1': tf.Variable(tf.random_normal([self.m_hidden, self.m_hidden], stddev=0.1),name='value_wh1'),
            'b_h1': tf.Variable(tf.random_normal([self.m_hidden], stddev=0.1),name='value_bh1'),
            'w_propensity': tf.Variable(tf.random_normal([self.m_hidden, self.n_treatments], stddev=0.1),name='value_w_propensity'),
            'b_propensity': tf.Variable(tf.random_normal([self.n_treatments], stddev=0.1),name='value_b_propensity'),
            'w_outcome_1' : tf.Variable(tf.random_normal([self.m_rnn_hidden, self.m_rnn_hidden],  stddev=0.1),name='value_w_outcome_1'),
            'b_outcome_1' : tf.Variable(tf.random_normal([self.m_rnn_hidden],stddev=0.1),name='value_b_outcome_1'),
            'w_outcome_2' : tf.Variable(tf.random_normal([self.m_rnn_hidden, self.m_rnn_hidden],  stddev=0.1),name='value_w_outcome_2'),
            'b_outcome_2' : tf.Variable(tf.random_normal([self.m_rnn_hidden],stddev=0.1),name='value_b_outcome_2'),
            'w_outcome_3' :tf.Variable(tf.random_normal([self.m_rnn_hidden, 1],  stddev=0.1),name='value_w_outcome_3'),
            'b_outcome_3' : tf.Variable(tf.random_normal([1],stddev=0.1),name='value_b_outcome__3')
        }

        self.m_params['epsilon_' + str(0)] = tf.Variable(tf.random_normal([1], stddev=0.1),name='value_epsilon')
        self.m_params['epsilon_' + str(0)] = self.m_params['epsilon_' + str(0)]/10000

        self.dtr_params = {
            'w_h0': tf.Variable(tf.random_normal([self.dtr_rnn_hidden + self.n_features, self.dtr_hidden], stddev=0.1),name='dtr_who'),
            'b_h0': tf.Variable(tf.random_normal([self.dtr_hidden], stddev=0.1),name='dtr_bho'),
            'w_h1': tf.Variable(tf.random_normal([self.dtr_hidden, self.dtr_hidden], stddev=0.1),name='dtr_wh1'),
            'b_h1': tf.Variable(tf.random_normal([self.dtr_hidden], stddev=0.1),name='dtr_bh1'),
            'w_prob': tf.Variable(tf.random_normal([self.dtr_hidden, self.n_treatments], stddev=0.1), name='dtr_wprob'),
            'b_prob': tf.Variable(tf.random_normal([self.n_treatments], stddev=0.1),name='dtr_bprob')
        }


        self.lstm_cell_m = DropoutWrapper(tf.contrib.rnn.GRUCell(self.m_rnn_hidden),
                                          output_keep_prob=0.5,
                                           state_keep_prob=0.5,
                                           variational_recurrent=True, dtype=tf.float32)

        self.lstm_cell_g = DropoutWrapper(tf.contrib.rnn.GRUCell(self.m_rnn_hidden),
                                          output_keep_prob=0.5,
                                           state_keep_prob=0.5,
                                           variational_recurrent=True, dtype=tf.float32)

        self.lstm_cell_dtr = DropoutWrapper(tf.contrib.rnn.GRUCell(self.dtr_rnn_hidden),
                                            output_keep_prob=0.5,
                                            state_keep_prob=0.5,
                                            variational_recurrent=True, dtype=tf.float32)

        self.lstm_cell_decoder = DropoutWrapper(tf.contrib.rnn.GRUCell(self.m_rnn_hidden),
                                                output_keep_prob=0.5,
                                                state_keep_prob=0.5,
                                                variational_recurrent=True, dtype=tf.float32)

        with tf.variable_scope("rnn_g"):
            output_g,_ = rnn.dynamic_rnn(cell=self.lstm_cell_g, inputs=rnn_input, dtype=tf.float32)

        with tf.variable_scope("rnn_value"):
            output_v, _ = rnn.dynamic_rnn(cell=self.lstm_cell_m, inputs=rnn_input, dtype=tf.float32)

        with tf.variable_scope("rnn_dtr"):
            output_dtr, _ = rnn.dynamic_rnn(cell=self.lstm_cell_dtr, inputs=rnn_input, dtype=tf.float32)

        return output_g, output_v, output_dtr


    def save_network(self, data_name, exp_index, destroy=False):

        checkpoint_name = 'GRVS_' + data_name + '_exp_' + str(exp_index)
        model_dir = "GRVS_model/" + data_name + '/exp_' + str(exp_index)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name)))

        if destroy:
            self.destroy_graph()


    def load_network(self, data_name, exp_index):

        checkpoint_name = 'GRVS_' + data_name + '_exp_' + str(exp_index)
        model_dir = "GRVS_model/" + data_name + '/exp_' + str(exp_index)

        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name)))


    def destroy_graph(self):
        tf.reset_default_graph()

