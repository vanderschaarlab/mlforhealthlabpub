import tensorflow as tf
from pylab import *
from scipy.special import erfc
from tensorflow.python.ops import rnn

from Evaluate import init_random_uniform


def get_quantiles(acquisition_par, fmin, m, s):

    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def _compute_acq(m, s, fmin, jitter):

    phi, Phi, u = get_quantiles(jitter, fmin, m, s)
    f_acqu = s * (u * Phi + phi)
    return f_acqu


def L_cholesky(x, DD):

    jitter = 1e-15

    L_matrix = tf.cholesky(x + jitter*tf.eye(DD, dtype=tf.float32))
    return L_matrix





class SMSDKL(object):
    def __init__(self, data_X, BO_data, bounds,lim_domain, rnn_hidden_size=64,h_size=32, num_h = 32, rho_size=2,lr=0.01):

        self.trainX, self.trainY = BO_data
        self.dims = [np.shape(self.trainX)[1], num_h, num_h, num_h, num_h]
        self.bounds, self.bound_type = bounds[0], bounds[1]
        self.rnn_hidden_size = rnn_hidden_size
        self.num_example = np.shape(data_X)[0]
        self.max_length = np.shape(data_X)[1]
        self.lim_domain = lim_domain
        self.rho_size = rho_size
        self.data_X =  data_X
        self.params = dict()
        self.n_layers = len(self.dims) - 1
        self.lr = lr
        self.feature_vector = {}
        self.ml_primal = {}
        self.ker_inv = {}
        self.mean = {}
        self.weight_deepset = {}
        self.bias_deepset = {}
        self.weight_bo = {}
        self.bias_bo = {}
        self.embed = {}
        self.beta = {}
        self.lam = {}
        self.r = {}
        self.Y = {}
        self.X = {}


        hidden_size = [[self.rnn_hidden_size, h_size], [h_size, h_size], [h_size, h_size], [h_size, h_size], [h_size, rho_size]]


        # create variables
        for i in range(len(hidden_size)):
            self.create_variable('deepset/layer%d' % (i), 'weight', hidden_size[i])

            self.create_variable('deepset/layer%d' % (i), 'biases', hidden_size[i][-1])

            self.weight_deepset[str(i)] = self.get_variable('deepset/layer%d' % (i), 'weight', True)
            self.bias_deepset[str(i)] = self.get_variable('deepset/layer%d' % (i), 'biases', True)



        for i in range(self.n_layers):
            self.create_variable('bo_net/layer%d' % (i), 'weight', [self.dims[i], self.dims[i + 1]])

            self.create_variable('bo_net/layer%d' % (i), 'biases', [self.dims[i + 1]])

            self.weight_bo[str(i)] = self.get_variable('bo_net/layer%d' % (i), 'weight', True)

            self.bias_bo[str(i)] = self.get_variable('bo_net/layer%d' % (i), 'biases', True)




        for i in range(self.max_length):

            self.create_variable('bo_net/task%d' % (i), 'lam', [1, 1])

            self.create_variable('bo_net/task%d' % (i), 'beta', [1, 1])

            lam = self.get_variable('bo_net/task%d' % (i), 'lam', True)
            self.lam[str(i)] = tf.math.softplus(lam)

            beta = self.get_variable('bo_net/task%d' % (i), 'beta', True)
            self.beta[str(i)] = tf.math.softplus(beta)

            self.r[str(i)] = self.beta[str(i)] / self.lam[str(i)]




        self.build_model()


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        for i in range(self.max_length):

            with tf.variable_scope('bo_net/task%d' % (i), reuse=True):
                var6 = tf.get_variable("lam")
                var7 = tf.get_variable("beta")


            var6 = tf.assign(var6, 100 * tf.ones(tf.shape(var6)))
            var7 = tf.assign(var7, 100 * tf.ones(tf.shape(var7)))

            self.sess.run(var6)
            self.sess.run(var7)




    def build_model(self):

        self.X = tf.placeholder(tf.float32, [None, self.dims[0]])

        self.Y = tf.placeholder(tf.float32, [None, self.max_length])

        self.N_sample = tf.placeholder(dtype=tf.float32, shape=[1])


        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.rnn_hidden_size)

        self.segment_ids = tf.placeholder(tf.float32, [self.max_length])

        self.rnn_dim_feature = np.shape(self.data_X)[-1]

        self.rnn_input = tf.constant(value=self.data_X, dtype=tf.float32, shape=[self.num_example,self.max_length, self.rnn_dim_feature]) #

        state = lstm_cell.zero_state(self.num_example, dtype=tf.float32)

        xx = tf.unstack(self.rnn_input, self.max_length, 1)

        rnn_output, state = rnn.static_rnn(lstm_cell, xx, initial_state=state)

        self.train_loss = 0
        self.YY = {}

        for step in range(self.max_length):


            self.YY[str(step)] = tf.slice(self.Y, [0, step], [-1, 1])

            ss = tf.slice(rnn_output, [step, 0, 0], [1, -1, -1])

            self.embed[str(step)] = self.deepset(tf.squeeze(ss))


            state_embedding = tf.tile(self.embed[str(step)], [tf.cast(self.N_sample[0], tf.int32), 1])



            self.bo_net(self.X,state_embedding, step)


            self.train_loss += self.ml_primal[str(step)]




        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.train_loss)



    def batch_optimization(self, max_iter=500):

        num_step = min(10, int(self.max_length))
        for i in range(max_iter):
            rr = np.random.randint(0, self.max_length, num_step)
            ss = np.zeros((self.max_length))
            ss[rr] = 1

            feed_input = {self.Y: self.trainY, self.X: self.trainX, self.N_sample: [self.trainX.shape[0]], self.segment_ids:ss}


            self.sess.run(self.opt, feed_dict = feed_input)


    def update_data(self, query, obs):

        query = np.expand_dims(query,axis=1)
        query = np.swapaxes(query, 0, 1)
        self.trainX = np.concatenate([self.trainX, query], axis=0)

        self.trainY = np.concatenate([self.trainY, obs], axis=0)

        return self.trainX, self.trainY




    def destroy_graph(self):
        tf.reset_default_graph()


    def get_params(self):
        mdict = dict()
        for scope_name, param in self.params.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict


    def create_variable(self, scope, name, shape, trainable=True):

        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable)
        self.params[w.name] = w

    def get_variable(self, scope, name, trainable=True):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
        return w



    def deepset(self, hidden_vec):





        Z0 = tf.add(tf.matmul(hidden_vec, self.weight_deepset['0']), self.bias_deepset['0'])
        A0 = tf.nn.relu(Z0)

        Z1 = tf.add(tf.matmul(A0, self.weight_deepset['1']), self.bias_deepset['1'])
        A1 = Z1

        mean_emb = tf.reduce_mean(A1, axis=0)
        mean_emb = tf.expand_dims(mean_emb,axis=0)

        Z2 = tf.add(tf.matmul(mean_emb, self.weight_deepset['2']), self.bias_deepset['2'])
        A2 = tf.nn.relu(Z2)

        Z3 = tf.add(tf.matmul(A2, self.weight_deepset['3']), self.bias_deepset['3'])
        A3 = tf.nn.relu(Z3)

        Z4 = tf.add(tf.matmul(A3, self.weight_deepset['4']), self.bias_deepset['4'])


        return Z4





    def bo_net(self, feature_vector, state_embedding, step):


        for i in range(self.n_layers):

            feature_vector = tf.nn.tanh(tf.matmul(feature_vector, self.weight_bo[str(i)]) + self.bias_bo[str(i)])



        self.feature_vector[str(step)] = tf.concat([state_embedding,feature_vector],axis=1)

        self.DD = tf.shape(self.feature_vector[str(step)])[1]

        phi_phi = self.r[str(step)] * tf.matmul(tf.transpose(self.feature_vector[str(step)]), self.feature_vector[str(step)])

        Ker = phi_phi + tf.eye(self.DD, dtype=tf.float32)

        L_matrix = L_cholesky(Ker, self.DD)

        L_inv_reduce = tf.linalg.triangular_solve(L_matrix, rhs=tf.eye(self.DD, dtype=tf.float32))

        L_y = tf.matmul(L_inv_reduce, tf.matmul(tf.transpose(self.feature_vector[str(step)]), self.YY[str(step)] ))

        term1 = 0.5 * self.beta[str(step)] * (tf.reduce_sum(tf.square(self.YY[str(step)])) - self.r[str(step)] * tf.reduce_sum(tf.square(L_y)))

        term2 = tf.reduce_sum(tf.log(tf.linalg.diag_part(L_matrix)))

        term3 = -0.5 * self.N_sample[0] * tf.log(self.beta[str(step)])

        self.ml_primal[str(step)] = term1 + term2 + term3

        self.ker_inv[str(step)] = tf.matmul(tf.transpose(L_inv_reduce), L_inv_reduce)

        self.mean[str(step)] = self.r[str(step)] * tf.matmul(tf.transpose(L_inv_reduce), L_y)



    def acquisition(self):


        domain_l = init_random_uniform(self.lim_domain)


        mean_list, XX_inv_list, featureX_list = self.sess.run([self.mean, self.ker_inv, self.feature_vector],
                                              feed_dict={ self.X: self.trainX, self.Y: self.trainY, self.N_sample: [self.trainX.shape[0]],
                                                         self.segment_ids:np.ones((self.max_length))})

        test_X_list = self.sess.run(self.feature_vector, feed_dict={self.X: domain_l, self.N_sample: [np.shape(domain_l)[0]],
                                               self.segment_ids:np.ones((self.max_length))})



        max_ei = np.zeros((self.max_length))
        index_list = []
        ei_total = []

        for i in range(self.max_length):

            lam = self.sess.run(self.lam[str(i)])

            mean, XX_inv, featureX, test_X = mean_list[str(i)], XX_inv_list[str(i)], featureX_list[str(i)], test_X_list[str(i)]



            s = []
            for row in range(test_X.shape[0]):

                x = test_X[[row], :]
                s_save = (1/lam * np.dot(np.dot(x, XX_inv), x.T)) ** 0.5
                s.append(s_save)

            prediction = np.dot(test_X, mean)

            sig = np.reshape(np.asarray(s), (test_X.shape[0], 1))


            ei = _compute_acq(prediction,  sig, np.min(self.trainY[:, i]), 0.01)

            ei = np.squeeze(ei)

            ei_total.append(ei)

            anchor_index = ei.argsort()[-1]


            max_ei[i] = np.max(ei)


            index_list.append(anchor_index)



        prob = np.log(max_ei + 1e-15) - np.log(np.sum(max_ei+ 1e-15))

        prob = np.exp(prob)

        prob = np.ones_like(prob) / (self.max_length)


        final_point = domain_l[index_list[np.where(np.random.multinomial(1, prob) > 0.1)[0][0]]]


        return final_point


