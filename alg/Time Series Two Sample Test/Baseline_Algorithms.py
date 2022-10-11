# %% Setup
import tensorflow as tf
import numpy as np
from scipy.stats import norm as normal
from tensorflow.python.framework import ops
from numpy.linalg import inv, norm, solve, pinv
from scipy.stats import chi2, f



def train_test_split(T1,X1,T2,X2,train_rate=0.8):
    """
    :param train_rate: fraction of data used for training
    :param parameters: specification for the data generation of two scenarios
    :return:training and testing data for C2ST, note each is a combination of data from two samples
    """

    # %% Data Preprocessing
    dataX1 = np.zeros((X1.shape[0],X1.shape[1],2))
    dataX2 = np.zeros((X2.shape[0], X2.shape[1], 2))


    # Dataset build
    for i in range(len(X1)):
        dataX1[i,:,:] = np.hstack((X1[i,np.newaxis].T,T1[i,np.newaxis].T))
        dataX2[i, :, :] = np.hstack((X2[i, np.newaxis].T, T2[i, np.newaxis].T))

    dataY1 = np.random.choice([0],size=(len(dataX1),));    dataY2 = np.random.choice([1],size=(len(dataX2),))
    dataY1 = dataY1[:,np.newaxis];    dataY2 = dataY2[:,np.newaxis]

    dataX = Permute(np.vstack((dataX1,dataX2)))
    dataY = Permute(np.vstack((dataY1,dataY2)))

    # %% Train / Test Division
    train_size = int(len(dataX) * train_rate)

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataX)])

    return trainX, trainY, testX, testY

def Permute(x,seed=1):
    np.random.seed(seed)
    n = len(x)
    idx = np.random.permutation(n)
    out = x[idx]
    return out


def C2ST(t1,y1,t2,y2,train_rate = 0.5):
    """
    Classifier two sample test following the procedure from Lopez-Paz 2017. We train an RNN that takes
    irregular time points and observations at these points to return a prediction of underlying sampling population

    Note: it requires an equal number of observations in each trajectory but observation times can be arbitrary

    :return:p-value for the hypothesis that two samples were generated from the same underlying stochastic process
    """
    train_rate = train_rate

    # 3. Data Loading
    trainX, trainY, testX, testY = train_test_split(t1,y1,t2,y2,train_rate)

    # %% Main Function
    # 1. Graph Initialization
    ops.reset_default_graph()

    # 2. Parameters
    seq_length = len(trainX[0, :, 0])
    input_size = len(trainX[0, 0, :])
    target_size = len(trainY[0, :])

    learning_rate = 0.01
    iterations = 500
    hidden_layer_size = 10
    batch_size = 64

    # 3. Weights and Bias
    Wr = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Ur = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    br = tf.Variable(tf.zeros([hidden_layer_size]))

    Wu = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Uu = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    bu = tf.Variable(tf.zeros([hidden_layer_size]))

    Wh = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Uh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    bh = tf.Variable(tf.zeros([hidden_layer_size]))

    # Weights for Attention
    Wa1 = tf.Variable(tf.truncated_normal([hidden_layer_size + input_size, hidden_layer_size], mean=0, stddev=.01))
    Wa2 = tf.Variable(tf.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
    ba1 = tf.Variable(tf.truncated_normal([hidden_layer_size], mean=0, stddev=.01))
    ba2 = tf.Variable(tf.truncated_normal([target_size], mean=0, stddev=.01))

    # Weights for output layers
    Wo = tf.Variable(tf.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
    bo = tf.Variable(tf.truncated_normal([target_size], mean=0, stddev=.01))

    # 4. Place holder
    # Target
    Y = tf.placeholder(tf.float32, [None, 1])
    # Input vector with shape[batch, seq, embeddings]
    _inputs = tf.placeholder(tf.float32, shape=[None, None, input_size], name='inputs')


    # Function to convert batch input data to use scan ops of tensorflow.
    def process_batch_input_for_RNN(batch_input):
        batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
        X = tf.transpose(batch_input_)

        return X


    # Processing inputs to work with scan function
    processed_input = process_batch_input_for_RNN(_inputs)

    # Initial Hidden States
    initial_hidden = _inputs[:, 0, :]
    initial_hidden = tf.matmul(initial_hidden, tf.zeros([input_size, hidden_layer_size]))


    # 5. Function for Forward GRU cell.
    def GRU(previous_hidden_state, x):
        # R Gate
        r = tf.sigmoid(tf.matmul(x, Wr) + tf.matmul(previous_hidden_state, Ur) + br)

        # U Gate
        u = tf.sigmoid(tf.matmul(x, Wu) + tf.matmul(previous_hidden_state, Uu) + bu)

        # Final Memory cell
        c = tf.tanh(tf.matmul(x, Wh) + tf.matmul(tf.multiply(r, previous_hidden_state), Uh) + bh)

        # Current Hidden state
        current_hidden_state = tf.multiply((1 - u), previous_hidden_state) + tf.multiply(u, c)

        return current_hidden_state


    # 6. Function to get the hidden and memory cells after forward pass
    def get_states():
        # Getting all hidden state through time
        all_hidden_states = tf.scan(GRU, processed_input, initializer=initial_hidden, name='states')

        return all_hidden_states


    # %% Attention

    # Function to get attention with the last input
    def get_attention(hidden_state):
        inputs = tf.concat((hidden_state, processed_input[-1]), axis=1)
        hidden_values = tf.nn.tanh(tf.matmul(inputs, Wa1) + ba1)
        e_values = (tf.matmul(hidden_values, Wa2) + ba2)

        return e_values


    # Function for getting output and attention coefficient
    def get_outputs():
        all_hidden_states = get_states()

        all_attention = tf.map_fn(get_attention, all_hidden_states)

        a_values = tf.nn.softmax(all_attention, axis=0)

        final_hidden_state = tf.einsum('ijk,ijl->jkl', a_values, all_hidden_states)

        output = tf.nn.sigmoid(tf.matmul(final_hidden_state[:, 0, :], Wo) + bo)

        return output, a_values


    # Getting all outputs from rnn
    outputs, attention_values = get_outputs()

    # reshape out for sequence_loss
    loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - Y)))

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # 3. Sample from the real data (Mini-batch index sampling)
    def sample_X(m, n):
        return np.random.permutation(m)[:n]


    # Training step
    for i in range(iterations):

        idx = sample_X(len(trainX[:, 0, 0]), batch_size)

        Input = trainX[idx, :, :]

        _, step_loss = sess.run([train, loss], feed_dict={Y: trainY[idx], _inputs: Input})

        if i % 100 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))

    # %% Evaluation
    final_outputs, final_attention_values = sess.run([outputs, attention_values], feed_dict={_inputs: testX})

    accuracy = np.mean(np.round(final_outputs)==testY)

    p_value = 1 - normal.cdf(accuracy,1/2,np.sqrt(1/(4*len(testX))))

    return(p_value)


def HOTELLING(t1,y1,t2,y2):
    """
	:param t_1: array of observation times of first population
    :param y_1: array of observation values of first population
    :param t_2: array of observation times of second population
    :param y_2: array of observation values of second population
	"""
    num_ref = 10
    data_1 = data_2 = np.empty(10)

    for i in range(len(t1)):
        temp1 = np.interp(np.linspace(0, 1, num_ref), t1[i], y1[i])
        temp2 = np.interp(np.linspace(0, 1, num_ref), t2[i], y2[i])
        data_1 = np.concatenate((data_1,temp1))
        data_2 = np.concatenate((data_2,temp2))


    data_1 = data_1[num_ref:].reshape(len(t1),num_ref)
    data_2 = data_2[num_ref:].reshape(len(t2),num_ref)

    n_1, n_2 = len(data_1), len(data_2)
    p = data_1.shape[1]
    N = n_1 + n_2 - 2
    cov_1, cov_2 = np.cov(data_1.T), np.cov(data_2.T)
    mean_1, mean_2 = np.mean(data_1,axis=0), np.mean(data_2,axis=0)
    cov  = (n_1*cov_1 + n_2*cov_2)/N
    cov_inv = pinv(cov)

    T_stat = n_1*n_2*np.matmul(np.matmul(mean_2 - mean_1, cov_inv),mean_2 - mean_1)/(n_1+n_2)
    f_coef = (n_1 + n_2 - p - 1)/(N*p)
    df = n_1 + n_2 - p - 1
    p_value = 1 - f.cdf(f_coef*T_stat, p,df)

    return float(p_value)