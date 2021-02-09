import random
import time
from copy import deepcopy

from keras import backend as K

K.tensorflow_backend._get_available_gpus()
import numpy as np
from keras.layers import Dense
from keras.regularizers import l2
import keras
from keras.models import Model
from keras.layers import Input, LSTM


def error_function(BO_output):

    rmse1 = np.min(np.mean(BO_output, axis=1), axis=0)
    rmse2 = np.mean(np.min(BO_output, axis=0))

    return rmse1,rmse2



def get_opt_domain():

    domain = get_hyperparameter_space()

    dim = len(domain)

    bounds = []
    bounds_type = []
    for i_domain in domain:
        bounds.append([i_domain['domain'][0], i_domain['domain'][-1]])
        bounds_type.append(i_domain['type'])

    bb = [bounds, bounds_type]
    return domain, dim, bb




def init_random_uniform(domain, n_points=25000,initial=False):


    list = []

    for k in range(int(n_points)):

        if initial:
            random.seed(k)
        else:
            random.seed(time.time())


        list_i = []
        for i_domain in domain:

            if i_domain['type'] == 'continuous':
                kk = float(random.uniform(i_domain['domain'][0], i_domain['domain'][1]))
                list_i.append(kk)
            else:
                list_i.append(int(random.sample(i_domain['domain'], 1)[0]))

        list.append(list_i)


    return list


def min_list(obs):

    obs = -obs[:,-1]
    leng = len(obs)
    list = []
    a = obs[0]
    list.append(a)
    for i in range(1, leng):
        if obs[i] <= a:
            list.append(obs[i])
            a = deepcopy(obs[i])
        else:
            list.append(a)
    return list



def get_hyperparameter_space():
    hyp_ = [{'name': 'RNN.hidden_size', 'type': 'discrete', 'domain':list(range(10, 251, 1)), 'dimensionality': 1},
            {'name': 'RNN.dropout_rate', 'type': 'discrete', 'domain': list(range(10, 91, 1)), 'dimensionality': 1},
            {'name': 'RNN.l2', 'type': 'continuous', 'domain': [-20, 1], 'dimensionality': 1},
            {'name': 'RNN.num_epoch', 'type': 'discrete', 'domain': list(range(10, 101, 1)), 'dimensionality': 1},
            {'name': 'RNN.batch_size', 'type': 'discrete', 'domain': list(range(10, 101, 1)), 'dimensionality': 1},
            {'name': 'RNN.recurrent_dropout_rate', 'type': 'discrete', 'domain': list(range(10, 91, 5)), 'dimensionality': 1}]

    return hyp_


def evaluate(train_X, test_X, train_y, test_y, list_domain):



        performance_list = []
        for i in range(np.shape(list_domain)[0]):
            performance_list.append(evaluate_reg_keras(train_X, test_X, train_y, test_y, list_domain[i]))

        obs = np.array(performance_list)

        return obs







def evaluate_reg_keras(train_X,test_X, train_y, test_y, param):


    num_units = int(param[0])
    dropout_rate = param[1]/100
    l2_rate = np.exp(param[2])
    epochs = int(param[3])
    batch = int(param[4])
    recurrent_dropout = param[5] / 100

    max_length = np.shape(train_y)[1]


    train_y = np.expand_dims(train_y,axis=2)

    test_y = np.expand_dims(test_y, axis=2)

    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm1 = LSTM(num_units, kernel_regularizer=l2(l2_rate), return_sequences=True,
                dropout=dropout_rate, recurrent_dropout= recurrent_dropout )(inputs)

    pred = Dense(1)(lstm1)


    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


    model = Model(inputs=inputs, outputs=pred)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.fit(train_X, train_y, epochs=epochs, batch_size=batch, verbose=0, shuffle=False)


    rmse_list = np.zeros((10, max_length))

    for row in range(10):

        model.fit(train_X, train_y, epochs=1, batch_size=batch, verbose=0, shuffle=False)

        for i in range(max_length):

            yhat = model.predict(test_X)

            rmse = np.sqrt(np.mean(np.square(test_y[:, i, 0] - yhat[:, i, 0]),axis=0))

            rmse_list[row, i] = rmse

    return np.mean(rmse_list, axis=0)



