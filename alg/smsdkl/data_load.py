import pickle

import numpy as np
from sklearn.model_selection import KFold


def load_():

    Dataset = pickle.load(open('data/household_power.pickle', 'rb'))
    Dataset = (Dataset - np.mean(Dataset, axis=0)) / (np.std(Dataset, axis=0))
    Dataset = Dataset[:7200, :]
    Dataset = np.reshape(Dataset, [-1, 24, 8])

    Dataset = Dataset[:100,:,:]

    skf = KFold(n_splits=2, random_state=1234)

    train_index, test_index = [[train_index, test_index] for train_index, test_index in skf.split(Dataset)][0]

    train = Dataset[train_index, :]
    test = Dataset[test_index, :]

    trainX, train_y = train[:, :, :-1], train[:, :, -1]
    valid_X, valid_y = test[:, :, :-1], test[:, :, -1]





    return trainX, valid_X, train_y, valid_y


def load_house():

    trainX, testX, label_train, label_test = load_()

    dataX = np.concatenate([trainX, testX], axis=0)
    data_label = np.concatenate([label_train, label_test], axis=0)
    Dataset = np.concatenate([dataX, np.expand_dims(data_label, axis=2)], axis=2)

    return Dataset, dataX, data_label, trainX, testX, label_train, label_test