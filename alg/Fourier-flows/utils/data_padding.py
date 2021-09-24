# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function


import numpy as np
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import itertools
import pickle


def save_model(model, filename):

    with open(filename, "wb") as saved_model_file:

        pickle.dump(model, saved_model_file)


def evaluate_RMSE(y_true, y_pred):

    y_pred_ = list(itertools.chain.from_iterable(y_pred))
    y_true_ = list(itertools.chain.from_iterable(y_true))

    return np.sqrt(np.mean((np.array(y_true_) - np.array(y_pred_)) ** 2))


def padd_arrays(X, max_length=None):

    if len(X[0].shape) == 1:

        X = [X[k].reshape((-1, 1)) for k in range(len(X))]

    if max_length is None:

        max_length = np.max(np.array([len(X[k]) for k in range(len(X))]))

    X_output = [
        np.expand_dims(np.vstack((X[k], np.zeros((max_length - X[k].shape[0], X[0].shape[1])))), axis=0)
        for k in range(len(X))
    ]
    _mask = [
        np.expand_dims(
            np.vstack((np.ones((X[k].shape[0], X[k].shape[1])), np.zeros((max_length - X[k].shape[0], X[0].shape[1])))),
            axis=0,
        )
        for k in range(len(X))
    ]

    return np.concatenate(X_output, axis=0), np.concatenate(_mask, axis=0)


def unpadd_arrays(X, masks):

    masks_lengths = np.sum(masks, axis=1)[:, 0]
    out_ = []

    for k in range(X.shape[0]):

        if len(X.shape) > 2:
            out_.append(X[k, : int(masks_lengths[k]), :])
        else:
            out_.append(X[k, : int(masks_lengths[k])])

    return out_


def get_data_split(X, Y, T, L, indexes):

    X_ = [X[u] for u in indexes]
    Y_ = [Y[u] for u in indexes]
    T_ = [T[u] for u in indexes]
    L_ = [L[u] for u in indexes]

    return X_, Y_, T_, L_
