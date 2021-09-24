# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function
import numpy as np

from models.sequential import RNNmodel


def train_RNN(X_synth):

    predictive_model = RNNmodel(mode="LSTM", HIDDEN_UNITS=100, NUM_LAYERS=2, MAX_STEPS=100, INPUT_SIZE=1)

    X_ = [X_synth[k][: len(X_synth[k]) - 1] for k in range(len(X_synth))]
    Y_ = [X_synth[k][1:] for k in range(len(X_synth))]

    predictive_model.fit(X_, Y_, verbosity=False)

    return predictive_model


def computeMAE(X_real, X_synth):

    X_true = [X_real[k][: len(X_real[k]) - 1] for k in range(len(X_real))]
    Y_true = [X_real[k][1:] for k in range(len(X_real))]

    predictive_model = train_RNN([X_synth[k] for k in range(len(X_synth))])

    X_pred = predictive_model.predict(X_true)
    MAEscore = np.mean(np.abs(np.array(X_pred) - np.array(Y_true)))

    return MAEscore
