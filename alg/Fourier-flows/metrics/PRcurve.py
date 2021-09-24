# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function
import numpy as np


def computeF1(X, X_synthetic):

    data_real = [[X[k][u] for k in range(len(X))] for u in range(len(X[0]))]
    real_support = [(np.min(data_real[v]), np.max(data_real[v])) for v in range(len(data_real))]

    data_synth = [[X_synthetic[k][u] for k in range(len(X_synthetic))] for u in range(len(X_synthetic[0]))]
    synth_support = [(np.min(data_synth[v]), np.max(data_synth[v])) for v in range(len(data_synth))]

    precision_t = [
        np.mean((data_synth[k] >= real_support[k][0]) * (data_synth[k] <= real_support[k][1]))
        for k in range(len(data_synth))
    ]
    precision = np.mean(precision_t)

    recall_t = [
        np.mean((data_real[k] >= synth_support[k][0]) * (data_real[k] <= synth_support[k][1]))
        for k in range(len(data_real))
    ]
    recall = np.mean(recall_t)

    F1_score = (2 * precision * recall) / (precision + recall)

    return F1_score
