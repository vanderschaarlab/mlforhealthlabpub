# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function


import numpy as np
import torch
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def autoregressive(X, w):

    return np.array([np.sum(X[0 : k + 1] * np.flip(w[0 : k + 1]).reshape(-1, 1)) for k in range(len(X))])


def create_autoregressive_data(
    n_samples=100,
    seq_len=6,
    n_features=1,
    X_m=1,
    X_v=2,
    noise_profile=[0, 0.2, 0.4, 0.6, 0.8, 1],
    memory_factor=0.9,
    mode="time-dependent",
):

    # Create the input features

    X = [np.random.normal(X_m, X_v, (seq_len, n_features)) for k in range(n_samples)]
    w = np.array([memory_factor ** k for k in range(seq_len)])

    if mode == "noise-sweep":

        Y = [
            [
                (
                    autoregressive(X[k], w).reshape(seq_len, n_features)
                    + np.random.normal(0, noise_profile[u], (seq_len, n_features))
                ).reshape(seq_len,)
                for k in range(n_samples)
            ]
            for u in range(len(noise_profile))
        ]

    elif mode == "time-dependent":

        Y = [
            (
                autoregressive(X[k], w).reshape(seq_len, n_features)
                + (torch.normal(mean=0.0, std=torch.tensor(noise_profile))).detach().numpy().reshape(-1, n_features)
            ).reshape(seq_len,)
            for k in range(n_samples)
        ]

    return X, Y
