# coding: utf-8
# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
import scipy.stats as st
import pickle

import warnings

warnings.filterwarnings("ignore")


from SequentialFlows import FourierFlow, RealNVP

from metrics.PRcurve import computeF1
from metrics.MAE import computeMAE


def MinMaxScaler(data):
    """Min Max normalizer.
  
    Args:
    - data: original data
  
    Returns:
    - norm_data: normalized data
    """

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)

    return norm_data


def real_data_loading(data_name, seq_len):

    """Load and preprocess real-world datasets.
  
    Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
    Returns:
    - data: preprocessed data.
    """
    assert data_name in ["stock", "energy", "lung"]

    if data_name == "stock":
        ori_data = np.loadtxt("data/stock_data.csv", delimiter=",", skiprows=1)
    elif data_name == "energy":
        ori_data = np.loadtxt("data/energy_data.csv", delimiter=",", skiprows=1)
    elif data_name == "lung":
        ori_data = pickle.load(open("data/lung_cancer.p", "rb"))

    if data_name in ["stock", "energy"]:

        # Flip the data to make chronological data
        ori_data = ori_data[::-1]

        # Normalize the data
        ori_data = MinMaxScaler(ori_data)

        # Preprocess the dataset
        temp_data = []

        # Cut data by sequence length
        for i in range(0, len(ori_data) - seq_len):

            _x = ori_data[i : i + seq_len]
            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        data = []

        for i in range(len(temp_data)):

            data.append(temp_data[idx[i]])

    # stock data
    if data_name == "stock":

        # X   = [np.hstack((0, data[k][:, 4])) for k in range(len(data))]
        X = [np.hstack((0, data[k][:, 0])) for k in range(len(data))]

    # energy data
    if data_name == "energy":
        X = [np.hstack((0, data[k][:, 0])) for k in range(len(data))]

    # lung data
    if data_name == "lung":
        X = [np.hstack((0, ori_data[k])) for k in range(len(ori_data))]
        X = [X[k] for k in range(2000)]

    return X


def mean_confidence_interval(data, confidence=0.95):

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h


# List model hyper-parameters

FF_model_params = dict(
    {
        "stock": dict({"hidden": 200, "n_flows": 3, "normalize": True}),
        "energy": dict({"hidden": 200, "n_flows": 5, "normalize": True}),
        "lung": dict({"hidden": 200, "n_flows": 5, "normalize": True}),
    }
)

FF_train_params = dict(
    {
        "stock": dict({"epochs": 1000, "batch_size": 500, "learning_rate": 1e-3, "display_step": 100}),
        "energy": dict({"epochs": 1500, "batch_size": 1000, "learning_rate": 1e-3, "display_step": 100}),
        "lung": dict({"epochs": 500, "batch_size": 2000, "learning_rate": 1e-3, "display_step": 100}),
    }
)

RNVP_model_params = dict(
    {
        "stock": dict({"hidden": 200, "n_flows": 5}),
        "energy": dict({"hidden": 200, "n_flows": 5}),
        "lung": dict({"hidden": 200, "n_flows": 5}),
    }
)

RNVP_train_params = dict(
    {
        "stock": dict({"epochs": 500, "batch_size": 500, "learning_rate": 1e-4, "display_step": 100}),
        "energy": dict({"epochs": 500, "batch_size": 500, "learning_rate": 1e-3, "display_step": 100}),
        "lung": dict({"epochs": 500, "batch_size": 500, "learning_rate": 1e-3, "display_step": 100}),
    }
)

TimeGAN_model_params = dict(
    {
        "stock": dict({"module": "gru", "hidden_dim": 24, "num_layer": 3, "iterations": 500, "batch_size": 128}),
        "energy": dict({"module": "gru", "hidden_dim": 12, "num_layer": 3, "iterations": 100, "batch_size": 128}),
        "lung": dict({"module": "gru", "hidden_dim": 24, "num_layer": 3, "iterations": 500, "batch_size": 128}),
    }
)

model_parameters = dict(
    {"Fourier flow": FF_model_params, "RealNVP": RNVP_model_params, "TimeGAN": TimeGAN_model_params}
)

train_parameters = dict({"Fourier flow": FF_train_params, "RealNVP": RNVP_train_params})


# main experiments


def run_experiments(T, data_sets, baselines, num_experiments=5, n_samples=10000):

    F1_scores = dict.fromkeys(data_sets)
    MAE_scores = dict.fromkeys(data_sets)

    for dataset in data_sets:

        F1_scores[dataset] = dict.fromkeys(baselines)
        MAE_scores[dataset] = dict.fromkeys(baselines)

        for baseline in baselines:

            F1_scores[dataset][baseline] = []
            MAE_scores[dataset][baseline] = []

    for dataset in data_sets:

        print("Dataset: ", dataset)
        print("-----------------------")

        X = real_data_loading(data_name=dataset, seq_len=T)

        for baseline in baselines:

            print("Baseline: ", baseline)
            print("-----------------------")

            for k in range(num_experiments):

                print("Experiment number: ", k)

                if baseline == "Fourier flow":

                    model = FourierFlow(**model_parameters[baseline][dataset], fft_size=T + 1)

                elif baseline == "RealNVP":

                    model = RealNVP(**model_parameters[baseline][dataset], T=T + 1)
                
                else:

                    raise ValueError(f"Baseline {baseline} not implemented.")

                _ = model.fit(X, **train_parameters[baseline][dataset])

                F1_scores[dataset][baseline].append(computeF1(X, model.sample(n_samples)))
                MAE_scores[dataset][baseline].append(computeMAE(X, model.sample(n_samples)))

                print("F1 score", F1_scores[dataset][baseline][-1])
                print("MAE score", MAE_scores[dataset][baseline][-1])

    return F1_scores, MAE_scores


# main function


def main(args):

    data_sets = args.data_sets
    baselines = args.baselines

    F1_scores, MAE_scores = run_experiments(
        args.T, data_sets, baselines, num_experiments=args.n_exps, n_samples=args.n_samples
    )

    for dataset in data_sets:

        print("Results for ", dataset)

        for baseline in baselines:
            print(baseline + "F1 scores: ", mean_confidence_interval(F1_scores[dataset][baseline]))
            print(baseline + "MAE scores: ", mean_confidence_interval(MAE_scores[dataset][baseline]))


if __name__ == "__main__":

    default_data_sets = ["stock", "energy", "lung"]
    default_baselines = ["Fourier flow", "RealNVP"]
    parser = argparse.ArgumentParser(description="Fourier Flows")

    parser.add_argument("-m", "--data-sets", nargs="+", default=default_data_sets)
    parser.add_argument("-s", "--baselines", nargs="+", default=default_baselines)
    parser.add_argument("-t", "--T", default=100, type=int)
    parser.add_argument("-n", "--n-samples", default=10000, type=int)
    parser.add_argument("-e", "--n-exps", default=5, type=int)

    args = parser.parse_args()

    main(args)
