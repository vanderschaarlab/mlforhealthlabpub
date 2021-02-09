'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''

import os
import argparse
import pickle
import logging

import numpy as np

from simulated_autoregressive import AutoregressiveSimulation
from time_series_deconfounder import test_time_series_deconfounder


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.6, type=float)
    parser.add_argument("--num_simulated_hidden_confounders", default=1, type=int)
    parser.add_argument("--num_substitute_hidden_confounders", default=1, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--exp_name", default='test_tsd_gamma_0.6')
    parser.add_argument("--b_hyperparm_tuning", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = init_arg()

    model_name = 'factor_model'
    hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, model_name)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Simulate dataset
    np.random.seed(100)
    autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
    dataset = autoregressive.generate_dataset(5000, 31)

    dataset_with_confounders_filename = '{}/{}_dataset_with_substitute_confounders.txt'.format(args.results_dir,
                                                                                               args.exp_name)
    factor_model_hyperparams_file = '{}/{}_factor_model_best_hyperparams.txt'.format(args.results_dir, args.exp_name)

    test_time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
                                  exp_name=args.exp_name,
                                  dataset_with_confounders_filename=dataset_with_confounders_filename,
                                  factor_model_hyperparams_file=factor_model_hyperparams_file,
                                  b_hyperparm_tuning=args.b_hyperparm_tuning)
