'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''
import logging
import numpy as np
import os
import shutil

from sklearn.model_selection import ShuffleSplit

from utils.evaluation_utils import write_results_to_file
from factor_model import FactorModel
from rmsn.script_rnn_fit import rnn_fit
from rmsn.script_rnn_test import rnn_test
from rmsn.script_propensity_generation import propensity_generation


def train_factor_model(dataset_train, dataset_val, dataset, num_confounders, hyperparams_file,
                       b_hyperparameter_optimisation):
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 50
    best_validation_loss = 100
    if b_hyperparameter_optimisation:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = np.random.choice([32, 64, 128, 256])
            hyperparams['fc_hidden_units'] = np.random.choice([32, 64, 128])
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001, 0.0001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = FactorModel(params, hyperparams)
            model.train(dataset_train, dataset_val)
            validation_loss = model.eval_network(dataset_val)

            if (validation_loss < best_validation_loss):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_loss, validation_loss))
                best_validation_loss = validation_loss
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        best_hyperparams = {
            'rnn_hidden_units': 128,
            'fc_hidden_units': 128,
            'learning_rate': 0.001,
            'batch_size': 128,
            'rnn_keep_prob': 0.8}

    model = FactorModel(params, best_hyperparams)
    model.train(dataset_train, dataset_val)
    predicted_confounders = model.compute_hidden_confounders(dataset)

    return predicted_confounders


def get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map


def train_rmsn(dataset_map, model_name, b_use_predicted_confounders):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    if not os.path.exists(MODEL_ROOT):
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")
    else:
        # Need to delete previously saved model.
        shutil.rmtree(MODEL_ROOT)
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")

    rnn_fit(dataset_map=dataset_map, networks_to_train='propensity_networks', MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

    propensity_generation(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                          b_use_predicted_confounders=b_use_predicted_confounders)

    rnn_fit(networks_to_train='encoder', dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

    rmsn_mse = rnn_test(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                        b_use_predicted_confounders=b_use_predicted_confounders)

    rmse = np.sqrt(np.mean(rmsn_mse)) * 100
    return rmse


def test_time_series_deconfounder(dataset, num_substitute_confounders, exp_name, dataset_with_confounders_filename,
                                  factor_model_hyperparams_file, b_hyperparm_tuning=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=False)

    dataset_train = dataset_map['training_data']
    dataset_val = dataset_map['validation_data']

    logging.info("Fitting factor model")
    predicted_confounders = train_factor_model(dataset_train, dataset_val,
                                               dataset,
                                               num_confounders=num_substitute_confounders,
                                               b_hyperparameter_optimisation=b_hyperparm_tuning,
                                               hyperparams_file=factor_model_hyperparams_file)

    dataset['predicted_confounders'] = predicted_confounders
    write_results_to_file(dataset_with_confounders_filename, dataset)

    dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)

    logging.info('Fitting counfounded recurrent marginal structural networks.')
    rmse_without_confounders = train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False)

    logging.info(
        'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(num_substitute_confounders))
    rmse_with_confounders = train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True)

    print("Outcome model RMSE when trained WITHOUT the hidden confounders.")
    print(rmse_without_confounders)

    print("Outcome model RMSE when trained WITH the substitutes for the hidden confounders.")
    print(rmse_with_confounders)
