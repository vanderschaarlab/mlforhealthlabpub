'''
Title: Estimating counterfactual treatment outcomes over time through adversarially balanced representations
Authors: Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
International Conference on Learning Representations (ICLR) 2020

Last Updated Date: January 15th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''

import logging
import pickle
import numpy as np

from utils.evaluation_utils import get_processed_data, get_mse_at_follow_up_time, \
    load_trained_model, write_results_to_file
from CRN_model import CRN_Model


def fit_CRN_decoder(dataset_train, dataset_val, model_name, model_dir,
                    encoder_hyperparams_file, decoder_hyperparams_file,
                    b_hyperparam_opt):
    logging.info("Fitting CRN decoder.")

    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]
    num_inputs = dataset_train['current_covariates'].shape[-1] + dataset_train['current_treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 30
    best_validation_mse = 1000000

    with open(encoder_hyperparams_file, 'rb') as handle:
        encoder_best_hyperparams = pickle.load(handle)

    if b_hyperparam_opt:
        logging.info("Performing hyperparameter optimization.")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            # The first rnn hidden state in the decoder is initialized with the balancing representation
            # outputed by the encoder.
            hyperparams['rnn_hidden_units'] = encoder_best_hyperparams['br_size']

            hyperparams['br_size'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['fc_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * (hyperparams['br_size']))
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001, 0.0001])
            hyperparams['batch_size'] = np.random.choice([256, 512, 1024])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = CRN_Model(params, hyperparams, b_train_decoder=True)
            model.train(dataset_train, dataset_val, model_name, model_dir)
            validation_mse, _ = model.evaluate_predictions(dataset_val)

            if (validation_mse < best_validation_mse):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_mse, validation_mse))
                best_validation_mse = validation_mse
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(decoder_hyperparams_file, best_hyperparams)

    else:
        # The rnn_hidden_units needs to be the same as the encoder br_size.
        logging.info("Using default hyperparameters")
        best_hyperparams = {
            'br_size': 18,
            'rnn_keep_prob': 0.9,
            'fc_hidden_units': 36,
            'batch_size': 1024,
            'learning_rate': 0.001,
            'rnn_hidden_units': encoder_best_hyperparams['br_size']}

        write_results_to_file(decoder_hyperparams_file, best_hyperparams)

    model = CRN_Model(params, best_hyperparams, b_train_decoder=True)
    model.train(dataset_train, dataset_val, model_name, model_dir)


def process_seq_data(data_map, states, projection_horizon):
    """
    Split the sequences in the training data to train the decoder.
    """

    outputs = data_map['outputs']
    sequence_lengths = data_map['sequence_lengths']
    active_entries = data_map['active_entries']
    current_treatments = data_map['current_treatments']
    previous_treatments = data_map['previous_treatments']
    current_covariates = data_map['current_covariates']

    num_patients, num_time_steps, num_features = outputs.shape

    num_seq2seq_rows = num_patients * num_time_steps

    seq2seq_state_inits = np.zeros((num_seq2seq_rows, states.shape[-1]))
    seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
    seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
    seq2seq_current_covariates = np.zeros((num_seq2seq_rows, projection_horizon, current_covariates.shape[-1]))
    seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
    seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
    seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)

    total_seq2seq_rows = 0  # we use this to shorten any trajectories later

    for i in range(num_patients):

        sequence_length = int(sequence_lengths[i])

        for t in range(1, sequence_length):  # shift outputs back by 1
            seq2seq_state_inits[total_seq2seq_rows, :] = states[i, t - 1, :]  # previous state output

            max_projection = min(projection_horizon, sequence_length - t)

            seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[i, t:t + max_projection, :]
            seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = previous_treatments[i,
                                                                                  t - 1:t + max_projection - 1, :]
            seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = current_treatments[i,
                                                                                 t:t + max_projection, :]
            seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t:t + max_projection, :]
            seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
            seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :] = current_covariates[i,
                                                                                 t:t + max_projection, :]

            total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
    seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
    seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
    seq2seq_current_covariates = seq2seq_current_covariates[:total_seq2seq_rows, :, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    # Package outputs
    seq2seq_data_map = {
        'init_state': seq2seq_state_inits,
        'previous_treatments': seq2seq_previous_treatments,
        'current_treatments': seq2seq_current_treatments,
        'current_covariates': seq2seq_current_covariates,
        'outputs': seq2seq_outputs,
        'sequence_lengths': seq2seq_sequence_lengths,
        'active_entries': seq2seq_active_entries,
        'unscaled_outputs': seq2seq_outputs * data_map['output_stds'] + data_map['output_means'],
        'output_means': data_map['output_means'],
        'output_stds': data_map['output_stds'],
    }

    return seq2seq_data_map


def process_counterfactual_seq_test_data(test_data, data_map, states, projection_horizon):
    sequence_lengths = test_data['sequence_lengths']

    outputs = data_map['outputs']
    current_treatments = data_map['current_treatments']
    previous_treatments = data_map['previous_treatments']
    current_covariates = data_map['current_covariates']

    num_patient_points = outputs.shape[0]
    sequence_lengths = sequence_lengths - 1

    seq2seq_state_inits = np.zeros((num_patient_points, states.shape[-1]))
    seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
    seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
    seq2seq_current_covariates = np.zeros((num_patient_points, projection_horizon, current_covariates.shape[-1]))
    seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
    seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
    seq2seq_sequence_lengths = np.zeros(num_patient_points)

    for i in range(num_patient_points):
        seq_length = int(sequence_lengths[i])
        seq2seq_state_inits[i] = states[i, seq_length - 1]
        seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
        seq2seq_previous_treatments[i] = previous_treatments[i, seq_length - 1:seq_length + projection_horizon - 1, :]
        seq2seq_current_treatments[i] = current_treatments[i, seq_length:seq_length + projection_horizon, :]
        seq2seq_outputs[i] = outputs[i, seq_length: seq_length + projection_horizon, :]
        seq2seq_sequence_lengths[i] = projection_horizon
        seq2seq_current_covariates[i] = np.repeat([current_covariates[i, seq_length - 1]], projection_horizon, axis=0)

    # Package outputs
    seq2seq_data_map = {
        'init_state': seq2seq_state_inits,
        'previous_treatments': seq2seq_previous_treatments,
        'current_treatments': seq2seq_current_treatments,
        'current_covariates': seq2seq_current_covariates,
        'outputs': seq2seq_outputs,
        'sequence_lengths': seq2seq_sequence_lengths,
        'active_entries': seq2seq_active_entries,
        'unscaled_outputs': seq2seq_outputs * data_map['output_stds'] + data_map['output_means'],
        'output_means': data_map['output_means'],
        'output_stds': data_map['output_stds'],
        'patient_types': test_data['patient_types'],
        'patient_ids_all_trajectories': test_data['patient_ids_all_trajectories'],
        'patient_current_t': test_data['patient_current_t']
    }

    return seq2seq_data_map


def test_CRN_decoder(pickle_map, max_projection_horizon, projection_horizon, models_dir,
                     encoder_model_name, encoder_hyperparams_file,
                     decoder_model_name, decoder_hyperparams_file,
                     b_decoder_hyperparm_tuning):
    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    scaling_data = pickle_map['scaling_data']
    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)

    encoder_model = load_trained_model(validation_processed, encoder_hyperparams_file, encoder_model_name, models_dir)
    training_br_states = encoder_model.get_balancing_reps(training_processed)
    validation_br_states = encoder_model.get_balancing_reps(validation_processed)

    training_seq_processed = process_seq_data(training_processed, training_br_states, max_projection_horizon)
    validation_seq_processed = process_seq_data(validation_processed, validation_br_states, max_projection_horizon)

    fit_CRN_decoder(dataset_train=training_seq_processed, dataset_val=validation_seq_processed,
                    model_dir=models_dir,
                    model_name=decoder_model_name, encoder_hyperparams_file=encoder_hyperparams_file,
                    decoder_hyperparams_file=decoder_hyperparams_file, b_hyperparam_opt=b_decoder_hyperparm_tuning)

    test_data_seq_actions = pickle_map['test_data_seq']
    test_processed = get_processed_data(pickle_map['test_data_seq'], scaling_data)
    encoder_model = load_trained_model(test_processed, encoder_hyperparams_file, encoder_model_name,
                                       models_dir)
    test_br_states = encoder_model.get_balancing_reps(test_processed)
    test_br_outputs = encoder_model.get_predictions(test_processed)

    test_seq_processed = process_counterfactual_seq_test_data(test_data_seq_actions, test_processed, test_br_states,
                                                              projection_horizon)
    CRN_deocoder = load_trained_model(test_seq_processed, decoder_hyperparams_file, decoder_model_name, models_dir,
                                      b_decoder_model=True)

    seq_predictions = CRN_deocoder.get_autoregressive_sequence_predictions(test_data_seq_actions, test_processed,
                                                                           test_br_states, test_br_outputs,
                                                                           projection_horizon)
    seq_predictions = seq_predictions * test_seq_processed['output_stds'] + test_seq_processed['output_means']

    # During the simulation some trajectories in the test set have nan values. These were removed when
    # computing the test metric. This only happens for the test set where we generate counterfactuals under different
    # treatment plans.
    nan_idx = np.unique(np.where(np.isnan(test_seq_processed['unscaled_outputs']))[0])
    not_nan = np.array([i for i in range(seq_predictions.shape[0]) if i not in nan_idx])
    mse = get_mse_at_follow_up_time(seq_predictions[not_nan], test_seq_processed['unscaled_outputs'][not_nan],
                                    test_seq_processed['active_entries'][not_nan])

    rmse = np.sqrt(mse[projection_horizon - 1]) / 1150 * 100  # Max tumour volume = 1150
    return rmse
