"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs
from rmsn.configs import load_optimal_parameters

import rmsn.core_routines as core
from rmsn.core_routines import test

import numpy as np
import logging
import os

import tensorflow as tf
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
expt_name = "treatment_effects"


def propensity_generation(dataset_map, MODEL_ROOT, b_use_predicted_confounders,
                          b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    action_inputs_only = load_optimal_parameters('treatment_rnn_action_inputs_only',
                                                         expt_name, MODEL_ROOT,
                                                         add_net_name=True)
    action_w_trajectory_inputs = load_optimal_parameters('treatment_rnn',
                                                                 expt_name, MODEL_ROOT,
                                                                 add_net_name=True)

    # Generate propensity weights for validation data as well - used for MSM which is calibrated on train + valid data
    b_with_validation = False
    # Generate non-stabilised IPTWs (default false)
    b_denominator_only = False

    # Setup tensorflow - setup session to use cpu/gpu
    tf_device = 'cpu'
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        tf_config= tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    # Config + activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'rnn_model_bptt': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid'),
                      'treatment_rnn_softmax': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only_softmax': ("tanh", 'sigmoid'),
                      }

    configs = {'action_num': action_inputs_only,
               'action_den': action_w_trajectory_inputs}

    # Setup the simulated datasets
    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    # Generate propensity weights for validation data if required
    if b_with_validation:
        for k in training_data:
            training_data[k] = np.concatenate([training_data[k], validation_data[k]])

    ##############################################################################################################
    # Functions
    def get_predictions(config):

        net_name = config[0]

        hidden_activation, output_activation = activation_map[net_name]

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name

        # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, b_predict_actions, b_use_actions_only,
                                                     b_use_predicted_confounders, b_use_oracle_confounders, b_remove_x1)
        validation_processed = core.get_processed_data(validation_data, b_predict_actions,
                                                       b_use_actions_only,
                                                       b_use_predicted_confounders, b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

        # Unpack remaining variables
        dropout_rate = config[1]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]



        model_folder = os.path.join(MODEL_ROOT, net_name)
        means, outputs, _, _ = test(training_processed, validation_processed, training_processed, tf_config,
                                     net_name, expt_name, dropout_rate, num_features, num_outputs,
                                     memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                                     hidden_activation, output_activation, model_folder)

        return means, outputs

    def get_weights(probs, targets):
        w = probs*targets + (1-probs) * (1-targets)
        return w.prod(axis=2)


    def get_weights_from_config(config):
        net_name = config[0]

        probs, targets = get_predictions(config)

        return get_weights(probs, targets)

    def get_probabilities_from_config(config):
        net_name = config[0]

        probs, targets = get_predictions(config)

        return probs


    ##############################################################################################################

    # Action with trajs
    weights = {k: get_weights_from_config(configs[k]) for k in configs}

    den = weights['action_den']
    num = weights['action_num']

    propensity_weights = 1.0/den if b_denominator_only else num/den

    # truncation @ 95th and 5th percentiles
    UB = np.percentile(propensity_weights, 99)
    LB = np.percentile(propensity_weights, 1)

    propensity_weights[propensity_weights > UB] = UB
    propensity_weights[propensity_weights < LB] = LB

    # Adjust so for 3 trajectories here
    horizon = 1
    (num_patients, num_time_steps) = propensity_weights.shape
    output = np.ones((num_patients, num_time_steps, horizon))

    tmp = np.ones((num_patients, num_time_steps))
    tmp[:, 1:] = propensity_weights[:, :-1]
    propensity_weights = tmp

    for i in range(horizon):
        output[:, :num_time_steps-i, i] = propensity_weights[:, i:]

    propensity_weights = output.cumprod(axis=2)

    suffix = "" if not b_denominator_only else "_den_only"

    if b_with_validation:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores_w_validation{}".format(suffix))
    else:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores{}".format(suffix))

    np.save(save_file, propensity_weights)


