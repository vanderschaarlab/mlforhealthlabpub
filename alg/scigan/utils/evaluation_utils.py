# Copyright (c) 2020, Ioana Bica

import numpy as np
from scipy.integrate import romb
import tensorflow as tf

from data_simulation import get_patient_outcome
from scipy.optimize import minimize


def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples


def get_model_predictions(sess, num_treatments, num_dosage_samples, test_data):
    batch_size = test_data['x'].shape[0]

    treatment_dosage_samples = sample_dosages(batch_size, num_treatments, num_dosage_samples)
    factual_dosage_position = np.random.randint(num_dosage_samples, size=[batch_size])
    treatment_dosage_samples[range(batch_size), test_data['t'], factual_dosage_position] = test_data['d']

    treatment_dosage_mask = np.zeros(shape=[batch_size, num_treatments, num_dosage_samples])
    treatment_dosage_mask[range(batch_size), test_data['t'], factual_dosage_position] = 1

    I_logits = sess.run('inference_outcomes:0',
                        feed_dict={'input_features:0': test_data['x'],
                                   'input_treatment_dosage_samples:0': treatment_dosage_samples})

    Y_pred = np.sum(treatment_dosage_mask * I_logits, axis=(1, 2))

    return Y_pred


def get_true_dose_response_curve(news_dataset, patient, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_patient_outcome(patient, news_dataset['metadata']['v'], treatment_idx, dosage)
        return y

    return true_dose_response_curve


def compute_eval_metrics(dataset, test_patients, num_treatments, num_dosage_samples, model_folder):
    mises = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []
    pred_vals = []
    true_best = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], model_folder)

        for patient in test_patients:
            for treatment_idx in range(num_treatments):
                test_data = dict()
                test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                test_data['d'] = treatment_strengths

                pred_dose_response = get_model_predictions(sess=sess, num_treatments=num_treatments,
                                                           num_dosage_samples=num_dosage_samples, test_data=test_data)
                pred_dose_response = pred_dose_response * (
                        dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                                     dataset['metadata']['y_min']

                true_outcomes = [get_patient_outcome(patient, dataset['metadata']['v'], treatment_idx, d) for d in
                                 treatment_strengths]

                mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
                mises.append(mise)

                best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

                def pred_dose_response_curve(dosage):
                    test_data = dict()
                    test_data['x'] = np.expand_dims(patient, axis=0)
                    test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                    test_data['d'] = np.expand_dims(dosage, axis=0)

                    ret_val = get_model_predictions(sess=sess, num_treatments=num_treatments,
                                                    num_dosage_samples=num_dosage_samples,
                                                    test_data=test_data)
                    ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                              dataset['metadata']['y_min']
                    return ret_val

                true_dose_response_curve = get_true_dose_response_curve(dataset, patient, treatment_idx)

                min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                        x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

                max_pred_opt_y = - min_pred_opt.fun
                max_pred_dosage = min_pred_opt.x
                max_pred_y = true_dose_response_curve(max_pred_dosage)

                min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                        x0=[0.5], method="SLSQP", bounds=[(0, 1)])
                max_true_y = - min_true_opt.fun
                max_true_dosage = min_true_opt.x

                dosage_policy_error = (max_true_y - max_pred_y) ** 2
                dosage_policy_errors.append(dosage_policy_error)

                pred_best.append(max_pred_opt_y)
                pred_vals.append(max_pred_y)
                true_best.append(max_true_y)

            selected_t_pred = np.argmax(pred_vals[-num_treatments:])
            selected_val = pred_best[-num_treatments:][selected_t_pred]
            selected_t_optimal = np.argmax(true_best[-num_treatments:])
            optimal_val = true_best[-num_treatments:][selected_t_optimal]
            policy_error = (optimal_val - selected_val) ** 2
            policy_errors.append(policy_error)

    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors))
