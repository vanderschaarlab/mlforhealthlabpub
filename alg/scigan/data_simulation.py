# Copyright (c) 2020, Ioana Bica

from __future__ import print_function

import numpy as np
import pickle

from sklearn.model_selection import StratifiedShuffleSplit


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def compute_beta(alpha, optimal_dosage):
    if (optimal_dosage <= 0.001 or optimal_dosage >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(optimal_dosage) + (2.0 - alpha)

    return beta


def generate_patient(x, v, num_treatments, treatment_selection_bias, dosage_selection_bias,
                     scaling_parameter, noise_std):
    outcomes = []
    dosages = []

    for treatment in range(num_treatments):
        if (treatment == 0):
            b = 0.75 * np.dot(x, v[treatment][1]) / (np.dot(x, v[treatment][2]))

            if (b >= 0.75):
                optimal_dosage = b / 3.0
            else:
                optimal_dosage = 1.0

            alpha = dosage_selection_bias
            dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))

            y = get_patient_outcome(x, v, treatment, dosage, scaling_parameter)

        elif (treatment == 1):
            optimal_dosage = np.dot(x, v[treatment][2]) / (2.0 * np.dot(x, v[treatment][1]))
            alpha = dosage_selection_bias
            dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
            if (optimal_dosage <= 0.001):
                dosage = 1 - dosage

            y = get_patient_outcome(x, v, treatment, dosage, scaling_parameter)

        elif (treatment == 2):
            optimal_dosage = np.dot(x, v[treatment][1]) / (2.0 * np.dot(x, v[treatment][2]))
            alpha = dosage_selection_bias
            dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
            if (optimal_dosage <= 0.001):
                dosage = 1 - dosage

            y = get_patient_outcome(x, v, treatment, dosage, scaling_parameter)

        outcomes.append(y)
        dosages.append(dosage)

    treatment_coeff = [treatment_selection_bias * (outcomes[i] / np.max(outcomes)) for i in range(num_treatments)]
    treatment = np.random.choice(num_treatments, p=softmax(treatment_coeff))

    return treatment, dosages[treatment], outcomes[treatment] + np.random.normal(0, noise_std)


def get_patient_outcome(x, v, treatment, dosage, scaling_parameter=10):
    if (treatment == 0):
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * dosage * (dosage - 0.75 * (
                np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2]))) ** 2)
    elif (treatment == 1):
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + np.sin(
            np.pi * (np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2])) * dosage))
    elif (treatment == 2):
        y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * (np.dot(x, v[treatment][
            1]) * dosage - np.dot(x, v[treatment][2]) * dosage ** 2))

    return y


def get_dataset_splits(dataset):
    dataset_keys = ['x', 't', 'd', 'y', 'y_normalized']

    train_index = dataset['metadata']['train_index']
    val_index = dataset['metadata']['val_index']
    test_index = dataset['metadata']['test_index']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index]
        dataset_val[key] = dataset[key][val_index]
        dataset_test[key] = dataset[key][test_index]

    dataset_train['metadata'] = dataset['metadata']
    dataset_val['metadata'] = dataset['metadata']
    dataset_test['metadata'] = dataset['metadata']

    return dataset_train, dataset_val, dataset_test


def get_split_indices(num_patients, patients, treatments, validation_fraction, test_fraction):
    num_validation_patients = int(np.floor(num_patients * validation_fraction))
    num_test_patients = int(np.floor(num_patients * test_fraction))

    test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_patients, random_state=0)
    rest_indices, test_indices = next(test_sss.split(patients, treatments))

    val_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_patients, random_state=0)
    train_indices, val_indices = next(val_sss.split(patients[rest_indices], treatments[rest_indices]))

    return train_indices, val_indices, test_indices


class TCGA_Data():
    def __init__(self, args):
        np.random.seed(3)

        self.num_treatments = args['num_treatments']
        self.treatment_selection_bias = args['treatment_selection_bias']
        self.dosage_selection_bias = args['dosage_selection_bias']

        self.validation_fraction = args['validation_fraction']
        self.test_fraction = args['test_fraction']

        self.tcga_data = pickle.load(open('datasets/tcga.p', 'rb'))
        self.patients = self.normalize_data(self.tcga_data['rnaseq'])

        self.scaling_parameteter = 10
        self.noise_std = 0.2

        self.num_weights = 3
        self.v = np.zeros(shape=(self.num_treatments, self.num_weights, self.patients.shape[1]))

        for i in range(self.num_treatments):
            for j in range(self.num_weights):
                self.v[i][j] = np.random.uniform(0, 10, size=(self.patients.shape[1]))
                self.v[i][j] = self.v[i][j] / np.linalg.norm(self.v[i][j])

        self.dataset = self.generate_dataset(self.patients, self.num_treatments)

    def normalize_data(self, patient_features):
        x = (patient_features - np.min(patient_features, axis=0)) / (
                np.max(patient_features, axis=0) - np.min(patient_features, axis=0))

        for i in range(x.shape[0]):
            x[i] = x[i] / np.linalg.norm(x[i])

        return x

    def generate_dataset(self, patient_features, num_treatments):
        tcga_dataset = dict()
        tcga_dataset['x'] = []
        tcga_dataset['y'] = []
        tcga_dataset['t'] = []
        tcga_dataset['d'] = []
        tcga_dataset['metadata'] = dict()
        tcga_dataset['metadata']['v'] = self.v
        tcga_dataset['metadata']['treatment_selection_bias'] = self.treatment_selection_bias
        tcga_dataset['metadata']['dosage_selection_bias'] = self.dosage_selection_bias
        tcga_dataset['metadata']['noise_std'] = self.noise_std
        tcga_dataset['metadata']['scaling_parameter'] = self.scaling_parameteter

        for patient in patient_features:
            t, dosage, y = generate_patient(x=patient, v=self.v, num_treatments=num_treatments,
                                            treatment_selection_bias=self.treatment_selection_bias,
                                            dosage_selection_bias=self.dosage_selection_bias,
                                            scaling_parameter=self.scaling_parameteter,
                                            noise_std=self.noise_std)
            tcga_dataset['x'].append(patient)
            tcga_dataset['t'].append(t)
            tcga_dataset['d'].append(dosage)
            tcga_dataset['y'].append(y)

        for key in ['x', 't', 'd', 'y']:
            tcga_dataset[key] = np.array(tcga_dataset[key])

        tcga_dataset['metadata']['y_min'] = np.min(tcga_dataset['y'])
        tcga_dataset['metadata']['y_max'] = np.max(tcga_dataset['y'])

        tcga_dataset['y_normalized'] = (tcga_dataset['y'] - np.min(tcga_dataset['y'])) / (
                np.max(tcga_dataset['y']) - np.min(tcga_dataset['y']))

        train_indices, validation_indices, test_indices = get_split_indices(num_patients=tcga_dataset['x'].shape[0],
                                                                            patients=tcga_dataset['x'],
                                                                            treatments=tcga_dataset['t'],
                                                                            validation_fraction=self.validation_fraction,
                                                                            test_fraction=self.test_fraction)

        tcga_dataset['metadata']['train_index'] = train_indices
        tcga_dataset['metadata']['val_index'] = validation_indices
        tcga_dataset['metadata']['test_index'] = test_indices

        return tcga_dataset
