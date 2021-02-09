"""
Creating Table C3: columns for CGP (Local)
"""
# !/usr/bin/env python
# coding: utf-8


# Creating Table C3: columns for CGP (Local)


import pickle

import numpy as np
import pandas as pds
import torch
from pyro.ops.stats import quantile

import data_loader
import pyro_model.helper

# ## loading data


countries_list = [
    'United Kingdom',
    'Italy',
    'Germany',
    'Spain',
    'France',
    'Korea, South',
    'Brazil',
    'Iran',
    'Netherlands',
    'Sweden',
    'Mexico',
    'India',
    'Russia',
    'Japan',
    'South Africa',
    'Egypt',
    'Norway'
]
# prefix = ''
prefix = 'trained_models/'

# for country in countries_list:
country = countries_list[0]
countries = [country]

pad = 24

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
data_dict = pyro_model.helper.smooth_daily(data_dict)

days = 14
train_len = data_dict['cum_death'].shape[0] - days
test_dates = data_dict['date_list'][train_len:]

print('Done')

err14_list = []

err7_list = []
for country in countries_list:
    print(country)
    countries = [country]

    pad = 24

    data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
    data_dict = pyro_model.helper.smooth_daily(data_dict)

    days = 14
    train_len = data_dict['cum_death'].shape[0] - days
    test_dates = data_dict['date_list'][train_len:]

    predictive_list = []
    samples_list = []

    for seed in range(10):
        model_id = country + '-ablation-day-{}-rng-{}'.format(days, seed)
        try:
            with open(prefix + 'AblationLoop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
                predictive = pickle.load(f)
        except Exception:
            continue
        predictive_list.append(predictive)

        with open(prefix + 'AblationLoop{}/{}-samples.pkl'.format(days, model_id), 'rb') as f:
            samples = pickle.load(f)
        samples_list.append(samples)

    val_window = 14

    seir_error_list = []

    for i in range(len(predictive_list)):
        seir_train = quantile(predictive_list[i]['prediction'].squeeze(), 0.5, dim=0)[-val_window + 1:].numpy()
        seir_train = np.diff(seir_train, axis=0)
        seir_label = data_dict['daily_death'][train_len - val_window:train_len, :].numpy()

        seir_error = np.abs(np.sum(seir_train, axis=0) - np.sum(seir_label, axis=0))
        seir_error_list.append(seir_error)

    seir_error = np.stack(seir_error_list, axis=0)
    best_model = np.argmin(seir_error, axis=0)

    test_len = 14
    best_error_list = []

    test_len = test_len - 1
    for j, i in zip(range(len(countries)), best_model):
        c = countries[j]
        samples = samples_list[i]
        p50 = quantile(torch.tensor(samples), 0.5, dim=0)[1:, :]
        pred = p50[test_len, :]
        truth = data_dict['actual_cum_death'][test_len - days]
        err = (pred[j] - truth[j]).item()

        best_error_list.append(err)

    err_14 = best_error_list[0]

    test_len = 7
    best_error_list = []

    test_len = test_len - 1
    for j, i in zip(range(len(countries)), best_model):
        c = countries[j]
        samples = samples_list[i]
        p50 = quantile(torch.tensor(samples), 0.5, dim=0)[1:, :]
        pred = p50[test_len, :]
        truth = data_dict['actual_cum_death'][test_len - days]
        err = (pred[j] - truth[j]).item()

        best_error_list.append(err)

    err_7 = best_error_list[0]
    err14_list.append(err_14)
    err7_list.append(err_7)
    print(country, err_14, err_7)

abl_df = pds.DataFrame({'country': countries_list, 'err7': err7_list, 'err14': err14_list})

abl_df.to_csv('tables/Table-C3-ablation-many-countries-d7-d14.csv')
