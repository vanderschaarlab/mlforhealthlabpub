#!/usr/bin/env python
# coding: utf-8


import pickle

import numpy as np
import pandas as pds
import torch
from pyro.ops.stats import quantile

import data_loader
import pyro_model.helper


prefix = 'trained_models/'
# prefix = ''

df_list = []

for days in [14, 28, 42]:

    # ## loading data

    countries = [
        'United Kingdom',
        'Italy',
        'Germany',
        'Spain',
        'US',
        'France',
        'Belgium',
        'Korea, South',
        'Brazil',
        'Iran',
        'Netherlands',
        'Canada',
        'Turkey',
        'Romania',
        'Portugal',
        'Sweden',
        'Switzerland',
        'Ireland',
        'Hungary',
        'Denmark',
        'Austria',
        'Mexico',
        'India',
        'Ecuador',
        'Russia',
        'Peru',
        'Indonesia',
        'Poland',
        'Philippines',
        'Japan',
        'Pakistan'
    ]

    pad = 24

    data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
    data_dict = pyro_model.helper.smooth_daily(data_dict)

    train_len = data_dict['cum_death'].shape[0] - days

    test_dates = data_dict['date_list'][train_len:]

    # ## loading results

    rmse_list = []
    seed_list = []
    predictive_list = []
    samples_list = []

    for seed in range(15):
        model_id = 'day-{}-rng-{}'.format(days, seed)
        try:
            df_rmse = pds.read_csv(prefix + 'Loop{}/{}-rmse.csv'.format(days, model_id))
        except Exception:
            continue
        rmse_list.append(df_rmse)

        with open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
            predictive = pickle.load(f)
        predictive_list.append(predictive)

        with open(prefix + 'Loop{}/{}-samples.pkl'.format(days, model_id), 'rb') as f:
            samples = pickle.load(f)
        samples_list.append(samples)
        seed_list.append(seed)

    print(len(rmse_list))

    # validation accuracy
    val_window = 14

    seir_error_list = []

    for i in range(len(predictive_list)):
        seir_train = quantile(predictive_list[i]['prediction'].squeeze(), 0.5, dim=0)[-val_window + 1:, :].numpy()
        seir_train = np.diff(seir_train, axis=0)
        seir_label = data_dict['daily_death'][train_len - val_window:train_len, :].numpy()

        seir_error = np.abs(np.sum(seir_train, axis=0) - np.sum(seir_label, axis=0))
        seir_error_list.append(seir_error)


    seir_error = np.stack(seir_error_list, axis=0)
    best_model = np.argmin(seir_error, axis=0)
    best_model[0] = 6

    best_seed = [seed_list[x] for x in best_model]

    for test_len in [7, 14]:

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

        test_date = data_dict['date_list'][test_len - days].date()

        train_end_date = data_dict['date_list'][train_len].date()

        df_ours = pds.DataFrame(
            {'countries': countries, 'best_err': best_error_list, 'best_seed': best_seed, 'best_model': best_model})

        df_save = df_ours[df_ours.countries == 'US'].copy()
        df_save['window'] = test_len + 1
        if days == 14:
            df_save['fcst_date'] = 'apr25'
        elif days == 28:
            df_save['fcst_date'] = 'apr11'
        else:
            df_save['fcst_date'] = 'mar28'

        df_list.append(df_save)


df_save = pds.concat(df_list)
df_save.to_csv('tables/Table-1-cgp-us.csv')
