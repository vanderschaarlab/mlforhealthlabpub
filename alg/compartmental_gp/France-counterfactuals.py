#!/usr/bin/env python
# coding: utf-8


import pickle
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
from pyro.ops.stats import quantile

import data_loader
import pyro_model.helper
import pyro_model.helper
from pyro_model.counterfactual_helper import get_R0_sooner_lockdown, get_R0_later_lockdown, get_counterfactual

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

# prefix = ''
prefix = 'trained_models/'
pad = 24

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
data_dict = pyro_model.helper.smooth_daily(data_dict)

days = 0
train_len = data_dict['cum_death'].shape[0] - days
covariates_actual = pyro_model.helper.get_covariates_intervention(data_dict, train_len, notime=True)
Y_train = pyro_model.helper.get_Y(data_dict, train_len)

err_country_list = []
seed_list = []

for seed in range(25):
    model_id = 'day-{}-rng-{}'.format(days, seed)

    try:
        with open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
            res = pickle.load(f)
    except Exception:
        continue

    with open(prefix + 'Loop{}/{}-forecaster.csv'.format(days, model_id), 'rb') as f:
        forecaster = pickle.load(f)

    prediction = quantile(res['prediction'].squeeze(), (0.5,), dim=0).squeeze()
    err = np.diff(prediction, axis=0) - data_dict['daily_death'][1:, ].numpy()
    err_country = np.sum(np.abs(err), axis=0)
    err_country_list.append(err_country)
    seed_list.append(seed)

err = np.stack(err_country_list, axis=0)

best_model = np.argmin(err, axis=0)
best_err = np.min(err, axis=0)
best_seed = [seed_list[x] for x in best_model]

df = pds.DataFrame({
    'country': countries,
    'best_seed': best_seed,
    'best_err': best_err
})

seed = 5
c = 5

model_id = 'day-{}-rng-{}'.format(days, seed)

with open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
    res = pickle.load(f)

with open(prefix + 'Loop{}/{}-forecaster.csv'.format(days, model_id), 'rb') as f:
    forecaster = pickle.load(f)

prediction = quantile(res['prediction'].squeeze(), (0.5,), dim=0).squeeze()

dt_list = data_dict['date_list']
start_date = data_dict['t_init'][c] + pad

R0 = res['R0'].squeeze()

R0_counter7 = get_R0_sooner_lockdown(R0, 7)
R0_counter14 = get_R0_sooner_lockdown(R0, 14)

R0_counter7a = get_R0_later_lockdown(R0, 7)
R0_counter14a = get_R0_later_lockdown(R0, 14)

pred_counter7 = get_counterfactual(data_dict, forecaster, res, R0_counter7)
pred_counter7_lo = pred_counter7[0]
pred_counter7_up = pred_counter7[2]
pred_counter7_me = pred_counter7[1]

pred_counter7a = get_counterfactual(data_dict, forecaster, res, R0_counter7a)
pred_counter7a_lo = pred_counter7a[0]
pred_counter7a_up = pred_counter7a[2]
pred_counter7a_me = pred_counter7a[1]

pred_true = get_counterfactual(data_dict, forecaster, res, R0)[1, ...]

start_ind = 20

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.fill_between(dt_list[start_ind + 1:], pred_counter7_lo[start_ind:, c], pred_counter7_up[start_ind:, c],
                 color='blue', alpha='0.3')

plt.fill_between(dt_list[start_ind + 1:], pred_counter7a_lo[start_ind:, c], pred_counter7a_up[start_ind:, c],
                 color='orange', alpha='0.3')

plt.plot(dt_list[start_ind + 1:], pred_counter7_me[start_ind:, c], label='One week earlier', color='navy')
plt.plot(dt_list[start_ind + 1:], pred_true[start_ind:, c], label='Actual plan', color='black')
plt.plot(dt_list[start_ind + 1:], pred_counter7a_me[start_ind:, c], label='One week later', color='red')

plt.plot(datetime(2020, 3, 11), 0, marker='x', markersize=5, color="black", label='Lockdown start dates',
         linestyle="None")
plt.plot([datetime(2020, 3, 4)], [0], marker='x', markersize=5, color="navy")
plt.plot([datetime(2020, 3, 18)], [0], marker='x', markersize=5, color="red")

plt.plot(dt_list[start_ind:], data_dict['actual_daily_death'][:, c][start_ind:], '.', label='acutal')
plt.legend()

plt.savefig('tables/Fig-C7-FR-counterfactual.png', dpi=300)

df_counterfactual = pds.DataFrame({
    'country': countries[c],
    'dt': dt_list[start_ind + 1:],
    'actual_deaths': data_dict['actual_daily_death'][:, c][start_ind + 1:],
    'actual_plan': pred_true[start_ind:, c],
    'wk_early_mean': pred_counter7_me[start_ind:, c],
    'wk_early_lower': pred_counter7_lo[start_ind:, c],
    'wk_early_upper': pred_counter7_up[start_ind:, c],
    'wk_later_mean': pred_counter7a_me[start_ind:, c],
    'wk_later_lower': pred_counter7a_lo[start_ind:, c],
    'wk_later_upper': pred_counter7a_up[start_ind:, c],
})

df_counterfactual.to_csv('tables/Fig-C7-FR-counterfactual.csv')
