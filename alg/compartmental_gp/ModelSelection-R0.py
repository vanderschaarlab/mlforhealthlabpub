#!/usr/bin/env python
# coding: utf-8


import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
from pyro.ops.stats import quantile
import scipy.stats

import data_loader
import pyro_model.helper

# ## loading data


countries = [
    'United Kingdom',
    'Italy',
    'Germany',
    'Spain',
    'US',
    'France',
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
    'Pakistan',
    'South Africa',
    'Egypt',
    'Norway'
]

prefix = 'trained_models/'
# prefix = ''

pad = 24

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad, legacy=False)
data_dict = pyro_model.helper.smooth_daily(data_dict)

eval_list = countries

days = 14
train_len = data_dict['cum_death'].shape[0] - days

# ## loading results


seed_list = []
predictive_list = []

for seed in range(25):
    model_id = 'all-countries-new-day-{}-rng-{}'.format(days, seed)

    try:
        with open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
            predictive = pickle.load(f)
    except Exception:
        continue
    predictive_list.append(predictive)

    seed_list.append(seed)

print(len(seed_list))

# validation accuracy
val_window = 28

seir_error_list = []

for i in range(len(predictive_list)):
    seir_train = quantile(predictive_list[i]['prediction'].squeeze(), 0.5, dim=0)[-val_window + 1:, :].numpy()
    seir_train = np.diff(seir_train, axis=0)
    seir_label = data_dict['daily_death'][train_len - val_window:train_len, :].numpy()

    seir_error = np.abs(np.sum(seir_train, axis=0) - np.sum(seir_label, axis=0))
    seir_error_list.append(seir_error)

seir_error = np.stack(seir_error_list, axis=0)
best_model = np.argmin(seir_error, axis=0)
best_seed = [seed_list[x] for x in best_model]

R0_list = []
s_index_list = []
R0_country = []
prediction_list = []
truth_list = []

for j, i in zip(range(len(countries)), best_model):
    c = countries[j]
    if c not in eval_list:
        continue
    t_init = data_dict['t_init'][j].squeeze()
    predictive = predictive_list[i]['R0']
    R0 = predictive.mean(axis=0).squeeze()[j, :].numpy()
    R0[:t_init] = np.nan
    R0_list.append(R0)
    R0_country.append(c)
    s_index = data_dict['s_index'][:train_len, j]
    s_index[:t_init] = np.nan
    s_index_list.append(s_index)
    predictions = predictive_list[i]['prediction'].mean(axis=0).squeeze()[:, j].numpy()
    prediction_list.append(predictions)
    truth = data_dict['cum_death'][:train_len, j]
    truth_list.append(truth)

R0 = np.stack(R0_list, axis=-1)
s_index = np.stack(s_index_list, axis=-1)
seir_predictions = np.stack(prediction_list, axis=-1)
truth = np.stack(truth_list, axis=-1)

df_r0 = pds.DataFrame(R0, columns=R0_country, index=data_dict['date_list'][:train_len])
df_s_index = pds.DataFrame(s_index, columns=R0_country, index=data_dict['date_list'][:train_len])

df_r0.head()

df_r0.to_csv('tables/Fig-C6-R0.csv')

c = 0
plt.plot(R0[:, c])
plt.plot(s_index[:, c])

c = 0
plt.plot(np.diff(seir_predictions[:, c]))
plt.plot(np.diff(truth[:, c]))

rho_list = []
r_init_list = []
for r, s in zip(R0_list, s_index_list):
    s = s.numpy()
    nonna = np.logical_and(~np.isnan(r), ~np.isnan(s))
    met = scipy.stats.pearsonr(s[nonna], r[nonna])
    r_init = np.mean(r[nonna][:7])
    rho_list.append(met[0])
    r_init_list.append(r_init)

df_c = pds.DataFrame({'country': countries, 'rho': rho_list, 'r_init': r_init_list})
df_c = df_c.sort_values('country')

df_c.head()

dat_feat = data_loader.get_country_feature(countries)
dat_feat = dat_feat.pivot('country', 'metric', 'value')

dat_feat.head()

row_list = []

for c in dat_feat.columns:
    nonna = ~np.isnan(dat_feat[c].values)
    met = scipy.stats.pearsonr(dat_feat[c][nonna], df_c['r_init'][nonna])

    row_list.append((c, met[0], met[1]))
dat_cor = pds.DataFrame(row_list, columns=['met', 'cor', 'p_value'])
dat_cor = dat_cor.sort_values('cor', ascending=False)
dat_cor_p = dat_cor[dat_cor['p_value'] < 0.05].iloc[:10, :]

dat_cor_p.to_csv('tables/Table-C5-features_R0_before_lockdown.csv')

# policy effect
row_list = []

for c in dat_feat.columns:
    nonna = ~np.isnan(dat_feat[c].values)
    met = scipy.stats.pearsonr(dat_feat[c][nonna], df_c['rho'][nonna])

    row_list.append((c, met[0], met[1]))
dat_cor = pds.DataFrame(row_list, columns=['met', 'cor', 'p_value'])
dat_cor = dat_cor.sort_values('cor', ascending=False)
dat_rho_p = dat_cor[dat_cor['p_value'] < 0.05]

dat_rho_p.to_csv('tables/Table-C4-features_effect_of_lockdown.csv')
