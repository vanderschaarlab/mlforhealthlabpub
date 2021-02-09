#!/usr/bin/env python
# coding: utf-8


from datetime import timedelta

import numpy as np
import pandas as pds
import torch
from scipy.stats import norm

import data_loader


# ## loading truth data


def smooth_curve_1d(x):
    w = np.ones(7, 'd')
    y = np.convolve(w / w.sum(), x, mode='valid')
    y = np.concatenate([np.zeros(3), y])
    return y


def smooth_daily(data_dict):
    daily = data_dict['daily_death']

    dy_list = list()
    for i in range(daily.size(1)):
        ds = daily[:, i]
        dy = smooth_curve_1d(ds)
        dy_list.append(dy)

    sy = np.stack(dy_list, axis=-1)
    cum_y = np.cumsum(sy, axis=0)
    new_len = cum_y.shape[0]

    return {
        'cum_death': torch.tensor(cum_y)[:new_len, :],
        'daily_death': torch.tensor(sy)[:new_len, :],
        'actual_daily_death': data_dict['daily_death'][:new_len, :],
        'actual_cum_death': data_dict['cum_death'][:new_len, :],
        's_index': data_dict['s_index'][:new_len, :],
        'i_index': data_dict['i_index'][:new_len, :],
        'population': data_dict['population'],
        't_init': data_dict['t_init'],
        'date_list': data_dict['date_list'][:new_len],
        'countries': data_dict['countries'],
        'country_feat': data_dict['country_feat']
    }


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

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad, rebuttal=True)

data_dict = smooth_daily(data_dict)

test_start = [str(x.date()) for x in list(data_dict['date_list'])].index('2020-04-25')

test_len = 30

actual_14 = data_dict['actual_daily_death'][test_start:test_start + test_len].numpy()

eval_days = [str(data_dict['date_list'][test_start].date() + timedelta(days=x)) for x in range(test_len)]

# ## loading imperial data


df = pds.read_csv('covid19hub/imperial-results.csv')

df_test = df[df.time.isin(eval_days)]
df_test = df_test[
    ['time', 'country', 'estimated_deaths_mean', 'estimated_deaths_lower_CI_95', 'estimated_deaths_higher_CI_95']]

df_test.time.unique()

# ## loading IHME data


q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
     0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
     0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

country_list = [
    'Denmark',
    'Italy',
    'Germany',
    'Spain',
    'United Kingdom',
    'France',
    'Belgium',
    'Austria',
    'Sweden',
    'Switzerland',
    'Portugal',
    'Netherlands',
    'Brazil'
]
country_list.sort()

dat_ihme = pds.read_csv('ihme_benchmarks/2020_04_26/Hospitalization_all_locs.csv')

dat_ihme = dat_ihme[(dat_ihme['location_name'].isin(country_list)) & (dat_ihme.date.isin(eval_days))]
dat_ihme = dat_ihme[['location_name', 'date', 'deaths_mean', 'deaths_lower', 'deaths_upper']]
df_test = dat_ihme

df_test.date.unique()

ihme_country = list(dat_ihme['location_name'].unique())

err_list = []
country_list = []
length_list = []
crps_list = []

for country in ihme_country:
    df_slice = df_test[df_test.location_name == country]

    try:
        c_ind = countries.index(country)
    except ValueError:
        print(country)
        continue

    length = df_slice['deaths_upper'] - df_slice['deaths_lower']
    sd = length.values / (1.96 * 2)
    mean = df_slice['deaths_mean'].values
    sd[sd == 0] = np.mean(sd)
    length = np.mean(length.values)

    if country in ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Switzerland']:
        err = np.nan
        length = np.nan
        crps_list.append(np.nan)
    else:
        actual = actual_14[:, c_ind]
        err = np.mean(np.abs(actual_14[:, c_ind] - df_slice['deaths_mean'].values))

        crps_list0 = list()

        for k in range(len(sd)):
            pred = norm.ppf(q, loc=mean[k], scale=sd[k])
            proba = q.copy()

            less_ind = pred < actual[k]
            proba_label = np.ones_like(proba)
            proba_label[less_ind] = 0
            crps_list0.append(np.mean((proba_label - proba) ** 2))

        crps_list.append(np.mean(np.array(crps_list0)))

    err_list.append(err)
    country_list.append(country)
    length_list.append(length)

df_ihme = pds.DataFrame({'country': country_list, 'err': err_list, 'length': length_list, 'crps': crps_list})

df_ihme.to_csv('tables/benchmark-ihme-countries-30d.csv')

# ## loading YYG data


err_list = []
length_list = []
crps_list = []

for c in country_list:
    dat_yyg = pds.read_csv('global-04-25/{}_ALL.csv'.format(c.replace(' ', '-')))
    dat_yyg = dat_yyg[dat_yyg.date.isin(eval_days)]
    pred = dat_yyg.predicted_deaths_mean.values

    length = dat_yyg.predicted_deaths_upper - dat_yyg.predicted_deaths_lower
    sd = length.values / (1.96 * 2)
    sd[np.isnan(sd)] = np.nanmean(sd)

    length = np.mean(length)
    mean = pred

    c_ind = countries.index(c)
    actual = actual_14[:, c_ind]

    err = np.nanmean(np.abs(actual_14[:, c_ind] - pred))

    crps_list0 = list()

    for k in range(len(sd)):
        pred = norm.ppf(q, loc=mean[k], scale=sd[k])
        proba = q.copy()

        less_ind = pred < actual[k]
        proba_label = np.ones_like(proba)
        proba_label[less_ind] = 0
        crps_list0.append(np.mean((proba_label - proba) ** 2))

    crps_list.append(np.mean(np.array(crps_list0)))

    err_list.append(err)
    length_list.append(length)

df_yyg = pds.DataFrame({'country': country_list, 'err': err_list, 'length': length_list, 'crps': crps_list})

df_yyg.to_csv('tables/benchmark-yyg-countries-30d.csv')
