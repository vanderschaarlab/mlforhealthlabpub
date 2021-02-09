#!/usr/bin/env python
# coding: utf-8


from datetime import timedelta

import numpy as np
import pandas as pds
from scipy.stats import norm

import data_loader
import pyro_model.helper

# ## loading truth data


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

test_len = 14

actual_14 = data_dict['actual_daily_death'][-test_len:].numpy()

eval_days = [str(data_dict['date_list'][-test_len].date() + timedelta(days=x)) for x in range(test_len)]

# ## loading imperial data


df = pds.read_csv('covid19hub/imperial-results.csv')

df_test = df[df.time.isin(eval_days)]
df_test = df_test[
    ['time', 'country', 'estimated_deaths_mean', 'estimated_deaths_lower_CI_95', 'estimated_deaths_higher_CI_95']]

imp_country = list(df_test['country'].unique())

q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
     0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
     0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

err_list = []
country_list = []
length_list = []
crps_list = []

for country in imp_country:
    df_slice = df_test[df_test.country == country]

    if country == 'United_Kingdom':
        country = 'United Kingdom'
    try:
        c_ind = countries.index(country)
    except ValueError:
        print(country)
        continue

    length = df_slice['estimated_deaths_higher_CI_95'] - df_slice['estimated_deaths_lower_CI_95']
    sd = length.values / (1.96 * 2)
    mean = df_slice['estimated_deaths_mean'].values

    length = np.mean(length.values)

    if country != 'Spain':
        actual = actual_14[:-1, c_ind]
        err = np.mean(np.abs(actual_14[:-1, c_ind] - df_slice['estimated_deaths_mean'].values))
    else:
        actual = actual_14[:-2, c_ind]
        err = np.mean(np.abs(actual_14[:-2, c_ind] - df_slice['estimated_deaths_mean'].values))

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


df_imperial = pds.DataFrame({'country': country_list, 'err': err_list, 'length': length_list, 'crps': crps_list})

df_imperial.to_csv('tables/benchmark-imperial-countries-14d.csv')

# ## loading IHME data


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

    length = np.mean(length.values)
    sd[sd == 0] = np.mean(sd)

    if country in ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Switzerland']:
        actual = actual_14[:-3, c_ind]
        err = np.mean(np.abs(actual_14[:-3, c_ind] - df_slice['deaths_mean'].values))
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

df_ihme.to_csv('tables/benchmark-ihme-countries-14d.csv')

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
    pred = dat_yyg.predicted_deaths_mean.values
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

df_yyg.to_csv('tables/benchmark-yyg-countries-14d.csv')
