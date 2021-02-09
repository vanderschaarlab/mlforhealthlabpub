#!/usr/bin/env python
# coding: utf-8


import pickle

import numpy as np
import pandas as pds
from pyro.ops.stats import quantile
from scipy.stats import norm

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
prefix = 'trained_models/'
# prefix = ''

pad = 24

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
data_dict = pyro_model.helper.smooth_daily(data_dict)

days = 14
train_len = data_dict['cum_death'].shape[0] - days

test_dates = data_dict['date_list'][train_len:]

len(data_dict['date_list'][train_len:])

# ## loading results


seed_list = []
predictive_list = []
samples_list = []

for seed in range(15):
    model_id = 'day-{}-rng-{}'.format(days, seed)
    try:
        with open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb') as f:
            predictive = pickle.load(f)
    except Exception:
        continue
    predictive_list.append(predictive)

    with open(prefix + 'Loop{}/{}-samples.pkl'.format(days, model_id), 'rb') as f:
        samples = pickle.load(f)
    samples_list.append(samples)
    seed_list.append(seed)

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

best_seed = [seed_list[x] for x in best_model]

test_len = 14

best_error_list = []
pred_low_list = []
pred_high_list = []
covered_list = []
length_list = []
crps_list = []

test_len = test_len - 1
for j, i in zip(range(len(countries)), best_model):
    c = countries[j]

    # get daily death label
    seir_label = data_dict['daily_death'][train_len:, j].numpy()

    samples = samples_list[i]
    sample_daily = np.diff(samples, axis=1)

    model_pred = np.mean(sample_daily, axis=0)[:, j]
    err = np.mean(np.abs(model_pred - seir_label)[:test_len + 1])
    best_error_list.append(err)

    # percentiles
    sample_daily[sample_daily < 0] = 0
    model_pred_low = np.quantile(sample_daily, 0.025, axis=0)[:, j]
    model_pred_high = np.quantile(sample_daily, 0.975, axis=0)[:, j]
    covered = np.mean((seir_label >= model_pred_low)[:test_len + 1] & (seir_label <= model_pred_high)[:test_len + 1])
    length = np.mean((model_pred_high - model_pred_low)[:test_len + 1])

    # crps

    q = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
         0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
         0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    model_quantile = np.quantile(sample_daily, q, axis=0)[:, :, j]
    crps_list0 = list()
    for k in range(model_quantile.shape[1]):
        pred = model_quantile[:, k]
        proba = q.copy()

        less_ind = pred < seir_label[k]
        proba_label = np.ones_like(proba)
        proba_label[less_ind] = 0
        crps_list0.append(np.mean((proba_label - proba) ** 2))

    crps_list.append(np.mean(np.array(crps_list0)))

    pred_low_list.append(model_pred_low)
    pred_high_list.append(model_pred_high)
    covered_list.append(covered)
    length_list.append(length)


test_date = data_dict['date_list'][test_len - days].date()

train_end_date = data_dict['date_list'][train_len].date()

df_ours = pds.DataFrame(
    {'countries': countries, 'best_err': best_error_list, 'best_length': length_list, 'crps': crps_list,
     'best_seed': best_seed, 'best_model': best_model})

eval_list = [
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
    'Brazil',
    'US'
]

df_save = df_ours[df_ours.countries.isin(eval_list)]

df_save.to_csv('tables/Table-2-cgp-countries-14d.csv')

# ## Get benchmarks


from datetime import timedelta

test_len = 14

model_list = []
err_list = []
length_list = []
cprs_list = []

eval_days = [str(train_end_date + timedelta(days=x)) for x in range(test_len)]

pred_name = [str(x) + ' day ahead inc death' for x in range(1, test_len + 2)]

loc = countries.index('US')
seir_label = data_dict['daily_death'][train_len:, loc].numpy()[:test_len]

seir_label

# ### LANL


file_list = ['covid19hub/benchmark-04-25/' + '2020-04-23-LANL-GrowthRate.csv']
f = file_list[0]

m = 'LANL'
model_list.append(m)

df_bench = pds.read_csv(f)

df_test = df_bench[
    (df_bench.type == 'point') & (df_bench.target.isin(pred_name)) & (df_bench.location.isin(range(1, 57)))]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]

# predictions not available for 2020-05-03
if test_len == 14:
    seir_label2 = np.concatenate([seir_label[:8], seir_label[9:]])
else:
    seir_label2 = seir_label
err = np.mean(np.abs(df_test.value.values - seir_label2))
err_list.append(err)

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.025) & (
        df_bench.location.isin(range(1, 57)))]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_low = df_test[df_test.target_end_date.isin(eval_days)]

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.975) & (
        df_bench.location.isin(range(1, 57)))]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_high = df_test[df_test.target_end_date.isin(eval_days)]

length = np.mean(df_high.value.values - df_low.value.values)
length_list.append(length)

cprs_list.append(np.nan)

# ### Imperial


file_list = ['covid19hub/benchmark-04-25/' + '2020-04-26-Imperial-ensemble2.csv']
f = file_list[0]

m = 'Imperial'
model_list.append(m)

df_bench = pds.read_csv(f)

df_test = df_bench[(df_bench.type == 'point') & (df_bench.target.isin(pred_name))]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]

if test_len == 7:
    seir_label2 = seir_label[2:]
else:
    seir_label2 = seir_label[2:9]
err = np.mean(np.abs(df_test.value.values - seir_label2))
err_list.append(err)

df_test = df_bench[(df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.025)]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_low = df_test[df_test.target_end_date.isin(eval_days)]

df_test = df_bench[(df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.975)]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_high = df_test[df_test.target_end_date.isin(eval_days)]
length = np.mean(df_high.value.values - df_low.value.values)
length_list.append(length)

## crps
seir_label2 = seir_label[2:9]

df_test = df_bench[(df_bench.type == 'quantile') & (df_bench.target.isin(pred_name))]
df_test = df_test[['target_end_date', 'value', 'quantile']].groupby(['target_end_date', 'quantile']).sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]

df_test.head()

avail_days = df_test.target_end_date.unique()

day = avail_days[0]
i = 0
crps_list = []

for i, day in enumerate(avail_days):
    df_sub = df_test[df_test['target_end_date'] == day]

    pred = df_sub.value.values
    proba = df_sub['quantile'].values

    less_ind = pred < seir_label2[i]
    proba_label = np.ones_like(proba)
    proba_label[less_ind] = 0
    crps_list.append(np.mean((proba_label - proba) ** 2))

np.mean(np.array(crps_list))

cprs_list.append(np.mean(np.array(crps_list)))

# ### IHME


file_list = ['covid19hub/benchmark-04-25/' + '2020-04-27-IHME-CurveFit.csv']
f = file_list[0]

m = 'IHME'
model_list.append(m)

df_bench = pds.read_csv(f)

df_bench['quantile'].unique()

df_test = df_bench[(df_bench.type == 'point') & (df_bench.target.isin(pred_name)) & (df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]
mean = df_test.value.values
seir_label2 = seir_label[3:]
actual = seir_label2
err = np.mean(np.abs(df_test.value.values - seir_label2))
err_list.append(err)

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.025) & (
                df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_low = df_test[df_test.target_end_date.isin(eval_days)]

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.975) & (
                df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_high = df_test[df_test.target_end_date.isin(eval_days)]
sd = (df_high.value.values - df_low.value.values) / (1.96 * 2)
length = np.mean(df_high.value.values - df_low.value.values)
length_list.append(length)

crps_list0 = list()

for k in range(len(sd)):
    pred = norm.ppf(q, loc=mean[k], scale=sd[k])
    proba = q.copy()

    less_ind = pred < actual[k]
    proba_label = np.ones_like(proba)
    proba_label[less_ind] = 0
    crps_list0.append(np.mean((proba_label - proba) ** 2))

np.mean(np.array(crps_list0))

cprs_list.append(np.mean(np.array(crps_list0)))

# ### MIT-DELPHI


file_list = ['covid19hub/benchmark-04-25/' + '2020-04-27-MIT_CovidAnalytics-DELPHI.csv']
f = file_list[0]

m = 'MIT-DELPHI'
model_list.append(m)

df_bench = pds.read_csv(f)

cum_name = [str(x) + ' day ahead cum death' for x in range(1, test_len + 2)]

df_test = df_bench[(df_bench.type == 'point') & (df_bench.target.isin(cum_name))]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]

seir_label2 = seir_label[4:]
err = np.mean(np.abs(np.diff(df_test.value.values) - seir_label2))
err_list.append(err)

length_list.append(np.nan)
cprs_list.append(np.nan)

# ### YYG


file_list = ['covid19hub/benchmark-04-25/' + '2020-04-27-YYG-ParamSearch.csv']
f = file_list[0]

m = 'YYG'
model_list.append(m)

df_bench = pds.read_csv(f)

df_bench['quantile'].unique()

df_test = df_bench[(df_bench.type == 'point') & (df_bench.target.isin(pred_name)) & (df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_test = df_test[df_test.target_end_date.isin(eval_days)]

if test_len == 7:
    seir_label2 = seir_label[3:]
else:
    seir_label2 = seir_label[3:10]
err = np.mean(np.abs(df_test.value.values - seir_label2))
err_list.append(err)

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.025) & (
                df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_low = df_test[df_test.target_end_date.isin(eval_days)]

df_test = df_bench[
    (df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench['quantile'] == 0.975) & (
                df_bench.location_name == 'US')]
df_test = df_test[['target_end_date', 'value']].groupby('target_end_date').sum().reset_index()
df_high = df_test[df_test.target_end_date.isin(eval_days)]
length = np.mean(df_high.value.values - df_low.value.values)
length_list.append(length)

## crps
seir_label2 = seir_label[3:10]

df_test = df_bench[(df_bench.type == 'quantile') & (df_bench.target.isin(pred_name)) & (df_bench.location_name == 'US')]
df_test = df_test[df_test.target_end_date.isin(eval_days)]

avail_days = df_test.target_end_date.unique()

day = avail_days[0]
i = 0
crps_list = []

df_sub = df_test[df_test['target_end_date'] == day]

pred = df_sub.value.values
proba = df_sub['quantile'].values

less_ind = pred < seir_label2[i]
proba_label = np.ones_like(proba)
proba_label[less_ind] = 0
crps_list.append(np.mean((proba_label - proba) ** 2))

np.mean(np.array(crps_list))

cprs_list.append(np.mean(np.array(crps_list)))

err_df = pds.DataFrame({
    'model': model_list,
    'error': err_list,
    'length': length_list,
    'cprs': cprs_list,
    'eval_date': test_date,
    'forecast_date': train_end_date
})

err_df.to_csv('tables/benchmark-US-14d.csv')
