import numpy as np
import pandas as pds
from pandas.plotting import register_matplotlib_converters
from scipy.optimize import curve_fit

import data_loader
import pyro_model.helper
from sir_model.sir import f_factory_opt

register_matplotlib_converters()
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
    'Romania',
    'Portugal',
    'Sweden',
    'Switzerland',
    'Ireland',
    'Hungary',
    'Denmark',
    'Austria',
    'Poland',
]


model_id = 'testing'
days = 14
niter = 500
n_sample = 1000

data_dict = data_loader.get_data_pyro(countries)
data_dict = pyro_model.helper.smooth_daily(data_dict)

train_len = data_dict['cum_death'].shape[0] - days
n_country = len(countries)

covariates_notime = pyro_model.helper.get_covariates_intervention(data_dict, train_len, notime = True)
Y_train = data_dict['actual_cum_death'][:train_len, :]

total_len = len(data_dict['date_list'])
covariates_full_notime = pyro_model.helper.get_covariates_intervention(data_dict, total_len, notime = True)
Y_full = data_dict['actual_cum_death'][:total_len, :]
Y_daily = data_dict['actual_daily_death']


def f_logi(x, a, b, c):
    prediction = N * a * np.exp(-1. * b * np.exp(-1. * c * x))
    return prediction

pred_list = []
for i in range(len(data_dict['countries'])):

    ind = data_dict['t_init'][i]
    y_tmp = Y_train[ind:, i].numpy()
    covariates = np.arange(len(y_tmp)) * 1.
    covariates_total = np.arange(total_len - ind) * 1
    N = data_dict['population'][i]
    try:
        params, _ = curve_fit(f_logi, covariates, y_tmp, p0=[0.1, 0.8, 0.1])
        pred = f_logi(covariates_total, *params)
        pred = np.concatenate([np.zeros(int(ind)), pred])
    except RuntimeError:
        print(i)
        pred = np.ones(total_len) * y_tmp[-1]
    pred_list.append(pred)
pred = np.stack(pred_list, axis=-1)
d14 = pred[-1] - Y_full.numpy()[-1]
d7 = pred[-8] - Y_full.numpy()[-8]
res = {'d7': d7, 'd14':d14, 'countries': data_dict['countries']}
res_df_logi = pds.DataFrame(res)

res_df_logi.to_csv('tables/benchmark-Gompertz-apr-25.csv')


def f_sigmoid(x, a, b, c):
    prediction = N * a / (1 + np.exp(-1. * c * x + b))
    return prediction

pred_list = []
for i in range(len(data_dict['countries'])):

    ind = data_dict['t_init'][i]
    y_tmp = Y_train[ind:, i].numpy()
    covariates = np.arange(len(y_tmp)) * 1.
    covariates_total = np.arange(total_len - ind) * 1
    N = data_dict['population'][i]
    try:
        params, _ = curve_fit(f_sigmoid, covariates, y_tmp, p0=[0.1, 0.8, 0.1])
        pred = f_sigmoid(covariates_total, *params)
        pred = np.concatenate([np.zeros(int(ind)), pred])
    except RuntimeError:
        print(i)
        pred = np.ones(total_len) * y_tmp[-1]
    pred_list.append(pred)
pred = np.stack(pred_list, axis=-1)
d14 = pred[-1] - Y_full.numpy()[-1]
d7 = pred[-8] - Y_full.numpy()[-8]
res = {'d7': d7, 'd14':d14, 'countries': data_dict['countries']}
res_df_sigmoid = pds.DataFrame(res)

# res_df_sigmoid.to_csv('tables/sigmoid-apr-25.csv')

# SIR benchmark

R0 = 2.25
infectious_days = 4.5
theta = 0.14 * 0.001
alpha = 1.
beta = 0.5

pred_list = []
for i in range(len(data_dict['countries'])):

    ind = data_dict['t_init'][i]
    y_tmp = Y_train[ind:, i].numpy()
    covariates = np.arange(len(y_tmp)) * 1.
    covariates_total = np.arange(total_len - ind) * 1
    N = data_dict['population'][i]

    f_train = f_factory_opt(N, 0, len(y_tmp))
    f_total = f_factory_opt(N, 0, total_len - ind)
    best_err = 1E9
    best_params = None

    for Psi in range(14, 40):
        try:
            params, _ = curve_fit(f_train, covariates, y_tmp, p0=[R0, infectious_days, Psi, theta])
        except RuntimeError:
            params = [R0, infectious_days, Psi, theta]

        err_val = (f_train(0, *params) - y_tmp)
        err_val = np.sqrt(np.mean(err_val ** 2))
        if err_val < best_err:
            best_params = params
            best_err = err_val


    pred = f_total(0, *best_params)
    pred = np.concatenate([np.zeros(int(ind)), pred])
    pred_list.append(pred)

pred = np.stack(pred_list, axis=-1)
d14 = (pred[-1] - pred[-14] + Y_full.numpy()[-14]) - Y_full.numpy()[-1]
d7 = (pred[-8] - pred[-14] + Y_full.numpy()[-14]) - Y_full.numpy()[-8]
res = {'d7': d7, 'd14': d14, 'countries': data_dict['countries']}
res_df_sir = pds.DataFrame(res)
res_df_sir.to_csv('tables/benchmark-seir-apr-25.csv')
