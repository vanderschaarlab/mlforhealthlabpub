import pandas as pds
import pyro
import torch
import pickle
from pandas.plotting import register_matplotlib_converters
from forecast import Forecaster
from pyro.ops.stats import quantile

import data_loader
import pyro_model.helper
import pyro_model.seir_gp
import argparse
import os

parser = argparse.ArgumentParser('CGP')
parser.add_argument('--days', type=str, default='14')
args = parser.parse_args()
days = int(args.days)

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

niter = 2000
n_sample = 500
pad = 24
data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
data_dict = pyro_model.helper.smooth_daily(data_dict)

train_len = data_dict['cum_death'].shape[0] - days
n_country = len(countries)

covariates_notime = pyro_model.helper.get_covariates_intervention(data_dict, train_len, notime=True)
Y_train = pyro_model.helper.get_Y(data_dict, train_len)

total_len = len(data_dict['date_list'])
covariates_full_notime = pyro_model.helper.get_covariates_intervention(data_dict, total_len, notime=True)
Y_full = pyro_model.helper.get_Y(data_dict, total_len)
Y_daily = data_dict['daily_death']

model_id_base = 'day-{}-rng-{}'

if not os.path.exists('Loop{}'.format(days)):
    os.makedirs('Loop{}'.format(days))

for seed in range(15):
    print('running with seed {}'.format(seed))
    model_id = model_id_base.format(days, seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    model = pyro_model.seir_gp.CGP(data_dict, mask_size=14)
    try:
        forecaster = Forecaster(model, Y_train, covariates_notime, learning_rate=0.01, num_steps=niter)
    except RuntimeError:
        continue

    with open('Loop{}/{}-forecaster.csv'.format(days, model_id), 'wb') as f:
        pickle.dump(forecaster, f)

    samples = forecaster(Y_train, covariates_full_notime, num_samples=n_sample, batch_size=50)

    samples = samples.squeeze()
    init = Y_train[-1, :][None, None, :]
    init = init.repeat(samples.shape[0], 1, 1)
    samples = torch.cat([init, samples], dim=1)
    daily_s = samples[:, 1:, :] - samples[:, :-1, :]
    p10, p50, p90 = quantile(daily_s, (0.1, 0.5, 0.9), dim=0).squeeze(-1)

    rmse = torch.sqrt(torch.mean((p50[-days:, :] - Y_daily[-days:, :]) ** 2, dim=0)).squeeze().numpy()
    off = (torch.sum(p50[-days:, :], dim=0) - torch.sum(Y_daily[-days:, :], dim=0)).squeeze().numpy()
    for i in zip(countries, rmse, off):
        print(i)
    d = {'countries': countries, 'rmse': rmse, 'total_error': off}
    df = pds.DataFrame(data=d)
    df.to_csv('Loop{}/{}-rmse.csv'.format(days, model_id))

    with open('Loop{}/{}-samples.pkl'.format(days, model_id), 'wb') as f:
        pickle.dump(samples.detach().numpy(), f)
    R0low, R0mid, R0high, map_estimates = model.get_R0(forecaster, Y_train, covariates_full_notime, n_sample, 50)

    with open('Loop{}/{}-map.pkl'.format(days, model_id), 'wb') as f:
        pickle.dump(map_estimates, f)

    with torch.no_grad():
        predictor = pyro.infer.predictive.Predictive(forecaster.model, guide=forecaster.guide, num_samples=100)
        res = predictor(Y_train, covariates_notime)

    with open('Loop{}/{}-predictive.pkl'.format(days, model_id), 'wb') as f:
        pickle.dump(res, f)
