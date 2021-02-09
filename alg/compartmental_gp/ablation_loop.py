import os
import pickle

import pyro
import torch
from pandas.plotting import register_matplotlib_converters
from pyro.ops.stats import quantile

import data_loader
import pyro_model.helper
import pyro_model.seir_gp
from forecast import Forecaster

register_matplotlib_converters()
countries_list = [
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
    'Sweden',
    'Mexico',
    'India',
    'Russia',
    'Japan',
    'South Africa',
    'Egypt',
    'Norway'
]
days = 14

if not os.path.exists('AblationLoop{}'.format(days)):
    os.makedirs('AblationLoop{}'.format(days))

for country in countries_list:
    countries = [country]

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

    model_id_base = country + '-ablation-day-{}-rng-{}'


    for seed in range(0, 10):
        print('running with seed {}'.format(seed))
        model_id = model_id_base.format(days, seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        model = pyro_model.seir_gp.CGP(data_dict, mask_size=14)
        try:
            forecaster = Forecaster(model, Y_train, covariates_notime, learning_rate=0.01, num_steps=niter)
        except RuntimeError:
            continue

        samples = forecaster(Y_train, covariates_full_notime, num_samples=n_sample, batch_size=50)
        samples = samples[:, 0, ...]
        init = Y_train[-1, :][None, None, :]
        init = init.repeat(samples.shape[0], 1, 1)
        samples = torch.cat([init, samples], dim=1)
        daily_s = samples[:, 1:, :] - samples[:, :-1, :]
        p10, p50, p90 = quantile(daily_s, (0.1, 0.5, 0.9), dim=0).squeeze(-1)

        with open('AblationLoop{}/{}-samples.pkl'.format(days, model_id), 'wb') as f:
            pickle.dump(samples.detach().numpy(), f)
        R0low, R0mid, R0high, map_estimates = model.get_R0(forecaster, Y_train, covariates_full_notime, n_sample, 50)

        with open('AblationLoop{}/{}-map.pkl'.format(days, model_id), 'wb') as f:
            pickle.dump(map_estimates, f)

        with torch.no_grad():
            predictor = pyro.infer.predictive.Predictive(forecaster.model, guide=forecaster.guide, num_samples=100)
            res = predictor(Y_train, covariates_notime)

        with open('AblationLoop{}/{}-predictive.pkl'.format(days, model_id), 'wb') as f:
            pickle.dump(res, f)

        with open('AblationLoop{}/{}-forecaster.pkl'.format(days, model_id), 'wb') as f:
            pickle.dump(forecaster, f)
