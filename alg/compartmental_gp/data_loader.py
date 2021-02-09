import pandas as pds
import numpy as np
from datetime import datetime
import torch


def numpy_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


def get_intervention(country, standarize=False, smooth=True, legacy=False):
    csvs = [
        'c1_schoolclosing.csv',
        'c2_workplaceclosing.csv',
        'c3_cancelpublicevents.csv',
        'c4_restrictionsongatherings.csv',
        'c5_closepublictransport.csv',
        'c6_stayathomerequirements.csv',
        'c7_domestictravel.csv',
        'c8_internationaltravel.csv',
        'e1_incomesupport.csv',
        'e2_debtcontractrelief.csv',
        'h1_publicinfocampaign.csv',
        'h2_testingpolicy.csv'
    ] + ['c{}_flag.csv'.format(x) for x in range(1, 8)] + ['e1_flag.csv', 'h1_flag.csv']

    if not legacy:
        files = ['ox-policy-tracker/data/timeseries/{}'.format(i) for i in csvs]
    else:
        files = ['covid-policy-tracker-legacy/data/timeseries/{}'.format(i) for i in csvs]

    idx_list = []

    for f in files:
        dat_ox = pds.read_csv(f)
        dat_ox.rename(columns={'Unnamed: 0': 'country', 'Unnamed: 1': 'country_code'}, inplace=True)
        dat_ox[dat_ox == '.'] = 'NaN'
        dt_list = [datetime.strptime(x, '%d%b%Y').date() for x in dat_ox.columns[2:]]

        dat_country = dat_ox[dat_ox['country'] == country]

        index_country = dat_country.iloc[0, 2:].values.astype(np.float)
        # fill na with previous value
        index_country = numpy_fill(index_country[None, :])
        # handle the case of initial zeros
        index_country[np.isnan(index_country)] = 0
        idx_list.append(index_country[0, :])

    idx = np.stack(idx_list, -1)
    if standarize:
        idx = (idx - np.mean(idx, axis=0)) / np.std(idx, axis=0)
        idx[np.isnan(idx)] = 0
    if smooth:
        dy_list = list()
        for i in range(idx.shape[1]):
            ds = idx[:, i]
            dy = smooth_curve_1d(ds)
            dy_list.append(dy)
        idx = np.stack(dy_list, axis=-1)

    return idx


def smooth_curve_1d(x):
    w = np.ones(7, 'd')
    y = np.convolve(w / w.sum(), x, mode='valid')
    y = np.concatenate([np.zeros(3), y])
    return y


def get_deaths(country, to_torch=False, legacy=False, smart_start=True, pad=0, rebuttal=False):
    # get time series
    if not legacy:
        file = 'ts-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    else:
        file = 'COVID-19-legacy/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

    if rebuttal:
        file = 'COVID-19-rebuttal-08-10/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    dat = pds.read_csv(file)
    dt_list = [datetime.strptime(x, '%m/%d/%y').date() for x in dat.columns[4:]]

    if country not in ['China', 'Canada']:
        country_data = dat[(dat['Country/Region'] == country) & (dat['Province/State'].isnull())].iloc[0, 4:].values
    else:
        country_data = np.sum(dat[(dat['Country/Region'] == country)].iloc[:, 4:].values, axis=0)

    ind = (country_data != 0).argmax() - pad
    if ind < 0:
        print(country)
        ind = 0
    # assert ind >= 0

    cum_deaths = country_data[ind:].astype(np.float64)
    dt_list = dt_list[ind:]
    daily_deaths = np.diff(np.append(np.zeros(1), cum_deaths))

    if country == 'Philippines':
        cum_deaths = cum_deaths[39:]
        dt_list = dt_list[39:]
        daily_deaths = daily_deaths[39:]
    if country == 'France':
        cum_deaths = cum_deaths[17:]
        dt_list = dt_list[17:]
        daily_deaths = daily_deaths[17:]

    # get population
    dat_feat = pds.read_csv('country_feature/country_feats.csv')
    if country == 'US':
        p_country = 'United States'
    elif country == 'Korea, South':
        p_country = 'Korea, Rep.'
    elif country == 'Iran':
        p_country = 'Iran, Islamic Rep.'
    elif country == 'Russia':
        p_country = 'Russian Federation'
    elif country == 'Egypt':
        p_country = 'Egypt, Arab Rep.'
    else:
        p_country = country
    population = dat_feat[(dat_feat['Country.Name'] == p_country) & (dat_feat['metric'] == 'Population, total')]
    population = population['value'].values[0]

    # define the starting point
    if smart_start:
        rate = 3.061029261722505e-08
        daily_death_min = rate * population
        ind_death = ((daily_deaths >= daily_death_min) * .1).argmax()
        cum_deaths = cum_deaths[ind_death:]
        dt_list = dt_list[ind_death:]
        daily_deaths = daily_deaths[ind_death:]

    # get oxford index
    if not legacy:
        dat_ox = pds.read_csv('ox-policy-tracker/data/timeseries/stringencyindex_legacy.csv')
    else:
        dat_ox = pds.read_csv('covid-policy-tracker-legacy/data/timeseries/stringencyindex_legacy.csv')
    dat_ox.rename(columns={'Unnamed: 0': 'country', 'Unnamed: 1': 'country_code'}, inplace=True)
    dt_list_ind = [datetime.strptime(x, '%d%b%Y').date() for x in dat_ox.columns[2:]]
    dat_ox[dat_ox == '.'] = 'NaN'
    if country == 'US':
        o_country = 'United States'
    elif country == 'Korea, South':
        o_country = 'South Korea'
    else:
        o_country = country
    dat_country = dat_ox[dat_ox['country'] == o_country]
    # 7d mv smooth
    index_country = dat_country.iloc[0, 2:].values.astype(np.float)
    ind_len = len(index_country)
    index_country = smooth_curve_1d(index_country)[:ind_len]
    index_country[np.isnan(index_country)] = np.nanmean(index_country)

    intervention = get_intervention(o_country, legacy)

    if not to_torch:
        return {
            'dt': dt_list,
            'cum_death': cum_deaths,
            'daily_death': daily_deaths,
            'population': population,
            's_index_dt': dt_list_ind,
            's_index': index_country,
            'intervention': intervention
        }
    else:
        return {
            'dt': dt_list,
            'cum_death': torch.tensor(cum_deaths),
            'daily_death': torch.tensor(daily_deaths),
            'population': population,
            's_index_dt': dt_list_ind,
            's_index': torch.tensor(index_country),
            'intervention': torch.tensor(intervention)
        }


def pad_sequence_trailing(sequences, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[-length:, i, ...] = tensor

    return out_tensor


def cut_s_index(data_dict):
    ind = data_dict['s_index_dt'].index(data_dict['dt'][0])
    s_len = len(data_dict['cum_death'])
    s_index = data_dict['s_index'][ind:ind + s_len]
    intervention = data_dict['intervention'][ind:ind + s_len]
    return s_index, intervention


def get_data_pyro(countries, legacy=False, smart_start=True, pad=0, rebuttal=False):
    data_list = [get_deaths(x, True, legacy, smart_start, pad, rebuttal) for x in countries]
    init_days = [x['dt'][0] for x in data_list]
    init_day = min(init_days)
    t_first_blood = [(x - init_day).days for x in init_days]

    cum_death = pad_sequence_trailing([x['cum_death'] for x in data_list])
    daily_death = pad_sequence_trailing([x['daily_death'] for x in data_list])

    si_cut = [cut_s_index(x) for x in data_list]

    s_index = pad_sequence_trailing([x[0] for x in si_cut]) / 100
    i_index = pad_sequence_trailing([x[1] for x in si_cut])
    N_list = [x['population'] for x in data_list]

    date_list = pds.date_range(init_day, periods=cum_death.size(0))

    country_feat = get_country_feature(countries)
    feat_list = [
        'Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)',
        'Mortality rate, adult, male (per 1,000 male adults)',
        'Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)',
        'Incidence of tuberculosis (per 100,000 people)',
        'Immunization, measles (% of children ages 12-23 months)',
        'Immunization, DPT (% of children ages 12-23 months)',
        'Immunization, HepB3 (% of one-year-old children)',
        'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)',
        'Prevalence of overweight (% of adults)'
    ]

    country_feat = country_feat[country_feat.metric.isin(feat_list)]
    dat_feat = country_feat.pivot('country', 'metric', 'value')

    feat = np.zeros_like(dat_feat.values)
    for i in range(len(countries)):
        feat[i] = dat_feat.loc[countries[i]].values

    feat = (feat - np.nanmean(feat, axis=0)) / np.nanstd(feat, axis=0)
    feat[np.isnan(feat)] = 0.

    return {
        'cum_death': cum_death,
        'daily_death': daily_death,
        's_index': s_index,
        'i_index': i_index,
        'population': N_list,
        't_init': torch.tensor(t_first_blood).unsqueeze(-1),
        'date_list': date_list,
        'countries': countries,
        'country_feat': torch.tensor(feat).to(i_index)
    }


def get_country_feature(country_list):
    dat_feat = pds.read_csv('country_feature/country_feats.csv')

    p_country_list = []

    for country in country_list:

        if country == 'US':
            p_country = 'United States'
        elif country == 'Korea, South':
            p_country = 'Korea, Rep.'
        elif country == 'Iran':
            p_country = 'Iran, Islamic Rep.'
        elif country == 'Russia':
            p_country = 'Russian Federation'
        elif country == 'Egypt':
            p_country = 'Egypt, Arab Rep.'
        else:
            p_country = country
        p_country_list.append(p_country)

    dat_feat = dat_feat[(dat_feat['Country.Name'].isin(p_country_list))]
    del dat_feat['Country.Code']
    dat_feat['country'] = dat_feat['Country.Name']
    del dat_feat['Country.Name']

    countries = dat_feat['country'].values
    countries[countries == 'United States'] = 'US'
    countries[countries == 'Korea, Rep.'] = 'Korea, South'
    countries[countries == 'Iran, Islamic Rep.'] = 'Iran'
    countries[countries == 'Russian Federation'] = 'Russia'
    countries[countries == 'Egypt, Arab Rep.'] = 'Egypt'
    dat_feat['country'] = list(countries)

    return dat_feat
