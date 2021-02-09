#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pds

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

pad = 24

data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
data_dict = pyro_model.helper.smooth_daily(data_dict)

df_list = []

for days in [14, 28, 42]:
    train_len = data_dict['cum_death'].shape[0] - days

    test_dates = data_dict['date_list'][train_len:]
    train_end_date = data_dict['date_list'][train_len].date()

    len(data_dict['date_list'][train_len:])

    # ## loading results

    # ## get benchmarks

    for test_len in [7, 14]:

        test_len = test_len - 1

        if days == 14:
            # 04-25: days = 14

            test_date = data_dict['date_list'][test_len - days].date()

            file_list = [
                '2020-04-23-LANL-GrowthRate.csv',
                '2020-04-24-JHU_IDD-CovidSP.csv',
                '2020-04-26-Imperial-ensemble2.csv',
                '2020-04-27-IHME-CurveFit.csv',
                '2020-04-27-MIT_CovidAnalytics-DELPHI.csv',
                '2020-04-27-YYG-ParamSearch.csv'
            ]
            file_list = ['covid19hub/benchmark-04-25/' + x for x in file_list]

            model_list = [
                'LANL',
                'JHU_IDD',
                'Imperial',
                'IHME',
                'MIT-DELPHI',
                'YYG'
            ]

            err_list = []
            for m, f in zip(model_list, file_list):
                df_bench = pds.read_csv(f)

                if m == 'LANL':
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                            df_bench.target.str[-9:] == 'cum death') & (df_bench.location.isin(range(1, 57)))]
                elif m in ['IHME', 'YYG']:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                            df_bench.target.str[-9:] == 'cum death') & (df_bench.location_name == 'US')]
                else:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                            df_bench.target.str[-9:] == 'cum death')]

                pred = np.sum(df_test.value.values)
                pred = pred if pred > 0 else np.nan
                err_list.append(pred - data_dict['actual_cum_death'][test_len - days][4].item())
        elif days == 28:
            test_date = data_dict['date_list'][test_len - days].date()

            file_list = [
                '2020-04-09-LANL-GrowthRate.csv',
                '2020-04-12-Imperial-ensemble2.csv',
                '2020-04-09-IHME-CurveFit.csv',
                '2020-04-13-YYG-ParamSearch.csv'
            ]

            file_list = ['covid19hub/benchmark-04-11/' + x for x in file_list]

            model_list = [
                'LANL',
                'Imperial',
                'IHME',
                'YYG'
            ]

            err_list = []
            for m, f in zip(model_list, file_list):
                df_bench = pds.read_csv(f)

                if m == 'LANL':
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death')]
                elif m in ['IHME', 'YYG']:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death') & (df_bench.location_name == 'US')]
                else:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death')]

                pred = np.sum(df_test.value.values)
                pred = pred if pred > 0 else np.nan
                err_list.append(pred - data_dict['actual_cum_death'][test_len - days][4].item())

        else:
            test_date = data_dict['date_list'][test_len - days].date()

            file_list = [
                '2020-03-29-Imperial-ensemble2.csv',
                '2020-03-27-IHME-CurveFit.csv'
            ]

            file_list = ['covid19hub/benchmark-03-28/' + x for x in file_list]

            model_list = [
                'Imperial',
                'IHME',
            ]

            err_list = []
            for m, f in zip(model_list, file_list):
                df_bench = pds.read_csv(f)

                if m == 'LANL':
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death')]
                elif m in ['IHME', 'YYG']:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death') & (df_bench.location_name == 'US')]
                else:
                    df_test = df_bench[(df_bench.type == 'point') & (df_bench.target_end_date == str(test_date)) & (
                                df_bench.target.str[-9:] == 'cum death')]

                pred = np.sum(df_test.value.values)
                pred = pred if pred > 0 else np.nan
                err_list.append(pred - data_dict['actual_cum_death'][test_len - days][4].item())

        err_df = pds.DataFrame({
            'model': model_list,
            'error': err_list,
            'eval_date': test_date,
            'forecast_date': train_end_date
        })
        df_list.append(err_df)

df_export = pds.concat(df_list)
df_export.to_csv('tables/benchmark-Table-1-US.csv')
