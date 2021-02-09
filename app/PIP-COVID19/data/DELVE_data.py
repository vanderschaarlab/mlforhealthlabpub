
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
from datetime import datetime

from scipy.integrate import odeint

import warnings
warnings.filterwarnings('ignore')

from utils.processing import *


def get_COVID_DELVE_data(country_list):
    
    country_dict = dict.fromkeys(country_list)
    
    PATH         = "https://raw.githubusercontent.com/rs-delve/covid19_datasets/master/dataset/combined_dataset_latest.csv"
    covid_data   = pd.read_csv(PATH, parse_dates=["DATE"]) 
    
    for country in country_list:
        
        country_dict[country] = extract_data(covid_data, country_name=country)
    
    return country_dict
    
    
def extract_data(covid_data, country_name="United Kingdom"):
    
    mobility_indicators = ['mobility_retail_recreation', 'mobility_grocery_pharmacy', 'mobility_parks', 
                           'mobility_transit_stations', 'mobility_workplaces', 'mobility_residential', 
                           'mobility_travel_driving', 'mobility_travel_transit', 'mobility_travel_walking']
    
    policy_vars         = ['npi_school_closing', 'npi_workplace_closing', 'npi_cancel_public_events', 
                           'npi_gatherings_restrictions', 'npi_close_public_transport', 'npi_stay_at_home', 
                           'npi_internal_movement_restrictions', 'npi_international_travel_controls', 
                           'npi_income_support', 'npi_debt_relief', 'npi_fiscal_measures', 'npi_international_support', 
                           'npi_public_information', 'npi_testing_policy', 'npi_contact_tracing',
                           'npi_healthcare_investment', 'npi_vaccine_investment', 'npi_stringency_index', 'npi_masks']
    
    testing_vars        = ['tests_total', 'tests_new', 'tests_total_per_thousand', 'tests_new_per_thousand',
                           'tests_new_smoothed', 'tests_new_smoothed_per_thousand']
       
    country_metadata    = ['stats_population', 'stats_population_density', 'stats_median_age',
                           'stats_gdp_per_capita', 'stats_smoking', 'stats_population_urban', 
                           'stats_population_school_age']   

    weather_vars        = ['weather_precipitation_mean', 'weather_humidity_mean', 'weather_sw_radiation_mean', 
                           'weather_temperature_mean', 'weather_temperature_min', 'weather_temperature_max',
                           'weather_wind_speed_mean']
    
    # Read the data
    covid_data             = covid_data.loc[covid_data["country_name"]==country_name]
 
    # Process the data
    weather_data           = covid_data[weather_vars]
    testing_data           = covid_data[testing_vars]

    metadata               = np.array(covid_data[country_metadata])[0, :]
    meta_data_dict         = dict.fromkeys(country_metadata)
    
    for k in range(len(country_metadata)):
        
        meta_data_dict[country_metadata[k]] = metadata[k]
    
    mobility_data          = covid_data[mobility_indicators]
    dates                  = covid_data["DATE"]    
    daily_deaths_data      = np.array(covid_data["deaths_new"])
    NPI_data               = covid_data[policy_vars]
    daily_cases_data       = np.array(covid_data["cases_new"])
 
    mobility_scores        = smoothen_mobility_scores(np.array(mobility_data))/100
    
    
    for k in range(mobility_scores.shape[1]):
    
        mobility_scores[np.where(np.isnan(mobility_scores[:, k]))[0], k] = 0 
    
        mobility_scores[:, k]   = np.hstack((np.zeros(6), moving_average(mobility_scores[:, k], n=7)))
    
    
    data_output            = dict({"Daily deaths":daily_deaths_data,
                                   "Daily cases": daily_cases_data,
                                   "Mobility data": mobility_data,
                                   "Smoothened mobility data": mobility_scores,
                                   "NPI data": NPI_data,
                                   "Metadata": meta_data_dict , 
                                   "Data dates": dates,
                                   "wheather data":weather_data,
                                   "testing data":testing_data})
    
    return data_output
