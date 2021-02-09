
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


# IF EXISTS REMOVE OS.MKDIR 

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import pandas as pd
import timeit
import logging
import traceback
from datetime import date
import pickle
import json
import os
from os import path
import warnings
warnings.filterwarnings('ignore')

today       = date.today()

key_path    = os.getcwd() + "\\" + str(today)
model_path  = os.getcwd() + "\\" + str(today) + "\\" + "models" 
result_path = os.getcwd() + "\\" + str(today) + "\\" + "results" 
projec_path = os.getcwd() + "\\" + str(today) + "\\" + "projections" 

try:
    
    os.mkdir(key_path)
    os.mkdir(model_path)
    os.mkdir(result_path)
    os.mkdir(projec_path)

except OSError:
    
    print ("Creation of the directory %s failed" % model_path)
    
else:
    
    print ("Successfully created the directory %s " % model_path)


from utils.processing import *
from data.DELVE_data import *
from model.base_model import *

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

hyperparam_config = pickle.load(open(os.getcwd() + "\\config\\hyperparameters", "rb"))


def main(args):

    """

    country_list = ["United Kingdom", "Egypt", "Japan", "Israel", "Croatia",  
                    "United States", "Australia", "Italy", "Germany", "Spain", 
                    "Brazil", "Canada", "Sweden", "Norway",
                    "Finland", "Estonia", "Switzerland", "Ireland", "France",
                    "Netherlands", "Czechia", "Romania", "Belgium", "Algeria",
                    "Argentina", "Chile", "Colombia", "Turkey", "South Africa", 
                    "India", "Pakistan", "Indonesia"]
    
    """

    country_list = ["United States", "United Kingdom", "Italy", "Germany", "Brazil", "Japan", "Egypt"]

    country_dict = get_COVID_DELVE_data(country_list)
    
    N_train      = args.n_train  
    onset_range  = np.linspace(5, 80, 16)
    pred_horizon = 30 * 5

    for country in country_list:
        
        print("\n-----------------------------------------")
        print("    Fitting model for %s" % country)
        print("-----------------------------------------")
        
        deaths_true   = country_dict[country]["Daily deaths"]
        cases_true    = country_dict[country]["Daily cases"]

        deaths_true[deaths_true < 0] = 0
        cases_true[cases_true < 0]   = 0

        deaths_smooth = smooth_curve_1d(deaths_true)
        N_population  = country_dict[country]["Metadata"]["stats_population"] 
        mobility      = country_dict[country]["Smoothened mobility data"]
        NPI_data      = np.array(country_dict[country]["NPI data"])


        if args.hyper_opt==1:

            onset_loss                                 = []

            try:

                for onset in onset_range:

                    start                              = timeit.default_timer()

                    SEIR_model                         = SEIR_base(N_population=N_population, 
                                                                   T_infectious=7, T_incubation=5,
                                                                   hyperparameters=dict({"outbreak_shift":int(onset), 
                                                                                     "change_pts":-1})) 
        
                    SEIR_model.fit(deaths_true[:N_train], cases_true[:N_train], 
                                   NPI_data[:N_train, :], mobility[:N_train, :])

                    days                               = len(deaths_true) 
                    deaths_pred, cases_pred, R0_t_pred = SEIR_model.predict(days)

                    onset_loss.append(compute_loss(deaths_smooth[:N_train], deaths_pred[:N_train]))
                
                    stop                               = timeit.default_timer()
                    runtime_onset                      = (stop-start)/60

                    print("\n * Onset %s --- \n   Accuracy %.2f --- \n   Runtime %f" % (onset, onset_loss[-1], runtime_onset))

                best_onset    = onset_range[np.argmin(np.array(onset_loss))] 
 
                SEIR_model    = SEIR_base(N_population=N_population, 
                                          T_infectious=7, T_incubation=5,
                                          hyperparameters=dict({"outbreak_shift":int(best_onset), 
                                                                "change_pts":-1})) 
        
                SEIR_model.fit(deaths_true[:N_train], cases_true[:N_train], 
                               NPI_data[:N_train, :], mobility[:N_train, :])

            
            except:

                print("Cannot fit the model for %s" % country)
                traceback.print_exc() 

        else: 

                SEIR_model    = SEIR_base(N_population=N_population, 
                                          T_infectious=7, T_incubation=5,
                                          hyperparameters=dict({"outbreak_shift":hyperparam_config[country], 
                                                                "change_pts":-1})) 
        
                SEIR_model.fit(deaths_true[:N_train], cases_true[:N_train], 
                               NPI_data[:N_train, :], mobility[:N_train, :])
            
        days                               = len(deaths_true) + pred_horizon
        deaths_pred, cases_pred, R0_t_pred = SEIR_model.predict(days)
        deaths_pred_u, deaths_pred_l       = deaths_pred, deaths_pred #SEIR_model.compute_confidence_intervals(horizon=pred_horizon)

        pickle.dump(SEIR_model, open(model_path + "\\" + country, 'wb'))
        pickle.dump((deaths_pred, deaths_pred_u, deaths_pred_l, R0_t_pred), open(projec_path + "\\" + country, 'wb'))

        plt.figure(figsize=(9, 6))

        x_data        = np.linspace(0, days - 1, days, dtype=int) 
        y_max         = np.maximum(np.max(deaths_true), np.max(deaths_pred)) + 1

        plt.plot(deaths_true, linewidth=1, marker="o", markersize=6, alpha=0.25, color="r", zorder=1)
        plt.plot(x_data[:N_train], deaths_pred[:N_train], color="grey", linewidth=2, zorder=2) 
        plt.plot(x_data[N_train:], deaths_pred[N_train:], color="black", linestyle="--", linewidth=2, zorder=3) 
        plt.vlines(x=[N_train], ymin=0, ymax=y_max, linewidth=1, color="black", zorder=4) 

        plt.plot(x_data[:N_train], deaths_smooth[:N_train], linestyle=":", linewidth=2, color="red", zorder=5)
        plt.plot(x_data[N_train:len(deaths_true)], deaths_smooth[N_train:], linestyle="--", linewidth=2, color="red", zorder=5)

        #plt.fill_between(x_data[N_train + 1:], deaths_pred_u, deaths_pred_l, color="r", alpha=0.5)

        plt.vlines(x=[N_train + k + 1 for k in range(len(x_data)-N_train)], ymin=0, ymax=y_max, 
                   linewidth=5, color="pink", alpha=0.3, linestyles="--", zorder=0) 

        plt.xlabel("Number of days since the outbreak", labelpad=22)
        plt.ylabel("Daily deaths", labelpad=22)

        plt.xlim(30, days)

        plt.tight_layout()
        plt.savefig(str(today) + "\\" + "results\\" + country + "_projections.pdf", dpi=3000, transparent=True)
   
            


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Fitting global SEIR models")

    parser.add_argument("-n", "--n-train", default=200, type=int)
    parser.add_argument("-v", '--hyper-opt', default=1, type=int)

    args = parser.parse_args()
    
    main(args)


# Conf Interval
# Each country has default policy
# Model update time

