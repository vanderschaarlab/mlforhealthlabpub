
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


# IF EXISTS REMOVE OS.MKDIR 

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import pandas as pd
import logging
from datetime import date
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from utils.processing import *
from data.DELVE_data import *

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
import ruptures as rpt
from datetime import datetime
#from bayes_opt import BayesianOptimization

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3

from scipy.integrate import odeint

import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
import ruptures as rpt
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3

from scipy.integrate import odeint

import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import warnings
warnings.filterwarnings('ignore')


def sample_binomial(n, p):
    
    if (n < 1000) and (n > 0):
        
        sampled_n = np.random.binomial(n=n, p=p)
        
    elif n < 0:  
        
        sampled_n = 0
    
    else:
        
        sampled_n = int(np.random.normal() * np.sqrt(n * p * (1-p)) + n * p)
    
    return sampled_n
    


def collapse_change_points(input_nkbps, offset_duration=14):

    new_nkbps = []

    for k in range(len(input_nkbps)):
    
        if k==0:
        
            new_nkbps.append(input_nkbps[k])
        
        elif input_nkbps[k] - input_nkbps[k-1] >= offset_duration:    
        
            new_nkbps.append(input_nkbps[k])
    
        else:
        
            new_nkbps[-1] = input_nkbps[k]
    
    return new_nkbps  



class SEIR_base:

    def __init__(self, N_population, T_incubation=3, T_infectious=9, #T_incubation=1-14, T_infectious=7-12, 
                 T_IC=12, T_CD=7.5, T_CR=6.5, fit_on="deaths",
                 hyperparameters=dict({"outbreak_shift":20, "change_pts":2})):


        """
         Set the model parameters using user input with default values from
         current clinical literature
        """
        
        # Fixed parameters
        
        self.gamma  = 1/T_infectious                      # Infection rate
        self.sigma  = 1/T_incubation                      # Incubation rate
        self.T_IC   = T_IC                                # Number of days from infected to critical
        self.T_CD   = T_CD                                # Number of days from critical to dead
        self.T_CR   = T_CR                                # Number of days from critical to recovered
        self.chpts  = hyperparameters["change_pts"]       # Number of change points in the R0 over time 
        self.shift  = hyperparameters["outbreak_shift"]
        self.fit_on = fit_on
        
        # Clear/Initialize learnable parameters
        
        self.p_IC   = 0   
        self.p_CD   = 0
        self.N      = N_population
    
        
        self.beta_pred   = None
        self.N_train     = 0
        
        self.initialze_hyperparameters()
            
    
    def initialze_hyperparameters(self):
        
        R0_params   = ["R_0_"+str(k) for k in range(self.chpts + 1)] 
        k_params    = ["k_"+str(k) for k in range(self.chpts)] 
        
        R0_dict     =  dict.fromkeys(R0_params)
        k_dict      =  dict.fromkeys(k_params)
        
        self.R0s    = [1.0] * (self.chpts + 1)
        self.k      = [1.0] * (self.chpts)
        
        self.params_init = {"p_IC": (0.05, 0.01, 0.1), 
                            "p_CD": (0.5, 0.05, 0.8)} 
        
        self.params_init.update(R0_dict)
        self.params_init.update(k_dict)
        
        for k in range(len(R0_params)):
            
            self.params_init[R0_params[k]] = (1.0, 0.25, 6) 
            
        for m in range(len(k_params)):
            
            self.params_init[k_params[m]]  = (2.5, 0.01, 5.0)
                
        self.params      = dict(self.params_init)

        
    def ODEs(self, compartments, t):
        
        #  Ordinary differential equations modeling the 
        #  compartments of the SEIR model

        
        S, E, I, C, R, D = compartments
        
        mu_IC            = self.p_IC / self.T_IC
        mu_CD            = self.p_CD /self.T_CD  
    
        dSdt = -self.beta(t) * I * S / self.N
        dEdt = self.beta(t) * I * S / self.N - self.sigma * E
        dIdt = self.sigma * E - mu_IC * I - self.gamma * (1 - self.p_IC) * I
        dCdt = mu_IC * I - mu_CD * C - (1 - self.p_CD) * (1/self.T_CR) * C
        dRdt = self.gamma * (1 - self.p_IC) * I + (1 - self.p_CD) * (1/self.T_CR) * C
        dDdt = mu_CD * C 
    
        return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    
    
    def compute_R_0(self, t):
        
        """
          Evaluate the R0 over time
          
        """
        
        Rt_     = []
        
        for k in range(self.chpts):
        
            Rt_curr  = (self.R0s[k] - self.R0s[k+1]) / (1 + np.exp(-1 * self.k[k] * (-t + self.n_bkps[k+1]))) + self.R0s[k+1]
            Rt_.append(Rt_curr)  

        t_mid   = [int((self.n_bkps[k+1] + self.n_bkps[k+2]) / 2) for k in range(self.chpts-1)]
        ret_val = None
        
        for m in range(len(t_mid)):
            
            if (t >= t_mid[m]):
                
                ret_val = Rt_[m + 1]

        if (self.chpts== 1) or (ret_val is None):
            
            ret_val = Rt_[0]
            
        return ret_val

    
    def beta(self, t):
        
        """
          Evaluate the contact rate
          
        """
        
        if (np.floor(t) <= self.N_train) or (self.beta_pred is None):
            
            beta_ = self.compute_R_0(t) * (self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma) / self.sigma
            
        else:
      
            beta_ = self.beta_pred[np.minimum(int(t - self.N_train), len(self.beta_pred) - 1)]  
        
        return beta_

    
    def evaluate(self, days, **kwargs): 
        
        """
          Evaluate the model compartments
          
        """
        
        self.p_CD        = kwargs["p_CD"]
        self.p_IC        = kwargs["p_IC"]
        
        for k in range(len(self.R0s)):
            
            self.R0s[k]  = kwargs["R_0_" + str(k)]
            
        for m in range(len(self.k)):
        
            self.k[m]    = kwargs["k_" + str(m)]
    
        compartments_0   = self.N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0
        t                = np.linspace(0, days - 1, days)
        ret              = odeint(self.ODEs, compartments_0, t)
        S, E, I, C, R, D = ret.T
        R0_t             = [(self.beta(i) * self.sigma) / ((self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma)) for i in range(len(t))]
                
        return t, S, E, I, C, R, D, R0_t
    
    
    def fit(self, deaths_train, cases_train, NPI_train, mobility_train, 
            fit_method="least_squares", smooth=True, n_avg=7): 
        
        lambda_l1       = 3
        L_reg           = 0.01
        
        cases_deaths    = dict({"deaths":np.cumsum(smooth_curve_1d(deaths_train[self.shift:])),    
                                "cases":np.cumsum(smooth_curve_1d(cases_train[self.shift:])),
                                "mix":np.hstack((np.cumsum(smooth_curve_1d(deaths_train[self.shift:])), 
                                                 L_reg * np.cumsum(smooth_curve_1d(cases_train[self.shift:]))))})
        
        NPI_masks       = np.array(NPI_train)[:, 18]
        NPI_non_sd      = np.hstack((np.array([0]), np.diff(NPI_masks)))
        msk_n_bkps      = list(np.where(NPI_non_sd > 0)[0])
  
        cpmodel         = rpt.Pelt(model="rbf").fit(mobility_train[self.shift:, :])
        self.n_bkps     = cpmodel.predict(pen=lambda_l1)
        self.n_bkps     = [0] + list(np.sort(np.array(msk_n_bkps + self.n_bkps))) 
        self.n_bkps     = collapse_change_points(self.n_bkps, offset_duration=7)
        self.chpts      = len(self.n_bkps) - 2  
 
        self.initialze_hyperparameters()
    
        y_train         = cases_deaths[self.fit_on]
        
        """
          Train the model
          
        """        
        
        self.N_train    = y_train.shape[0] 
         
        mobility_scores = mobility_train[self.shift:, :]
        
        days            = len(deaths_train[self.shift:])
        x_days          = np.linspace(0, days - 1, days, dtype=int)  

        param_          = self.params

        
        def fitter(t, **param_): 
            
            ret = self.evaluate(len(deaths_train[self.shift:]), **param_)
            
            if self.fit_on=="deaths":
                
                out = ret[6][t]
            
            elif self.fit_on=="cases": 
                
                out = ret[3][t]
                
            elif self.fit_on=="mix":
                
                out = np.hstack((ret[6][t], L_reg * ret[3][t]))
    
            return out
        
        mod    = lmfit.Model(fitter)

        for kwarg, (init, mini, maxi) in self.params_init.items():
            
            mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

        params           = mod.make_params()
        train_result     = mod.fit(y_train, params, method=fit_method, t=x_days)
        
        self.best_params = dict(train_result.best_values)
        #self.opt_results = train_result
        #self.lmfit_model = mod

        # optimize last R0 period
        
        Y_trn  = smooth_curve_1d(deaths_train[self.shift:])[self.N_train - 30:]
        R_L    = np.linspace(0.25, 6.0, 1000)
        losses = []

        for R_ in R_L:
            
            self.best_params["R_0_" + str(self.chpts)] = R_
            deaths_pred, _, _                          = self.predict(days + self.shift)
            Y_pred                                     = np.array(deaths_pred[self.N_train + self.shift - 30:])

            losses.append(np.mean(np.abs(Y_trn - Y_pred)))
            
        self.best_params["R_0_" + str(self.chpts)] = R_L[np.argmin(np.array(losses))]   
        
    
    def predict(self, days, R0_forecast=None):
        
        if R0_forecast is None:
            
            self.beta_pred   = None
            
        else:    
        
            self.beta_pred   = R0_forecast * (self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma) / self.sigma
        
        if days > self.shift:
            
            params              = self.best_params
            outputs             = self.evaluate(days - self.shift, **params)
            cases, deaths, R0_t = outputs[3], outputs[6], outputs[7]
        
            deaths              = np.array([0] + list(np.diff(deaths)))
            
            cases               = np.hstack((np.zeros(self.shift), cases))
            deaths              = np.hstack((np.zeros(self.shift), deaths))
            R0_t                = np.hstack((R0_t[0] * np.ones(self.shift), R0_t))
            
        else:    
            
            cases        = np.zeros(days)
            deaths       = np.zeros(days)
            R0_t         = np.zeros(days)
            
        return deaths, cases, R0_t
    
    
    def sample_compartments(self, t, S, E, I, C, R, D):
    
        mu_IC    = self.p_IC / self.T_IC
        mu_CD    = self.p_CD / self.T_CD
        
        I_sample = sample_binomial(n=int(S), p=(I / self.N))
        E_sample = sample_binomial(n=int(E), p=self.sigma)
        I_IC     = sample_binomial(n=int(I), p=mu_IC) 
        C_CD     = sample_binomial(n=int(C), p=mu_CD)
        C_CR_CD  = sample_binomial(n=int(C), p=(1 - self.p_CD) * (1 / self.T_CR))
        I_IC_g   = sample_binomial(n=int(I), p=self.gamma * (1 - self.p_IC))    
    
        dSdt     = -1 * self.beta(t) * I_sample 
        dEdt     = self.beta(t) * I_sample - E_sample
        dIdt     = E_sample - I_IC - sample_binomial(n=int(I), p=self.gamma * (1 - self.p_IC))
        dCdt     = I_IC - C_CD - C_CR_CD
        dRdt     = I_IC_g + C_CR_CD 
        dDdt     = C_CD
    
        return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt


    def sample_forecasts(self, horizon=30):
        
        t_all, S_all, E_all, I_all, C_all, R_all, D_all, _ = self.evaluate(self.N_train, **self.best_params)
        t, S, E, I, C, R, D                                = t_all[-1], S_all[-1], E_all[-1], I_all[-1], C_all[-1], R_all[-1], D_all[-1]  
        forecast_                                          = []  
    
        for t_ in range(horizon):
    
            dSdt, dEdt, dIdt, dCdt, dRdt, dDdt = self.sample_compartments(self.N_train + t_, S, E, I, C, R, D)
        
            S = S + dSdt 
            E = E + dEdt
            I = I + dIdt 
            C = C + dCdt 
            R = R + dRdt 
            D = D + dDdt

            forecast_.append(D)
    
        return forecast_


    def compute_confidence_intervals(self, horizon=30, n_samples=1000, q=0.95):
        
        D_samples = []

        for _ in range(n_samples):
    
            D_sample = self.sample_forecasts(horizon=horizon)
    
            D_samples.append(np.diff(np.array(D_sample)))
    
        
        D_upper  = np.quantile(np.array(D_samples), q=q, axis=0)
        D_lower  = np.quantile(np.array(D_samples), q=1-q, axis=0)
        
        return D_upper, D_lower


