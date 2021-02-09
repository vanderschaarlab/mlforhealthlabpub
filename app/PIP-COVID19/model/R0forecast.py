
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np
from copy import deepcopy
import time

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from scipy.integrate import odeint

from utils.data_padding import *

torch.manual_seed(1) 


npi_vars       = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                  "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                  "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]

meta_features  = ["stats_population_density", "stats_median_age", "stats_gdp_per_capita", 
                  "stats_smoking", "stats_population_urban", "stats_population_school_age"]


def get_country_features(country_dict_input):
    
    if np.isnan(country_dict_input["Metadata"]["stats_population_school_age"]):
    
        country_dict_input["Metadata"]["stats_population_school_age"] = 15000000
    
    country_dict_input["Metadata"]["stats_population_urban"]      = country_dict_input["Metadata"]["stats_population_urban"]/country_dict_input["Metadata"]["stats_population"]
    country_dict_input["Metadata"]["stats_population_school_age"] = country_dict_input["Metadata"]["stats_population_school_age"]/country_dict_input["Metadata"]["stats_population"]
    
    X_whether = np.array(country_dict_input["wheather data"].fillna(method="ffill"))
    X_cases   = np.array(country_dict_input["Daily cases"] / country_dict_input["Metadata"]["stats_population"]).reshape((-1, 1))
    X_meta    = np.repeat(np.array([country_dict_input["Metadata"][meta_features[k]] for k in range(len(meta_features))]).reshape((1, -1)), X_whether.shape[0], axis=0)
    X_NPI     = np.array(country_dict_input["NPI data"][npi_vars].fillna(method="ffill"))
    X_moblty  = np.array(country_dict_input["Smoothened mobility data"])
    X_strngy  = np.array(country_dict_input["NPI data"]["npi_stringency_index"].fillna(method="ffill").values/100)
    
    return X_whether, X_meta, X_moblty, X_NPI, X_strngy


def get_beta(R0_t_pred, SEIR_model):
    
    beta_t_pred = R0_t_pred * (SEIR_model.gamma + (SEIR_model.p_CD/SEIR_model.T_CD)) * ((SEIR_model.p_CD/SEIR_model.T_CD) + SEIR_model.sigma) / SEIR_model.sigma
    
    return beta_t_pred


def get_R0(beta_t_pred, SEIR_model):
    
    R0_t_pred   = (beta_t_pred * SEIR_model.sigma) / ((SEIR_model.gamma + (SEIR_model.p_CD/SEIR_model.T_CD)) * ((SEIR_model.p_CD/SEIR_model.T_CD) + SEIR_model.sigma)) 
    
    return R0_t_pred


def compute_stringency_index(npi_policy):
    
    """
    
    npi_policy['npi_workplace_closing']              = 3 
    npi_policy['npi_school_closing']                 = 3
    npi_policy['npi_cancel_public_events']           = 2 
    npi_policy['npi_gatherings_restrictions']        = 4
    npi_policy['npi_close_public_transport']         = 2
    npi_policy['npi_stay_at_home']                   = 3
    npi_policy['npi_internal_movement_restrictions'] = 2
    npi_policy['npi_international_travel_controls']  = 4
    
    """
    
    w   = 0.29
    I_1 = (npi_policy["npi_workplace_closing"] * (1-w)/3 + w)
    I_2 = (npi_policy["npi_school_closing"] * (1-w)/3 + w)
    I_3 = (npi_policy["npi_cancel_public_events"] * (1-w)/2 + w)
    I_4 = (npi_policy["npi_gatherings_restrictions"] * (1-w)/4 + w)
    I_5 = (npi_policy["npi_close_public_transport"] * (1-w)/2 + w)
    I_6 = (npi_policy["npi_stay_at_home"] * (1-w)/3 + w)
    I_7 = (npi_policy["npi_internal_movement_restrictions"] * (1-w)/2 + w)
    I_8 = (npi_policy["npi_international_travel_controls"] * (1-w)/4 + w)
    I_9 = 1
    
    I   = (1/9) * (I_1 + I_2 + I_3 + I_4 + I_5 + I_6 + I_7 + I_8 + I_9) 
    
    return I
    
            
def model_loss_single(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.mean(torch.sum(single_loss, axis=0) / torch.sum(masks, axis=0)) 
    
    return loss

def single_losses(model):

    return model.masks * (model(model.X).view(-1, model.MAX_STEPS) - model.y)**2


def model_loss(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.sum(torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1)))
    
    return loss


def quantile_loss(output, target, masks, q):
    
    single_loss  = masks * ((output - target) * (output >= target)  * q + (target - output) * (output < target)  * (1-q))
    loss         = torch.sum(torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1))) 
    
    return loss    


class R0Forecaster(nn.Module):
    
    def __init__(self, 
                 mode="LSTM",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,  
                 INPUT_SIZE=30,     
                 LR=0.01,   
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50,
                 alpha=0.05,
                 beta_max=2,
                 country_parameters=None,
                 country_models=None):
        
        super(R0Forecaster, self).__init__()
        
        self.EPOCH          = EPOCH      
        self.BATCH_SIZE     = BATCH_SIZE
        self.MAX_STEPS      = MAX_STEPS  
        self.INPUT_SIZE     = INPUT_SIZE     
        self.LR             = LR   
        self.OUTPUT_SIZE    = OUTPUT_SIZE
        self.HIDDEN_UNITS   = HIDDEN_UNITS
        self.NUM_LAYERS     = NUM_LAYERS 
        self.N_STEPS        = N_STEPS
        self.q              = alpha
        self.mode           = mode
        self.country_params = country_parameters
        self.country_models = country_models

        rnn_dict = {"RNN" : nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "LSTM": nn.LSTM(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "GRU" : nn.GRU(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,)
                    }

        self.rnn            = rnn_dict[self.mode]
        self.out            = nn.Linear(self.HIDDEN_UNITS, 8) 
        self.out_q          = nn.Linear(self.HIDDEN_UNITS, 8) 
        self.out_w          = nn.Sequential(nn.Linear(21, self.HIDDEN_UNITS),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.HIDDEN_UNITS, self.HIDDEN_UNITS),
                                            nn.Tanh(),
                                            nn.Linear(self.HIDDEN_UNITS, 8))
        
        self.masks_w        = nn.Parameter(torch.rand(1))        
        self.masks_w_q      = nn.Parameter(torch.rand(1))
        #self.linear_model   = nn.Parameter(torch.rand(8))

        self.npi_normalizer = StandardScaler()
        self.model_mob_npi  = []
        

    def forward(self, x):
        
        REAL_ST = x.shape[1]
        #poly    = PolynomialFeatures(interaction_only=True)
        #X_numpy = x[:, :, 22:30].detach().numpy().reshape((-1, 8)) 
        #X_intrc = torch.Tensor(poly.fit_transform(X_numpy).reshape((-1, REAL_ST, 37)))
        X_intrc = x[:, :, 22:30]
        
        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x[:, :, :21], None)   # None represents zero initial hidden state

        else:

            r_out, h_n        = self.rnn(x[:, :, :21], None)

        # choose r_out at the last time step
 
        #w_    = torch.squeeze(F.sigmoid(self.out_w(r_out[:, :, :])), dim=2)
        #p_eff = torch.sum(torch.mul(-1 * torch.abs(self.linear_model), x[:, :, 22:30]))

        #out   = F.sigmoid((1 - w_) * x[:, :, -1] + w_ * torch.sum(-1 * torch.mul(torch.abs(self.out(r_out[:, :, :])), x[:, :, 22:30]), dim=2) - torch.abs(self.masks_w) * (1-x[:, :, 31]) * x[:, :, 30])
        #out   = F.sigmoid(torch.sum(-1 * torch.mul(torch.abs(self.out(r_out[:, :, :])), x[:, :, 22:30]), dim=2) - torch.abs(self.masks_w) * (1-x[:, :, 31]) * x[:, :, 30])
        
        #print(torch.abs(self.out_w(x[:, :, :21])))
        #print(X_intrc)
        
        out   = F.sigmoid(torch.sum(-1 * torch.mul(torch.abs(self.out_w(x[:, :, :21])), X_intrc), dim=2) - torch.abs(self.masks_w) * (1-x[:, :, 31]) * x[:, :, 30])
        out_q = F.sigmoid(torch.sum(-1 * torch.abs(self.out_q(r_out[:, :, :])) * x[:, :, 22:30], dim=2) - torch.abs(self.masks_w_q) * (1-x[:, :, 31]) * x[:, :, 30])
         
        return torch.squeeze(torch.stack([torch.unsqueeze(out, dim=2), torch.unsqueeze(out_q, dim=2)], dim=2), dim=3)
    
    
    def fit(self, X_whether, X_metas, X_mobility, X_NPIs, X_stringency, Y):
                
        self.model_mob_npi    = self.train_NPI_mobility_layers(X_NPIs, X_mobility, X_metas)
    
        country_names         = list(self.country_params.keys())
        self.beta_nromalizers = dict.fromkeys(list(self.country_params.keys()))
        self.beta_min         = dict.fromkeys(list(self.country_params.keys()))
        self.Y_latest_values  = dict.fromkeys(list(self.country_params.keys())) 
        
        Y_shifted             = [np.hstack((np.array([1]), Y[k][:len(Y[k])-1])) for k in range(len(Y))]
        
        for k in range(len(country_names)):
        
            self.Y_latest_values[country_names[k]] = Y_shifted[k]
        
        X_NPI_input           = [np.hstack((np.hstack((X_NPIs[k][:, :], X_stringency[k].reshape((-1, 1)))), Y_shifted[k].reshape((-1, 1)))) for k in range(len(X_NPIs))]
        
        X                     = [np.hstack((np.hstack((np.hstack((X_whether[k], X_metas[k])), X_mobility[k])), X_NPI_input[k])) for k in range(len(X_whether))]
        Y_                    = Y.copy() 
        
        for k in range(len(country_names)):

            self.beta_nromalizers[country_names[k]] = np.max(Y[k] - np.min(Y[k])) 
            self.beta_min[country_names[k]]         = np.min(Y[k])
            Y_[k]                                   = (Y[k] - np.min(Y[k])) / np.max(Y[k] - np.min(Y[k]))
        
        self.normalizer      = StandardScaler()

        self.normalizer.fit(np.array(X).reshape((-1, X[0].shape[1])))
        
        X                    = [self.normalizer.transform(X[k]) for k in range(len(X))]              
        
        for k in range(len(X)):
            
            X[k][:, 31]      = X_stringency[k]
        
        X_padded, _          = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(padd_arrays(Y_, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(padd_arrays(Y_, max_length=self.MAX_STEPS)[1], axis=2)
        
        X                    = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y_                   = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks           = Variable(torch.tensor(loss_masks), volatile=True).type(torch.FloatTensor)
        
        self.X               = X
        self.Y               = Y_
        self.masks           = loss_masks
        
        optimizer            = torch.optim.Adam(self.parameters(), lr=self.LR)   # optimize all rnn parameters
        self.loss_func       = quantile_loss 
        
        # training and testing
        for epoch in range(self.EPOCH):

            for step in range(self.N_STEPS):
                
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)
                
                x      = torch.tensor(X[batch_indexes, :, :])
                y      = torch.tensor(Y_[batch_indexes])
                msk    = torch.tensor(loss_masks[batch_indexes])
                
                b_x    = Variable(x[:, :, :].view(-1, self.MAX_STEPS, 33))       # self.INPUT_SIZE))   # reshape x to (batch, time_step, input_size)
                b_y    = Variable(y)                                             # batch y
                b_m    = Variable(msk)
                
                output = self(b_x).view(-1, self.MAX_STEPS, 2)                   # rnn output

                L_reg  = 0
                loss   = (1 - L_reg) * model_loss(output[:, :, 0], b_y, b_m) + L_reg * (self.loss_func(output[:, :, 0] + output[:, :, 1], b_y, b_m, self.q) + self.loss_func(output[:, :, 0] - output[:, :, 1], b_y, b_m, 1 - self.q)) 
                
                optimizer.zero_grad()                           # clear gradients for this training step
                loss.backward()                                 # backpropagation, compute gradients
                optimizer.step()                                # apply gradients

                if step % 50 == 0:

                    #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
                    print("Epoch: %d \t| \ttrain loss: %.4f" % (epoch, loss.data))
        
    
    def predict(self, X):
        
        stringency      = []   
        
        for k in range(len(X)):
            
            stringency.append(X[k][:, 31]) 
        
        X               = [self.normalizer.transform(X[k]) for k in range(len(X))] 
        
        for k in range(len(X)):
            
            X[k][:, 31] = stringency[k] 
        
        if type(X) is list:
            
            X_, masks   = padd_arrays(X, max_length=self.MAX_STEPS)
        
        else:
            
            X_, masks   = padd_arrays([X], max_length=self.MAX_STEPS)
        
        X_test          = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)
        
        predicts_      = self(X_test).view(-1, self.MAX_STEPS, 2) 
        prediction_0   = unpadd_arrays(predicts_[:, :, 0].detach().numpy(), masks)
        prediction_1   = unpadd_arrays(predicts_[:, :, 1].detach().numpy(), masks)
        
        return prediction_0, prediction_1
    
    
    def projection(self, days, npi_policy, country="United Kingdom"):
        
        # self.Y_shifted
    
        X_whether, X_metas, X_mobility, X_NPIs, X_stringency = get_country_features(self.country_params[country])
        
        X_NPI_input    = np.hstack((X_NPIs, X_stringency.reshape((-1, 1)))) 
        
        X              = [np.hstack((np.hstack((np.hstack((np.hstack((X_whether, X_metas)), X_mobility)), X_NPI_input)), self.Y_latest_values[country].reshape((-1, 1))))]
         
        X_NPI_new      = np.array(list(npi_policy.values()))
        X_NPI_new[-1]  = compute_stringency_index(npi_policy)
        
        X_features     = np.hstack((X_NPI_new[:8].reshape((1, -1)), X_metas[-1, :].reshape((1, -1))))
        X_features     = self.npi_normalizer.transform(X_features)
        
        X_mob_pred     = np.array([self.model_mob_npi[k].predict(X_features) for k in range(len(self.model_mob_npi))])
        
        X_new          = np.hstack((np.hstack((np.hstack((X_whether[-1, :], X_metas[-1, :])), X_mob_pred.reshape((-1,)))), X_NPI_new)) 
        X_new          = np.hstack((X_new, np.array([self.Y_latest_values[country][-1]]).reshape((-1,))))
        
        X_new[22-9:22] = X_mobility[-1, :] 
        
        X_forecast     = np.vstack((X[0], np.repeat(X_new.reshape((1, -1)), days, axis=0))) 
        
        self.X_input   = X_forecast
        
        beta_preds     = self.predict([X_forecast])[0][0] * self.beta_nromalizers[country] + self.beta_min[country]
        beta_pred_CI   = self.predict([X_forecast])[1][0] * self.beta_nromalizers[country] + self.beta_min[country]
        
        beta_pred_u    = beta_preds + beta_pred_CI
        beta_pred_l    = beta_preds - beta_pred_CI
        
        beta_pred_l    = beta_pred_l * (beta_pred_l >= 0) 

        R0_frc         = get_R0(beta_preds, self.country_models[country])
        
        #R0_frc[len(X[0]):] = np.mean(R0_frc[len(X[0]):])
        
        R0_frc_u       = R0_frc + 0.1 #get_R0(beta_pred_u, self.country_models[country])

        R0_frc_l       = R0_frc - 0.1 #get_R0(beta_pred_l, self.country_models[country])
 
        y_pred, _, _   = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc[len(X[0]):])
        y_pred_u, _, _ = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc_u[len(X[0]):])
        y_pred_l, _, _ = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc_l[len(X[0]):])
    
        return (y_pred, y_pred_u, y_pred_l), (R0_frc, R0_frc_u, R0_frc_l) 
    
    
    def train_NPI_mobility_layers(self, X_NPIs, X_mobility, X_metas):
    
        # add meta vars and normalize

        X_NPI_flat     = np.array(X_NPIs).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
        X_NPI_flat     = X_NPI_flat[:, :8] 
        X_mob_flat     = np.array(X_mobility).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
        X_met_flat     = np.array(X_metas).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
    
        X_features     = np.hstack((X_NPI_flat, X_met_flat))
    
        self.npi_normalizer.fit(X_features)
    
        #model_mob_npi  = [MLPRegressor(hidden_layer_sizes=(500, 500, )) for k in range(X_mob_flat.shape[1])] 
        #model_mob_npi  = [XGBRegressor(n_estimators=100, params={"monotone_constraints": str(tuple([-1] * X_features.shape[1]))}) for k in range(X_mob_flat.shape[1])]
        model_mob_npi  = [LinearRegression() for k in range(X_mob_flat.shape[1])]
        
        for k in range(X_mob_flat.shape[1]):
        
            model_mob_npi[k].fit(self.npi_normalizer.transform(X_features), X_mob_flat[:, k])
            
        return model_mob_npi