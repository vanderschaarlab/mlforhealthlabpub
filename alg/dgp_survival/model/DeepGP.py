# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
import itertools
import torch

import pyro

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

from scipy.cluster.vq import kmeans2

import torch
import torch.nn as nn
from torch.distributions.transforms import AffineTransform
from torchvision import transforms

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.infer as infer
import pyro.infer.mcmc as mcmc
from pyro.contrib import autoname
from pyro.contrib.examples.util import get_data_loader

from tqdm import tqdm

pyro.set_rng_seed(0)


def output_layer(constant, y):
    
    return 1 - 1/(1 + np.exp(-1 * constant * y))



def sigmoid_calibrate_survival_predictions(model, y_input):
    
    min_const       = -100 
    max_const       = 100 
    num_contsts     = 10000 
    
    range_constants = np.linspace(min_const, max_const, num_contsts) 
    range_vals      = [np.abs(np.mean(output_layer(constant=const_, y=y_input)) - model.offset_probability) for const_ in range_constants]
    
    return range_constants[np.argmin(np.array(range_vals))]
    

    
class LinearT(nn.Module):
    """Linear transform and transpose"""
    def __init__(self, dim_in, dim_out):
        
        super(LinearT, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        return self.linear(x).t()


class DGPSurv(gp.parameterized.Parameterized):
    
    def __init__(self, 
                 X, 
                 T,
                 c,
                 prediction_horizon,
                 layer_dim=30, 
                 num_causes=1,
                 num_inducing=100,
                 calibration_fraction=0.5,
                 calibrate=False
                ):
        
        super(DGPSurv, self).__init__()
        
        self.prediction_horizon   = prediction_horizon
        self.calibrate            = calibrate
        
        # Refine inputs
        inclusion_criteria = (T >= self.prediction_horizon) | (c!=0)
        X_                 = np.array(X)[inclusion_criteria, :]
        c_                 = np.array(((np.array(c)[inclusion_criteria]!=0) & (np.array(T)[inclusion_criteria] < self.prediction_horizon)) * np.array(c)[inclusion_criteria]) 
        
        # Set all model attributes
        self.minmax_ = StandardScaler()

        self.X  = torch.tensor(np.array(self.minmax_.fit_transform(X_))).float() 
        self.T  = torch.tensor(np.array(T).astype(float) / 365).float()
        self.c  = torch.tensor(np.array(c_).astype(float)).float()
        
        self.num_inducing = min(num_inducing, self.X.shape[0])
        self.num_causes   = num_causes + 1
        self.num_dim      = self.X.shape[1]
        
        self.Xu = torch.from_numpy(kmeans2(self.X.numpy(), self.num_inducing, minit='points')[0])

        # handle erroneous settings for the model's parameters
        
        try:
            
            self.layer_dim = layer_dim 
            
            if self.layer_dim < 2:
                raise ValueError("Bad inputs: number of intermediate dimensions must be greater than 2.")

        except ValueError as ve:
            print(ve)
        
        # computes the weight for mean function of the first layer using a PCA transformation
        _, _, V = np.linalg.svd(self.X.numpy(), full_matrices=False)
        W = torch.from_numpy(V[:self.layer_dim, :])

        mean_fn                    = LinearT(self.num_dim, self.layer_dim)
        mean_fn.linear.weight.data = W
        mean_fn.linear.weight.requires_grad_(False)
        
        self.mean_fn = mean_fn
        

        # Initialize the first DGP layer
        
        linear         = torch.nn.Linear(self.num_dim, 20)
        pyro_linear_fn = lambda x: pyro.module("linear", linear)(x)
        kernel         = gp.kernels.Matern32(input_dim=self.num_dim, lengthscale=torch.tensor(1.))
        warped_kernel  = gp.kernels.Warping(kernel, pyro_linear_fn)
            
        self.layer_0 = gp.models.VariationalSparseGP(self.X, None,  
                                                     gp.kernels.Matern52(self.num_dim, 
                                                                    variance=torch.tensor(1.), 
                                                                    lengthscale=torch.ones(self.num_dim)), 
                                                     #warped_kernel,
                                                     Xu=self.Xu,
                                                     likelihood=None,
                                                     mean_function=self.mean_fn,
                                                     latent_shape=torch.Size([self.layer_dim]))
        
        h  = self.mean_fn(self.X).t()
        hu = self.mean_fn(self.Xu).t()

        self.layer_1 = gp.models.VariationalSparseGP(h, 
                                                     self.c,
                                                     gp.kernels.Matern52(self.layer_dim, 
                                                                    variance=torch.tensor(1.), 
                                                                    lengthscale=torch.tensor(1.)),
                                                     Xu=hu,
                                                     likelihood=gp.likelihoods.MultiClass(num_classes=self.num_causes),
                                                     latent_shape=torch.Size([self.num_causes]))                   

        #self.layer_0.u_scale_tril = self.layer_0.u_scale_tril * 1e-5 
        #self.layer_0.set_constraint("u_scale_tril", torch.distributions.constraints.lower_cholesky)
        
        if self.calibrate:
    
            self.kmf                  = KaplanMeierFitter()
        
            self.kmf.fit(T, event_observed=c)
        
            self.offset_probability   = self.kmf.survival_function_at_times(times=[self.prediction_horizon])._values[0]
            self.calibration_fraction = calibration_fraction
            
        
    @autoname.name_count
    def model(self, X, c):
                                  
        self.layer_0.set_data(X, None)
        
        h_loc, h_var = self.layer_0.model()
        h            = dist.Normal(h_loc, h_var.sqrt())()
        
        self.layer_1.set_data(h.t(), c)
        self.layer_1.model()

    @autoname.name_count
    def guide(self, X, c):
                                  
        self.layer_0.guide()
        self.layer_1.guide()

    # make prediction
    def forward(self, X_new):
        
        # because prediction is stochastic (due to Monte Carlo sample of hidden layer),
        # we make 100 prediction and take the most common one (as in [4])
        
        pred           = []
        num_MC_samples = 100
        
        for _ in range(num_MC_samples):
                                  
            h_loc, h_var = self.layer_0(X_new)
            h            = dist.Normal(h_loc, h_var.sqrt())()
            
            f_loc, f_var = self.layer_1(h.t())

            pred.append(f_loc) # change for multiclass

        return torch.stack(pred).mode(dim=0)[0]
    
    def train(self, num_epochs=5, num_iters=60, batch_size=1000, learning_rate=0.01):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn   = infer.TraceMeanField_ELBO().differentiable_loss 
        
        self.loss = []

        for i in range(num_epochs):
            
            self.loss.append(self.train_update(optimizer, loss_fn, batch_size, num_iters, i))
        
        
        self.loss = np.array(self.loss).reshape((-1, 1))
        
        
        if self.calibrate:
            
            print("Calibrating the trained model...")
        
            calibration_indexes       = np.random.choice(list(range(self.X.shape[0])), int(np.ceil(self.calibration_fraction * self.X.shape[0]))) 
            y_uncalibrated            = self.predict_survival(self.X[calibration_indexes, :].detach().numpy(), calibrate=1)
            y_raw                     = np.log((1 - y_uncalibrated) / y_uncalibrated)
        
            self.calibration_constant = sigmoid_calibrate_survival_predictions(self, y_raw)
        
            print("Done training!")
            
        else: 
            
            self.calibration_constant = 1
        
    def train_update(self, optimizer, loss_fn, batch_size, num_iters, epoch):
        
        losses = []
    
        for _ in range(num_iters):
                
            batch_indexes  = np.random.choice(list(range(self.X.shape[0])), batch_size)
            
            features_      = self.X[batch_indexes, :]
            event_censor   = self.c[batch_indexes]

            features_      = features_.reshape(-1, self.X.shape[1])
            
            optimizer.zero_grad()
        
            loss = loss_fn(self.model, self.guide, features_, event_censor)
            
            losses.append(loss)
            
            loss.backward()
            optimizer.step()

        print("Train Epoch: {:2d} \t[Iteration: {:2d}] \tLoss: {:.6f}".format(epoch, _, loss))

            
        return losses     
    

    def predict_survival(self, X_new, calibrate=None):
        
        s_preds        = []
        y_pred         = []
        
        index          = 0
        base_size      = 1000
        predictor_size = np.min((X_new.shape[0], base_size))
        num_batches_   = int(np.ceil(X_new.shape[0] / predictor_size))
        
        if calibrate==None:
            
            calibration_factor = self.calibration_constant
        
        else:
            
            calibration_factor = calibrate
            
        
        for u in range(num_batches_):
        
            if (u == (num_batches_ - 1)) and (np.mod(X_new.shape[0], predictor_size) > 0):
            
                X_curr  = np.array(X_new)[index: , :]
        
            else:
            
                X_curr  = np.array(X_new)[index: index + predictor_size, :]
    
            X_new_numpy = self.minmax_.transform(X_curr)
            X_new_      = torch.tensor(X_new_numpy).float()  
            
            f_output    = self(X_new_).detach().numpy()
            
            if u == 0:
                s_preds = f_output
            else:
                s_preds = np.hstack((s_preds, f_output))
              
            index += predictor_size
   
        
        for v in range(self.num_causes):
            
            y_pred.append(output_layer(constant=calibration_factor, y=s_preds[v, :]))
        
        
        y_pred = ((1 - np.array(y_pred)) / np.sum((1 - np.array(y_pred)), axis=0))

        return y_pred
