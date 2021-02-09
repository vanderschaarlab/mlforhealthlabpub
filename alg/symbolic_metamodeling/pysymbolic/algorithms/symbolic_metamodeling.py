

# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
from scipy.special import digamma, gamma
import itertools
import copy

from mpmath import *
from sympy import *
#from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from pysymbolic.models.special_functions import *

from tqdm import tqdm, trange, tqdm_notebook, tnrange

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sympy import Integral, Symbol
from sympy.abc import x, y

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def is_ipython():
    
    try:
        
        __IPYTHON__
        
        return True
    
    except NameError:
        
        return False


def basis(a, b, c, x, hyper_order=[1, 2, 2, 2]):
    
    epsilon = 0.001
    
    func_   = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=3)
    
    return func_.evaluate(x + epsilon)

def basis_expression(a, b, c, hyper_order=[1, 2, 2, 2]):
    
    func_ = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=3)
    
    return func_
    

def basis_grad(a, b, c, x, hyper_order=[1, 2, 2, 2]):
    
    K1     = sc.special.digamma(a - b + 1)
    K2     = sc.special.digamma(a - b + 2)
    K3     = sc.special.digamma(a - b + 3)
    K4     = sc.special.digamma(a - b + 4)
    
    G1     = sc.special.gamma(a - b + 1)
    G2     = sc.special.gamma(a - b + 2)
    G3     = sc.special.gamma(a - b + 3)
    G4     = sc.special.gamma(a - b + 4)
    
    nema1  = 6 * ((c * x)**3) * (K4 - np.log(c * x))
    nema2  = 2 * ((c * x)**2) * (-K3 + np.log(c * x))
    nema3  = (c * x) * (K2 - np.log(c * x))
    nema4  = -1 * (K1 - np.log(c * x))
    
    nemb1  = -1 * 6 * ((c * x)**3) * K4 
    nemb2  = 2 * ((c * x)**2) * K3 
    nemb3  = -1 * (c * x) * K2 
    nemb4  = K1 
    
    nemc1  = -1 * (c**2) * (x**3) * (6 * a + 18)
    nemc2  = (c * (x**2)) * (4 + 2 * a) 
    nemc3  = -1 * x * (1 + a)
    nemc4  = a / c
    
    grad_a = ((c * x) ** a) * (nema1/G4 + nema2/G3 + nema3/G2 + nema4/G1) 
    grad_b = ((c * x) ** a) * (nemb1/G4 + nemb2/G3 + nemb3/G2 + nemb4/G1) 
    grad_c = ((c * x) ** a) * (nemc1/G4 + nemc2/G3 + nemc3/G2 + nemc4/G1) 

    return grad_a, grad_b, grad_c



def tune_single_dim(lr, n_iter, x, y, verbosity=False):
    
    epsilon   = 0.001
    x         = x + epsilon
    
    a         = 2
    b         = 1
    c         = 1
    
    batch_size  = np.min((x.shape[0], 500)) 
    
    for u in range(n_iter):
        
        batch_index = np.random.choice(list(range(x.shape[0])), size=batch_size)
        
        new_grads   = basis_grad(a, b, c, x[batch_index])
        func_true   = basis(a, b, c, x[batch_index])
        
        loss        =  np.mean((func_true - y[batch_index])**2)
        
        if verbosity:
        
            print("Iteration: %d \t--- Loss: %.3f" % (u, loss))
        
        grads_a   = np.mean(2 * new_grads[0] * (func_true - y[batch_index]))
        grads_b   = np.mean(2 * new_grads[1] * (func_true - y[batch_index]))
        grads_c   = np.mean(2 * new_grads[2] * (func_true - y[batch_index]))
        
        a         = a - lr * grads_a
        b         = b - lr * grads_b
        c         = c - lr * grads_c
        
    return a, b, c 


def compose_features(params, X):
    
    X_out = [basis(a=float(params[k, 0]), b=float(params[k, 1]), c=float(params[k, 2]), 
                   x=X[:, k], hyper_order=[1, 2, 2, 2]) for k in range(X.shape[1])] 
    
    return np.array(X_out).T
    

class symbolic_metamodel:
    
    def __init__(self, model, X, mode="classification"):
        
        self.feature_expander = PolynomialFeatures(2, include_bias=False, interaction_only=True)
        self.X                = X
        self.X_new            = self.feature_expander.fit_transform(X) 
        self.X_names          = self.feature_expander.get_feature_names()
        
        if mode == "classification": 
        
            self.Y                = model.predict_proba(self.X)[:, 1]
            self.Y_r              = np.log(self.Y/(1 - self.Y))
            
        else:
            
            self.Y_r              = model.predict(self.X)
        
        
        self.num_basis        = self.X_new.shape[1]
        self.params_per_basis = 3
        self.total_params     = self.num_basis * self.params_per_basis + 1
        
        a_init                = 1.393628702223735 
        b_init                = 1.020550117939659
        c_init                = 1.491820813243337
        
        self.params           = np.tile(np.array([a_init, b_init, c_init]), [self.num_basis, 1])
        
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            
            self.tqdm_mode = tqdm_notebook
            
        else:
            
            self.tqdm_mode = tqdm
            
    
    def set_equation(self, reset_init_model=False):
         
        self.X_init           = compose_features(self.params, self.X_new)
        
        if reset_init_model:
            
            self.init_model   = Ridge(alpha=.1, fit_intercept=False, normalize=True) #LinearRegression
            
            self.init_model.fit(self.X_init, self.Y_r)
    
    def get_gradients(self, Y_true, Y_metamodel, batch_index=None):
        
        param_grads = self.params * 0
        epsilon     = 0.001 
        
        for k in range(self.params.shape[0]):
            
            a                 = float(self.params[k, 0])
            b                 = float(self.params[k, 1])
            c                 = float(self.params[k, 2])
            
            if batch_index is None:
                grads_vals    = basis_grad(a, b, c, self.X_new[:, k] +  epsilon)
            else:
                grads_vals    = basis_grad(a, b, c, self.X_new[batch_index, k] +  epsilon)
            
            param_grads[k, :] = np.array(self.loss_grads(Y_true, Y_metamodel, grads_vals))
        
        return param_grads
        
    
    def loss(self, Y_true, Y_metamodel):
        
        return np.mean((Y_true - Y_metamodel)**2)
    
    def loss_grads(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_a = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[0])
        loss_grad_b = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[1])
        loss_grad_c = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[2])
        
        return loss_grad_a, loss_grad_b, loss_grad_c 
    
    def loss_grad_coeff(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_ = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x)
        
        return loss_grad_
        
    
    def fit(self, num_iter=10, batch_size=100, learning_rate=.01):
        
        print("---- Tuning the basis functions ----")
        
        for u in self.tqdm_mode(range(self.X.shape[1])):
            
            self.params[u, :] = tune_single_dim(lr=0.1, n_iter=500, x=self.X_new[:, u], y=self.Y_r)
            
        self.set_equation(reset_init_model=True)

        self.metamodel_loss = []
        
        print("----  Optimizing the metamodel  ----")
        
        for _ in self.tqdm_mode(range(num_iter)):
            
            batch_index = np.random.choice(list(range(self.X_new.shape[0])), size=batch_size)
            
            curr_func   = self.init_model.predict(self.X_init[batch_index, :])
            
            self.metamodel_loss.append(self.loss(self.Y_r[batch_index], curr_func))
            
            param_grads  = self.get_gradients(self.Y_r[batch_index], curr_func, batch_index)
            self.params  = self.params - learning_rate * param_grads
            
            coef_grads            = [self.loss_grad_coeff(self.Y_r[batch_index], curr_func, self.X_init[batch_index, k]) for k in range(self.X_init.shape[1])]
            self.init_model.coef_ = self.init_model.coef_ - learning_rate * np.array(coef_grads)
             
            self.set_equation()  
            
            self.exact_expression, self.approx_expression = self.symbolic_expression()
            
    def evaluate(self, X):
        
        X_modified  = self.feature_expander.fit_transform(X)
        X_modified_ = compose_features(self.params, X_modified)
        Y_pred_r    = self.init_model.predict(X_modified_)
        Y_pred      = 1 / (1 + np.exp(-1 * Y_pred_r))
        
        return Y_pred 
    
    def symbolic_expression(self):
    
        dims_ = []

        for u in range(self.num_basis):

            new_symb = self.X_names[u].split(" ")

            if len(new_symb) > 1:
    
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
                S2 = Symbol(new_symb[1].replace("x", "X"), real=True)
        
                dims_.append(S1 * S2)
    
            else:
        
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
    
                dims_.append(S1)
        
        self.dim_symbols = dims_
        
        sym_exact   = 0
        sym_approx  = 0
        x           = symbols('x')

        for v in range(self.num_basis):
    
            f_curr      = basis_expression(a=float(self.params[v,0]), 
                                           b=float(self.params[v,1]), 
                                           c=float(self.params[v,2]))
        
            sym_exact  += sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))).subs(x, dims_[v])
            sym_approx += sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))).subs(x, dims_[v])    
        
        return 1/(1 + exp(-1*sym_exact)), 1/(1 + exp(-1*sym_approx))   
    
    
    def get_gradient_expression(self):
        
        diff_dims  = self.dim_symbols[:self.X.shape[1]]
        gradients_ = [diff(self.approx_expression, diff_dims[k]) for k in range(len(diff_dims))]

        diff_dims  = [str(diff_dims[k]) for k in range(len(diff_dims))]
        evaluator  = [lambdify(diff_dims, gradients_[k], modules=['math']) for k in range(len(gradients_))]
    
        return gradients_, diff_dims, evaluator
    

    def _gradient(self, gradient_expressions, diff_dims, evaluator, x_in):
    
        Dict_syms  = dict.fromkeys(diff_dims)

        for u in range(len(diff_dims)):

            Dict_syms[diff_dims[u]] = x_in[u]
         
        grad_out  = [np.abs(evaluator[k](**Dict_syms)) for k in range(len(evaluator))]
    
        
        return np.array(grad_out)    
    
    
    def get_instancewise_scores(self, X_in):
    
        gr_exp, diff_dims, evaluator = self.get_gradient_expression()
    
        gards_ = [self._gradient(gr_exp, diff_dims, evaluator, X_in[k, :]) for k in range(X_in.shape[0])]
    
        return gards_
    
        
