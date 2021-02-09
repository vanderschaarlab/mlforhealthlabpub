# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
import itertools

from mpmath import *
from sympy import *

from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pysymbolic.models.special_functions import MeijerG
from pysymbolic.utilities.performance import compute_Rsquared

from pysymbolic.benchmarks.synthetic_datasets import *
from pysymbolic.utilities.instancewise_metrics import *
from pysymbolic.algorithms.keras_predictive_models import *
from pysymbolic.algorithms.symbolic_expressions import *
from pysymbolic.algorithms.symbolic_metamodeling import *

from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from sklearn.neural_network import MLPClassifier

from lime.lime_tabular import LimeTabularExplainer
import shap

import tensorflow as tf
from collections import defaultdict
import re 
import sys
import os
import time
from keras.callbacks import ModelCheckpoint    
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer 
import json
import random
from keras import optimizers


def Symbolic_reg_expr(X, y):

    est_gp   = SymbolicClassifier(parsimony_coefficient=.01, random_state=1)
    est_gp   = SymbolicRegressor(population_size=5000,
                                 generations=20, stopping_criteria=0.01,
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.9, verbose=0,
                                 parsimony_coefficient=0.01, random_state=0) 

    est_gp.fit(X, y)

    sym_expr = str(est_gp._program)
    
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9  = symbols('X0 X1 X2 X3 X4 X5 X6 X7 X8 X9')

    converter = {
        'sub': lambda x, y : x - y,
        'div': lambda x, y : x/y,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'neg': lambda x    : -x,
        'pow': lambda x, y : x**y
    }

    sym_reg  = simplify(sympify(sym_expr, locals=converter))
    sym_reg  = sym_reg.subs((X0, X1, X2, X3, X4, X5, X6, X7, X8, X9), (X0, X1, X2, X3, X4, X5, X6, X7, X8, X9))

    
    vars_        = [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9 ]  
    gradients_   = [] 

    for var in vars_:
        
        gradients_.append(diff(sym_reg, var))

    return sym_reg, gradients_

def get_symbolic_reg_scores(model, x_train, x_test):

    y_pred_quantized    = model.predict(x_train)
    sym_reg, gradients_ = Symbolic_reg_expr(x_train, y_pred_quantized)

    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9  = symbols('X0 X1 X2 X3 X4 X5 X6 X7 X8 X9')
    vars_               = [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9]
    scores_             = [] 

    for k in range(x_test.shape[0]):

        scores_.append([gradients_[v].subs(tuple(vars_), tuple(list(x_test[k,:]))) for v in range(len(gradients_))])


    return np.array(scores_)


def get_lime_scores(predictive_model, x_train, x_test):

    lime_scores = []
    explainer   = LimeTabularExplainer(x_train, feature_names=['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9'])

    for w in range(x_test.shape[0]):
        exp        = explainer.explain_instance(x_test[w], predictive_model.predict_proba, num_features=10)
        rank_list  = exp.as_list()
    
        curr_scores                           = [np.where(np.array([pd.Series(rank_list[v][0]).str.contains('X'+str(k))[0]*1 for k in range(10)])==1)[0][0] for v in range(len(rank_list))]
        lime_score_                           = np.zeros((1, x_train.shape[1]))
        lime_score_[0, np.array(curr_scores)] = np.array([np.abs(rank_list[v][1]) for v in range(len(rank_list))])

        lime_scores.append(lime_score_)

    lime_scores = np.array(lime_scores).reshape(-1, x_train.shape[1]) 
    
    return lime_scores


def get_shap_scores(predictive_model, x_train, x_test):

	explainer    = shap.KernelExplainer(predictive_model.predict_proba, x_train, link="logit")
	shap_values  = explainer.shap_values(x_test, nsamples=100)
	shap_scores_ = np.array(shap_values)[0,:,:]

	return shap_scores_


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 
    """
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random_uniform(shape =(batch_size, self.k, d), 
            minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval = 1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.sigmoid(noisy_logits)
        samples = K.max(samples, axis = 1) 

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        
        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape 

"""
The code for L2X is adapted from: 
https://github.com/Jianbo-Lab/L2X/blob/master/synthetic/explain.py

"""

def L2X(datatype, activation, num_samples, tot_num_features, num_selected_features, out_activation='sigmoid', 
        loss_='binary_crossentropy', optimizer_='adam', num_hidden=200, num_layers=2, train = True): 
    
    BATCH_SIZE  = num_samples
    x_train, y_train, x_val, y_val, datatype_val = create_data(datatype, n = num_samples)
    input_shape = tot_num_features
     
    activation = 'relu' if datatype in ['orange_skin','XOR'] else 'selu'
    
    # P(S|X)
    model_input = Input(shape=(input_shape,), dtype='float32') 

    net = Dense(num_hidden, activation=activation, name = 's/dense1', 
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    
    
    for _ in range(num_layers-1):
        
        net = Dense(num_hidden, activation=activation, name = 's/dense'+str(_+2), 
                    kernel_regularizer=regularizers.l2(1e-3))(net) 

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = Dense(input_shape)(net) 
    
    # [BATCH_SIZE, max_sents, 1]  
    k = num_selected_features; tau = 0.1
    samples = Sample_Concrete(tau, k, name = 'sample')(logits)

    # q(X_S)
    new_model_input = Multiply()([model_input, samples]) 
    net = Dense(num_hidden, activation=activation, name = 'dense1', 
                kernel_regularizer=regularizers.l2(1e-3))(new_model_input) 
    
    net = BatchNormalization()(net) # Add batchnorm for stability.
    net = Dense(num_hidden, activation=activation, name = 'dense2', 
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)

    preds = Dense(1, activation=out_activation, name = 'dense4', 
                  kernel_regularizer=regularizers.l2(1e-3))(net) 
    model = Model(model_input, preds)

    if train: 
        adam = optimizers.Adam(lr = 1e-3)
        model.compile(loss=loss_, 
                      optimizer=optimizer_, 
                      metrics=['acc']) 
        filepath="L2X.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                                     verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, verbose=0, callbacks = callbacks_list, epochs=50, batch_size=BATCH_SIZE) #validation_data=(x_val, y_val)
         
    else:
        model.load_weights('L2X.hdf5', by_name=True) 


    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, 
                       optimizer='rmsprop',
                       metrics=None)  #metrics=[None]) 
    
    scores = pred_model.predict(x_val, verbose = 0, batch_size = BATCH_SIZE) 

    median_ranks = compute_median_rank(scores, k = num_selected_features, datatype_val=datatype_val)

    return median_ranks

def get_instancewise_median_ranks(dataset_, num_samples, num_selected_features, method_):
    
    model_types_ = {'SR': "keras",
                    'LIME':"modified_keras", 
                    'SHAP':"sklearn",
                    'DeepLIFT':"keras",
                    'SM':"sklearn"} 
    
    evaluators_  = {'SR': SR_instancewise,
                    'LIME': LIME_instancewise, 
                    'SHAP': SHAP_instancewise,
                    'DeepLIFT': DeepLIFT_instancewise,
                    'SM': Metamodel_instancewise} 
    
    if method_ != 'L2X':
        
        model_type   = model_types_[method_]
    
        eval_method  = evaluators_[method_]
    
    
        x_train, y_train, x_test, y_test, datatypes_val = create_data(dataset_, n = num_samples)
    
        eval_scores  = eval_method(x_train, y_train, x_test, model_type=model_type) 
        eval_ranks   = compute_median_rank(eval_scores, k=num_selected_features, datatype_val = datatypes_val)
        
    else:   
        
        eval_ranks   = L2X(datatype=dataset_, activation='relu', num_samples=1000, tot_num_features=10, 
                           num_selected_features=num_selected_features, out_activation='sigmoid', 
                           loss_='binary_crossentropy', optimizer_='adam', num_hidden=200, num_layers=2, train = True)
  
    return eval_ranks
    
def SR_instancewise(x_train, y_train, x_test,  model_type):
    
    predictive_model   = get_predictive_model(x_train, y_train, model_type=model_type)
    sym_reg_scores_    = get_symbolic_reg_scores(predictive_model, x_train, x_test)
    
    return sym_reg_scores_

    
def LIME_instancewise(x_train, y_train, x_test,  model_type):
    
    predictive_model   = get_predictive_model(x_train, y_train, model_type=model_type)
    lime_scores        = get_lime_scores(predictive_model, x_train, x_test)
    
    return lime_scores

def SHAP_instancewise(x_train, y_train, x_test,  model_type):
    
    predictive_model   = get_predictive_model(x_train, y_train, model_type=model_type)
    shap_scores        = get_shap_scores(predictive_model, x_train, x_test)
    
    return shap_scores
 
def DeepLIFT_instancewise(x_train, y_train, x_test,  model_type): 
    
    predictive_model   = get_predictive_model(x_train, y_train, model_type=model_type)
    background         = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    deepLIFT_explainer = shap.DeepExplainer(predictive_model, background)
    deepLIFT_scores    = deepLIFT_explainer.shap_values(x_test)[0]
    
    return deepLIFT_scores


def Metamodel_instancewise(x_train, y_train, x_test,  model_type): 
    
    predictive_model   = get_predictive_model(x_train, y_train, model_type=model_type)

    metamodel          = symbolic_metamodel(predictive_model, x_train)

    metamodel.fit(num_iter=10, batch_size=x_train.shape[0], learning_rate=.01)

    metamodel_scores   = np.array(metamodel.get_instancewise_scores(x_test)).reshape((-1, x_train.shape[1]))

    return metamodel_scores 



