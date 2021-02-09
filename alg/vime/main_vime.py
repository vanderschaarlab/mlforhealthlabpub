"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

main_vime.py
- Main function for VIME framework
(1) supervised_model_training: Train supervised model
(2) vime_main: main function for VIME
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
  
from data_loader import load_mnist_data
from supervised_models import logit, xgb_model, mlp

from vime_self import vime_self
from vime_semi import vime_semi
from vime_utils import perf_metric

#%%
def supervised_model_training (x_train, y_train, x_test, 
                               y_test, model_name, metric):
  """Train supervised learning models and report the results.
  
  Args:
    - x_train, y_train: training dataset
    - x_test, y_test: testing dataset
    - model_name: logit, xgboost, or mlp
    - metric: acc or auc
    
  Returns:
    - performance: prediction performance
  """
  
  # Train supervised model
  # Logistic regression
  if model_name == 'logit':
    y_test_hat = logit(x_train, y_train, x_test)
  # XGBoost
  elif model_name == 'xgboost':
    y_test_hat = xgb_model(x_train, y_train, x_test)      
  # MLP
  elif model_name == 'mlp':    
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100
      
    y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
    
  # Report the performance
  performance = perf_metric(metric, y_test, y_test_hat)    
    
  return performance    

#%% 
def vime_main (label_data_rate, model_sets, label_no, p_m, alpha, K, beta):
  """VIME Main function.
  
  Args:
    - model_sets: supervised model sets
    - label_no: number of labeled data to be used
    - p_m: corruption probability
    - alpha: hyper-parameter to control two self-supervied loss
    - K: number of augmented data
    - beta: hyper-parameter to control two semi-supervied loss
    
  Returns:
    - results: performances of supervised, VIME-self and VIME-semi performance
  """
  
  # Define outputs
  results = np.zeros([len(model_sets)+2])
  
  # Load data
  x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)
    
  # Use subset of labeled data
  x_train = x_train[:label_no, :]
  y_train = y_train[:label_no, :]  
  
  # Metric
  metric = 'acc'
    
  # Train supervised models
  for m_it in range(len(model_sets)):
    model_name = model_sets[m_it]
    results[m_it] = supervised_model_training (x_train, y_train, x_test, 
                                               y_test, model_name, metric)
  
  # Train VIME-Self
  vime_self_parameters = dict()
  vime_self_parameters['batch_size'] = 128
  vime_self_parameters['epochs'] = 10
  vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)
    
  # Save encoder
  if not os.path.exists('save_model'):
    os.makedirs('save_model')
  
  file_name = './save_model/encoder_model.h5'
  vime_self_encoder.save(file_name)  
  
  # Test VIME-Self
  x_train_hat = vime_self_encoder.predict(x_train)
  x_test_hat = vime_self_encoder.predict(x_test)
  
  model_name = 'mlp'
  results[len(model_sets)] = supervised_model_training (x_train_hat, y_train, 
                                                        x_test_hat, y_test,
                                                        model_name, 
                                                        metric)
  
  # Train VIME-Semi
  vime_semi_parameters = dict()
  vime_semi_parameters['hidden_dim'] = 100
  vime_semi_parameters['batch_size'] = 128
  vime_semi_parameters['iterations'] = 1000
  y_test_hat = vime_semi(x_train, y_train, x_unlab, x_test, 
                         vime_semi_parameters, p_m, K, beta, file_name)
  
  # Test VIME-Semi
  results[len(model_sets)+1] = perf_metric(metric, y_test, y_test_hat)

  # Print the results for each iteration
  print(np.round(results, 4))
  
  return results


def exp_main(args):
  """Main function for experiments.
  
  Args:
    - iterations: Number of experiments iterations
    - label_no: Number of labeled data to be used
    - model_name: supervised model name (mlp, logit, or xgboost)
    - p_m: corruption probability for self-supervised learning
    - alpha: hyper-parameter to control the weights of feature and mask losses
    - K: number of augmented samples
    - beta: hyperparameter to control supervised and unsupervised loss
    - label_data_rate: ratio of labeled data
      
  Returns:
    - results: performances of 3 different models (supervised only, VIME-self, and VIME)
  """
  
  # Define output
  results = np.zeros([args.iterations, 3])  
  
  # Iterations
  for it in range(args.iterations):
    results[it, :] = vime_main(args.label_data_rate, 
                               [args.model_name], 
                               args.label_no, 
                               args.p_m, 
                               args.alpha, 
                               args.K, 
                               args.beta)
  
  #%% Print results    
  print('Supervised Performance, Model Name: ' + args.model_name + 
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 0]), 4)) + 
        ', Std Perf: ' + str(np.round(np.std(results[:, 0]), 4)))
    
  print('VIME-Self Performance' +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 1]), 4)) + 
        ', Std Perf: ' + str(np.round(np.std(results[:, 1]), 4)))
  
  print('VIME Performance' +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 2]), 4)) + 
        ', Std Perf: ' + str(np.round(np.std(results[:, 2]), 4)))
  
  
#%%  
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--iterations',
      help='number of experiments iterations',
      default=10,
      type=int)
  parser.add_argument(
      '--model_name',
      choices=['logit','xgboost','mlp'],
      default='xgboost',
      type=str)
  parser.add_argument(
      '--label_no',
      help='number of labeled data to be used',
      default=1000,
      type=int)
  parser.add_argument(
      '--p_m',
      help='corruption probability for self-supervised learning',
      default=0.3,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyper-parameter to control the weights of feature and mask losses',
      default=2.0,
      type=float)
  parser.add_argument(
      '--K',
      help='number of augmented samples',
      default=3,
      type=int)
  parser.add_argument(
      '--beta',
      help='hyperparameter to control supervised and unsupervised loss',
      default=1.0,
      type=float)
  parser.add_argument(
      '--label_data_rate',
      help='ratio of labeled data',
      default=0.1,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  results = exp_main(args)