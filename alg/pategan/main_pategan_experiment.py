"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees," 
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
Last updated Date: Feburuary 15th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------
main_pategan_experiment.py
- Main function for PATEGAN framework
(1) pategan_main: main function for PATEGAN
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from data_generator import data_generator
from utils import supervised_model_training
from pate_gan import pategan
from sklearn.preprocessing import MinMaxScaler

#%% 
def pategan_main (args):
  """PATEGAN Main function.
  
  Args:
    data_no: number of generated data
    data_dim: number of data dimensions
    noise_rate: noise ratio on data
    iterations: number of iterations for handling initialization randomness
    n_s: the number of student training iterations
    batch_size: the number of batch size for training student and generator
    k: the number of teachers
    epsilon, delta: Differential privacy parameters
    lamda: noise size
    
  Returns:
    - results: performances of Original and Synthetic performances
    - train_data: original data
    - synth_train_data: synthetically generated data
  """
  
  # Supervised model types
  models = ['logisticregression','randomforest', 'gaussiannb','bernoullinb',
            'svmlin', 'Extra Trees','LDA', 'AdaBoost','Bagging','gbm', 'xgb']
  
  # Data generation
  if args.dataset == 'random':
      train_data, test_data = data_generator(args.data_no, args.data_dim, 
                                         args.noise_rate)
      data_dim = args.data_dim
  elif args.dataset == 'credit':
      # Insert relevant dataset here, and scale between 0 and 1.
      data = pd.read_csv('creditcard.csv').to_numpy()
      data = MinMaxScaler().fit_transform(data)
      train_ratio = 0.5
      train = np.random.rand(data.shape[0])<train_ratio 
      train_data, test_data = data[train], data[~train]
      data_dim = data.shape[1]
  
  # Define outputs
  results = np.zeros([len(models), 4])
  
  # Define PATEGAN parameters
  parameters = {'n_s': args.n_s, 'batch_size': args.batch_size, 'k': args.k, 
                'epsilon': args.epsilon, 'delta': args.delta, 
                'lamda': args.lamda}
  
  # Generate synthetic training data
  best_perf = 0.0
  
  for it in range(args.iterations):
    print('Iteration',it)
    synth_train_data_temp = pategan(train_data, parameters)
    temp_perf, _ = supervised_model_training(
        synth_train_data_temp[:, :(data_dim-1)], 
        np.round(synth_train_data_temp[:, (data_dim-1)]),
        train_data[:, :(data_dim-1)], 
        np.round(train_data[:, (data_dim-1)]),
        'logisticregression')
    
    # Select best synthetic data
    if temp_perf > best_perf:
      best_perf = temp_perf.copy()
      synth_train_data = synth_train_data_temp.copy()
      
    print('Iteration: ' + str(it+1))
    print('Best-Perf:' + str(best_perf))
  
  # Train supervised models
  for model_index in range(len(models)):
    model_name = models[model_index]
    
    # Using original data
    results[model_index, 0], results[model_index, 2] = (
        supervised_model_training(train_data[:, :(data_dim-1)], 
                                  np.round(train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))
        
    # Using synthetic data
    results[model_index, 1], results[model_index, 3] = (
        supervised_model_training(synth_train_data[:, :(data_dim-1)], 
                                  np.round(synth_train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))

    
    
  # Print the results for each iteration
  results = pd.DataFrame(np.round(results, 4), 
                         columns=['AUC-Original', 'AUC-Synthetic', 
                                  'APR-Original', 'APR-Synthetic'])
  print(results)
  print('Averages:')
  print(results.mean(axis=0))
  
  return results, train_data, synth_train_data

  
#%%  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_no',
      help='number of generated data',
      default=10000,
      type=int)
  parser.add_argument(
      '--data_dim',
      help='number of dimensions of generated dimension (if random)',
      default=10,
      type=int)
  parser.add_argument(
      '--dataset',
      help='dataset to use',
      default='random',
      type=str)
  parser.add_argument(
      '--noise_rate',
      help='noise ratio on data',
      default=1.0,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of iterations for handling initialization randomness',
      default=50,
      type=int)
  parser.add_argument(
      '--n_s',
      help='the number of student training iterations',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of batch size for training student and generator',
      default=64,
      type=int)
  parser.add_argument(
      '--k',
      help='the number of teachers',
      default=10,
      type=float)
  parser.add_argument(
      '--epsilon',
      help='Differential privacy parameters (epsilon)',
      default=1.0,
      type=float)
  parser.add_argument(
      '--delta',
      help='Differential privacy parameters (delta)',
      default=0.00001,
      type=float)
  parser.add_argument(
      '--lamda',
      help='PATE noise size',
      default=1.0,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  results, ori_data, synth_data = pategan_main(args)