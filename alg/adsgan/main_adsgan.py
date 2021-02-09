"""Anonymization through Data Synthesis using Generative Adversarial Networks:
A harmonizing advancement for AI in medicine (ADS-GAN) Codebase.

Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar, 
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine," 
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
Last updated Date: December 22th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
main_adsgan.py
- Main function for ADSGAN framework
(1) Load data
(2) Generate synthetic data
(3) Measure the quality and identifiability of generated synthetic data
"""

#%% Import necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

#%% Import necessary functions
from data_loader import load_maggic_data
from adsgan import adsgan
from metrics.feature_distribution import feature_distribution
from metrics.compute_wd import compute_wd
from metrics.compute_identifiability import compute_identifiability


#%% Experiment main function
def exp_main(args):
  
  # Data loading
  orig_data = load_maggic_data()
  print("Finish data loading")
  
  # Generate synthetic data
  params = dict()
  params["lamda"] = args.lamda
  params["iterations"] = args.iterations
  params["h_dim"] = args.h_dim
  params["z_dim"] = args.z_dim
  params["mb_size"] = args.mb_size
  
  synth_data = adsgan(orig_data, params)
  print("Finish synthetic data generation")
  
  ## Performance measures
  # (1) Feature distributions
  feat_dist = feature_distribution(orig_data, synth_data)
  print("Finish computing feature distributions")
  
  # (2) Wasserstein Distance (WD)
  print("Start computing Wasserstein Distance")
  wd_measure = compute_wd(orig_data, synth_data, params)
  print("WD measure: " + str(wd_measure))
    
  # (3) Identifiability 
  identifiability = compute_identifiability(orig_data, synth_data)
  print("Identifiability measure: " + str(identifiability))

  return orig_data, synth_data, [feat_dist, wd_measure, identifiability]
  
#%%  
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--iterations',
      help='number of adsgan training iterations',
      default=10000,
      type=int)
  parser.add_argument(
      '--h_dim',
      help='number of hidden state dimensions',
      default=30,
      type=int)
  parser.add_argument(
      '--z_dim',
      help='number of random state dimensions',
      default=10,
      type=int)
  parser.add_argument(
      '--mb_size',
      help='number of mini-batch samples',
      default=128,
      type=int)
  parser.add_argument(
      '--lamda',
      help='hyper-parameter to control the identifiability and quality',
      default=0.1,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  orig_data, synth_data, measures = exp_main(args)