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
compute_identifiability.py
- Compare Identifiability between original data and synthetic data
"""

# Necessary packages
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

# Function start
def compute_identifiability (orig_data, synth_data):
  """Compare Wasserstein distance between original data and synthetic data.
  
  Args:
    orig_data: original data
    synth_data: synthetically generated data
      
  Returns:
    WD_value: Wasserstein distance
  """
  
  # Entropy computation
  def compute_entropy(labels):
    value,counts = np.unique(np.round(labels), return_counts=True)
    return entropy(counts)
  
  # Original data
  orig_data = np.asarray(orig_data)
        
  # Parameters
  no, x_dim = np.shape(orig_data)
    
  #%% Weights
  W = np.zeros([x_dim,])
    
  for i in range(x_dim):
    W[i] = compute_entropy(orig_data[:,i])
    
  # Normalization
  orig_data_hat = orig_data.copy()
  synth_data_hat = synth_data.copy()
    
  for i in range(x_dim):
    orig_data_hat[:,i] = orig_data[:,i] * 1./W[i]
    synth_data_hat[:,i] = synth_data[:,i] * 1./W[i]
    
  #%% r_i computation       
  nbrs = NearestNeighbors(n_neighbors = 2).fit(orig_data_hat)
  distance, _ = nbrs.kneighbors(orig_data_hat)

  # hat{r_i} computation
  nbrs_hat = NearestNeighbors(n_neighbors = 1).fit(synth_data_hat)
  distance_hat, _ = nbrs_hat.kneighbors(orig_data_hat)

  # See which one is bigger
  R_Diff = distance_hat[:,0] - distance[:,1]
  identifiability_value = np.sum(R_Diff<0) / float(no)
    
  return identifiability_value