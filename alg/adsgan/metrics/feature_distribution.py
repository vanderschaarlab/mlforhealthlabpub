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
feature_distribution.py
- Compare feature distribution between original data and synthetic data
"""

# Import necessary packages
import numpy as np

def feature_distribution (orig_data, synth_data):
  """Compare feature distribution between orig data and synth data
  
  Args:
    orig_data: original data
    synth_data: synthetically generated data
    
  Returns:
    dist_comp_table: distribution comparison table
  """
  
  orig_data = np.asarray(orig_data)
  
  # Parameters
  no, dim = np.shape(orig_data)
    
  # Output initialization
  dist_comp_table = np.zeros([dim, 4])
    
  for i in range(dim):
            
    if len(np.unique(orig_data[:, i])) > 2:
      dist_comp_table[i,0] = np.mean(synth_data[:,i])
      dist_comp_table[i,1] = np.std(synth_data[:,i])
              
      dist_comp_table[i,2] = np.mean(orig_data[:,i])
      dist_comp_table[i,3] = np.std(orig_data[:,i])
      
    else:
      dist_comp_table[i,0] = np.sum(synth_data[:,i]==1)
      dist_comp_table[i,1] = np.sum(synth_data[:,i]==1) / float(no)
            
      dist_comp_table[i,2] = np.sum(orig_data[:,i]==1)
      dist_comp_table[i,3] = np.sum(orig_data[:,i]==1) / float(no)
            
  return dist_comp_table
            