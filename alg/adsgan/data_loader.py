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
data_loader.py
- data loading function for ADSGAN framework
(1) Load data and return pandas dataframe
"""

# Import necessary packages
import pandas as pd

def load_maggic_data():
  """Load MAGGIC data.
  
  Returns:
    orig_data: Original data in pandas dataframe
  """
  # Read csv files
  file_name = 'data/Maggic.csv'
  orig_data = pd.read_csv(file_name, sep=',')

  # Remove NA
  orig_data = orig_data.dropna(axis=0, how='any')
        
  # Remove labels
  orig_data = orig_data.drop(['death_all','days_to_fu'], axis = 1)
  
  return orig_data