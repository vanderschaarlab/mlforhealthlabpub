'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Main Function
- Import Dataset
- Generate Synthetic Dataset
- Evaluate the performances in three ways
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score

Inputs
- Dataset
- Network Parameters

Outputs
- time-series synthetic data
- Performances
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score
'''

#%% Necessary Packages

import numpy as np
import sys

#%% Functions
# 1. Models
from tgan import tgan

# 2. Data Loading
from data_loading import google_data_loading, sine_data_generation

# 3. Metrics
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

#%% Main Parameters
# Data
data_set = ['google','sine']
data_name = data_set[1]

# Experiments iterations
Iteration = 2
Sub_Iteration = 10

#%% Data Loading
seq_length = 24

if data_name == 'google':
    dataX = google_data_loading(seq_length)
elif data_name == 'sine':
    No = 10000
    F_No = 5
    dataX = sine_data_generation(No, seq_length, F_No)

print(data_name + ' dataset is ready.')

#%% Newtork Parameters
parameters = dict()

parameters['hidden_dim'] = len(dataX[0][0,:]) * 4
parameters['num_layers'] = 3
parameters['iterations'] = 50000
parameters['batch_size'] = 128
parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:]) 

#%% Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()

# Each Iteration
for it in range(Iteration):

    # Synthetic Data Generation
    dataX_hat = tgan(dataX, parameters)   
      
    print('Finish Synthetic Data Generation')

    #%% Performance Metrics
    
    # 1. Discriminative Score
    Acc = list()
    for tt in range(Sub_Iteration):
        Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
        Acc.append(Temp_Disc)
    
    Discriminative_Score.append(np.mean(Acc))
    
    # 2. Predictive Performance
    MAE_All = list()
    for tt in range(Sub_Iteration):
        MAE_All.append(predictive_score_metrics (dataX, dataX_hat))
        
    Predictive_Score.append(np.mean(MAE_All))        
        
#%% 3. Visualization
PCA_Analysis (dataX, dataX_hat)
tSNE_Analysis (dataX, dataX_hat)

# Print Results
print('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
print('Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))
