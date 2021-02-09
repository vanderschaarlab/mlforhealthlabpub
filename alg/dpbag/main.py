'''
2019 NeurIPS Submission
Title: Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate
Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Last Updated Date: May 28th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Main Function
- Load dataset
- Run DPBag (Our algorithm)
- Measure the performances

Inputs
- Raw data
- Parameters (epsilon, delta, teacher_no, part number)

Outputs
- AUROC
- AUPRC
- Accuracy
- Budget
'''

#%% Necessary Packages
import numpy as np
from tqdm import tqdm
import pandas as pd

#%% Functions
from DPBag_Final import DPBag

# 1. Models
from data_loading import Data_Loading_MAGGIC, Data_Loading_Adult

#%% Parameters

# Select dataset
data_sets = ['maggic','adult']
data_name = data_sets[1]

# Number of iterations
Iterations = 10

# Algorithm parameters
parameters = dict()

parameters['epsilon'] = 5
parameters['delta'] = 1e-3
parameters['teacher_no'] = 250
parameters['lamda'] = float(2)/parameters['teacher_no']
parameters['part_no'] = 100

#%% DPBag

# Output initialization
Output_AUC = list()
Output_APR = list()
Output_ACC = list()
Output_Budget = list()

# Iterate DPBag experiments

for itr in tqdm(range(Iterations)):
    
    # Load Data
    if data_name == 'maggic':
        x_train, y_train, x_valid, y_valid, x_test, y_test = Data_Loading_MAGGIC()
    elif data_name == 'adult':
        x_train, y_train, x_valid, y_valid, x_test, y_test = Data_Loading_Adult()
    
    print(data_name + ' Data Loaded')
    
    # DPBag Algorithm
    Temp_ACC, Temp_AUC, Temp_APR, Temp_Budget, _ = DPBag(x_train, y_train, x_valid, x_test, y_test, parameters)
    
    print('Finish DPBag Algorithm')
        
    # Gather performance metrics
    Output_ACC.append(Temp_ACC)
    Output_AUC.append(Temp_AUC)
    Output_APR.append(Temp_APR)
    Output_Budget.append(Temp_Budget)
        
#%% Performance Table
        
dict_metrics = {'Epsilon':[i+1 for i in range(len(Output_ACC[0]))],
                'Accuracy': np.mean(Output_ACC,0),
                'AUROC': np.mean(Output_AUC,0),
                'AUPRC': np.mean(Output_APR,0),
                'Budget': np.mean(Output_Budget,0)}

Output_Metric = pd.DataFrame(dict_metrics)

# Print Final Metric
print(Output_Metric)

