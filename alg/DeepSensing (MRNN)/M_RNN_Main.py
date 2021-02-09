'''
Jinsung Yoon (03/08/2019)
M-RNN Main
'''
#%% Packages
import numpy as np

#%% Functions
# 1. Data Preprocessing
'''
If the original data is complete, use Complete, 
'''
from Data_Loader_ICU import Data_Loader_ICU
from Data_Loader_MIMIC import Data_Loader_MIMIC

# 2. Imputation Block
from M_RNN import M_RNN
import argparse

#%% Parameters
# train Parameters
train_rate = 0.8
missing_rate = 0.2

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ICU')
    return parser.parse_args()


args = init_arg()

# Mode
# data_sets = ['ICU', 'MIMIC']
# data_mode = data_sets[0]

data_mode = args.dataset

#%% Main

# 1. Data Preprocessing (Add missing values)
'''
X: Original Feature
Z: Feature with Missing
M: Missing Matrix
T: Time Gap
'''
if data_mode == 'ICU': 
    _, trainZ, trainM, trainT, testX, testZ, testM, testT = Data_Loader_ICU(train_rate, missing_rate)
elif data_mode == 'MIMIC': 
    _, trainZ, trainM, trainT, testX, testZ, testM, testT = Data_Loader_MIMIC(train_rate, missing_rate)
    
# 2. M_RNN_Imputation (Recovery)
_, Recover_testX = M_RNN(trainZ, trainM, trainT, testZ, testM, testT)
    
# 3. Imputation Performance Evaluation
MSE = np.sum ( np.abs ( testX * (1-testM) - Recover_testX * (1-testM) ) )  / np.sum(1-testM) 

print('Data: ' + data_mode + ', Mean Square Error: ' + str(MSE))    
    
