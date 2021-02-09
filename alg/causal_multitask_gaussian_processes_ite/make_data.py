
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pandas as pd
import numpy as np


def draw_ihdp_data(fn_data):
    
    # Read the covariates and treatment assignments from the original study
    # ----------------------------------------------------------------------
    Raw_Data       = pd.read_csv(fn_data)
    X              = np.array(Raw_Data[['X5','X6','X7','X8','X9','X10',
                                        'X11','X12','X13','X14','X15',
                                        'X16','X17','X18','X19','X20',
                                        'X21','X22','X23','X24','X25',
                                        'X26','X27','X28','X29']])
    W              = np.array(Raw_Data['Treatment'])
    
    # Sample random coefficients
    # --------------------------
    coeffs_        = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB_c        = np.random.choice(coeffs_, size=6, replace=True, p=[0.5,0.125,0.125,0.125,0.125])
    BetaB_d        = np.random.choice(coeffs_, size=19, replace=True, p=[0.6, 0.1, 0.1, 0.1,0.1])
    BetaB          = np.hstack((BetaB_d,BetaB_c))

    # Simulating the two response surfaces 
    # ------------------------------------
    Y_0            = np.random.normal(size=len(X)) + np.exp(np.dot(X + 0.5, BetaB))
    Y_1            = np.random.normal(size=len(X)) + np.dot(X, BetaB)
    
    AVG            = np.mean(Y_1[W==1] - Y_0[W==1])
    Y_1            = Y_1 - AVG + 4
    
    TE             = np.dot(X, BetaB) - AVG + 4 - np.exp(np.dot(X + 0.5, BetaB))
    Y              = np.transpose(np.array([W, (1-W)*Y_0 + W*Y_1, TE]))

    # Prepare the output dataset 
    # --------------------------
    DatasetX       = pd.DataFrame(X,columns='X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'.split())
    DatasetY       = pd.DataFrame(Y,columns='Treatment Response TE'.split())
    Dataset        = DatasetX.join(DatasetY)
    
    Dataset['Y_0'] = Y_0
    Dataset['Y_1'] = Y_1
    
    return Dataset


def sample_IHDP(fn_data, test_frac=0.2):
    
    Dataset     = draw_ihdp_data(fn_data)
    
    num_samples = len(Dataset)
    train_size  = int(np.floor(num_samples * (1 - test_frac)))

    train_index = list(np.random.choice(range(num_samples), train_size, replace=False))
    test_index  = list(set(list(range(num_samples))) - set(train_index))
    
    feat_name   = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'
    
    Data_train  = Dataset.loc[Dataset.index[train_index]]
    Data_test   = Dataset.loc[Dataset.index[test_index]]
    
    X_train       = np.array(Data_train[feat_name.split()])
    W_train       = np.array(Data_train['Treatment'])
    Y_train       = np.array(Data_train['Response'])
    T_true_train  = np.array(Data_train['TE'])
    Y_cf_train    = np.array(Data_train['Treatment'] * Data_train['Y_0'] + (1- Data_train['Treatment']) * Data_train['Y_1'])

    Y_0_train     = np.array(Data_train['Y_0'])
    Y_1_train     = np.array(Data_train['Y_1'])
    
    X_test        = np.array(Data_test[feat_name.split()])
    W_test        = np.array(Data_test['Treatment'])
    Y_test        = np.array(Data_test['Response'])
    T_true_test   = np.array(Data_test['TE'])
    Y_cf_test     = np.array(Data_test['Treatment'] * Data_test['Y_0'] + (1- Data_test['Treatment']) * Data_test['Y_1'])

    Y_0_test      = np.array(Data_test['Y_0'])
    Y_1_test      = np.array(Data_test['Y_1']) 

    train_data    = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data     = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)
    
    return train_data, test_data 




