## Jobs data import (Jinsung Yoon, 10/11/2017)

import numpy as np
from scipy.special import expit
import argparse
import pandas as pd
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab

'''
Input: train_rate: 0.8
Outputs:
  - Train_X, Test_X: Train and Test features
  - Train_Y: Observable outcomes
  - Train_T: Assigned treatment
  - Opt_Train_Y, Test_Y: Potential outcomes.
'''

def Data_Twins(fn_csv, train_rate = 0.8):
  
    #%% Data Input (11400 patients, 30 features, 2 potential outcomes)
    Data = np.loadtxt(fn_csv, delimiter=",", skiprows=1)

    # Features
    X = Data[:,:30]

    # Feature dimensions and patient numbers
    Dim = len(X[0])
    No = len(X)
    
    # Labels
    Opt_Y = Data[:,30:]
    
    for i in range(2):
      idx = np.where(Opt_Y[:,i] > 365)
      Opt_Y[idx,i] = 365
    
    Opt_Y = 1-(Opt_Y / 365.0)
            
    #%% Patient Treatment Assignment
    coef = 0* np.random.uniform(-0.01, 0.01, size = [Dim,1])
    Temp =  expit(np.matmul(X,coef) + np.random.normal(0,0.01, size = [No,1]) )
    
    Temp = Temp/(2*np.mean(Temp))
    
    Temp[Temp>1] = 1
    
    T = np.random.binomial(1,Temp,[No,1])
    T = T.reshape([No,])

    #%% Observable outcomes
    Y = np.zeros([No,1])

    # Output
    Y = np.transpose(T) * Opt_Y[:,1] + np.transpose(1-T) * Opt_Y[:,0]    
    Y = np.transpose(Y)
    Y = np.reshape(Y,[No,])

    #%% Train / Test Division
    temp = np.random.permutation(No)
    Train_No = int(train_rate*No)
    train_idx = temp[:Train_No]
    test_idx = temp[Train_No:No]
    
    Train_X = X[train_idx,:]
    Train_T = T[train_idx]
    Train_Y = Y[train_idx]
    Opt_Train_Y = Opt_Y[train_idx,:]

    Test_X = X[test_idx,:]
    Test_Y = Opt_Y[test_idx,:]
        
    return [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y]


def Data_Jobs(fn_csv):
    # Data Input

    Data = np.loadtxt(fn_csv, delimiter=",", skiprows=1)

    # Feature, Treatment and Output 
    X = Data[:,:7]

    T = Data[:,7]

    Raw_Y = Data[:,8] 
    Y = np.array(Raw_Y>0,dtype=float)
    
    # Parameters
    No = len(X)
    RCT_No = 723
    Train_Rate = 0.8

    # RCT / NoRCT
    RCT_X = X[:RCT_No,:]
    RCT_T = T[:RCT_No]
    RCT_Y = Y[:RCT_No]

    NoRCT_X = X[RCT_No:,:]
    NoRCT_T = T[RCT_No:]
    NoRCT_Y = Y[RCT_No:]

    # Train / Test Division
    temp = np.random.permutation(RCT_No)
    Train_No = int(Train_Rate*RCT_No)
    Test_No = RCT_No - Train_No
    train_idx = temp[:Train_No]
    test_idx = temp[Train_No:]
    
    Train_X_Test = RCT_X[train_idx,:]
    Train_T_Test = RCT_T[train_idx]
    Train_Y_Test = RCT_Y[train_idx]

    Train_Test_No = len(train_idx)

    Test_X = RCT_X[test_idx,:]
    Test_T = RCT_T[test_idx]
    Test_Y = 1- (1*(RCT_Y[test_idx] == 0))

    Train_X = np.concatenate([Train_X_Test,NoRCT_X])
    Train_T = np.concatenate([Train_T_Test,NoRCT_T])
    Train_Y = np.concatenate([Train_Y_Test,NoRCT_Y])

    Train_No = No - Test_No
    
    return [Train_X, Train_T, Train_Y, Test_X, Test_T, Test_Y, Train_X_Test, Train_T_Test, Train_Y_Test, Train_No, Test_No, Train_Test_No]


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="twins")
    parser.add_argument("--trainx", default="trainx.csv")
    parser.add_argument("--trainy", default="trainy.csv")
    parser.add_argument("--traint", default="traint.csv")
    parser.add_argument("--testx", default="testx.csv")
    parser.add_argument("--testy", default="testy.csv")
    parser.add_argument("--testt", default="testt.csv")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()
    dataset = args.dataset
    fn_trainx, fn_trainy, fn_traint = args.trainx, args.trainy, args.traint
    fn_testx, fn_testy, fn_testt = args.testx, args.testy, args.testt

    Test_T = None
    if dataset == 'twins':
        train_rate = 0.8
        fn_twins_csv = utilmlab.get_data_dir() + "/twins/Twin_Data.csv.gz"
        [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] \
            = Data_Twins(fn_twins_csv, train_rate)
    elif dataset == 'jobs':
        fn_jobs_csv = utilmlab.get_data_dir() + "/jobs/Jobs_Lalonde_Data.csv.gz"
        [Train_X, Train_T, Train_Y, Test_X, Test_T, Test_Y, Train_X_Test,
         Train_T_Test, Train_Y_Test, Train_No, Test_No, Train_Test_No] \
         = Data_Jobs(fn_jobs_csv)
    else:
        assert 0

    pd.DataFrame(Train_X).to_csv(fn_trainx, index=False)
    pd.DataFrame(Train_Y).to_csv(fn_trainy, index=False)
    pd.DataFrame(Train_T).to_csv(fn_traint, index=False)
    pd.DataFrame(Test_X).to_csv(fn_testx, index=False)
    pd.DataFrame(Test_Y).to_csv(fn_testy, index=False)
    if Test_T is not None:
        pd.DataFrame(Test_T).to_csv(fn_testt, index=False)
