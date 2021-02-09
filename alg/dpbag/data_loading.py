'''
2019 NeurIPS Submission
Title: Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate
Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Last Updated Date: May 28th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
- Load two real-world data (MAGGIC and UCI Adults)
- Transform the raw data to preprocessed data

Inputs
- Raw data

Outputs
- trainX, trainY
- validX, validY
- testX, testY
'''

#%% Necessary packages
import pandas as pd
import numpy as np

#%% MAGGIC data (Label: 3-year mortality)

def Data_Loading_MAGGIC():
    
    # Load Data from Data folder
    File_name = 'Data/Maggic.csv'
    Data = pd.read_csv(File_name, sep=',')
    
    # Remover any NA
    Data = Data.dropna(axis=0, how='any')
    
    # Set label                      
    year = 3                      
    Time_hor = 365*year

    # Exclude the patients who are censored before 3 year
    Data = Data[(Data['days_to_fu']>Time_hor) | ((Data['days_to_fu']<=Time_hor) & (Data['death_all']==1))]
    
    # Label generation
    N  = Data.shape[0]
    Data['Label']=[0]*N
    # Set label = 1 if the patient were dead within 3 year
    Data.loc[((Data['days_to_fu']<=Time_hor) & (Data['death_all']==1)), 'Label'] =1   
    
    # Divide feature (X) and label (Y)
    X = Data.drop(['death_all', 'days_to_fu', 'Label'], axis=1)
    Y = Data['Label']
            
    # Treat X and Y as numpy array
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Normalization with Minmax Scaler
    for i in range(len(X[0,:])):
        
        X[:,i] = X[:,i] - np.min(X[:,i])
        X[:,i] = X[:,i] / (np.max(X[:,i]) + 1e-8)
    
    # Divide the data into train, valid, and test set (1/3 each)
    idx = np.random.permutation(len(Y))
    
    # train
    trainX = X[idx[:int(len(Y)/3)],:]
    trainY = Y[idx[:int(len(Y)/3)]]    
    
    # valid
    validX = X[idx[int(len(Y)/3):(2*int(len(Y)/3))],:]
    validY = Y[idx[int(len(Y)/3):(2*int(len(Y)/3))]]
    
    # test
    testX = X[idx[(2*int(len(Y)/3)):],:]
    testY = Y[idx[(2*int(len(Y)/3)):]]

    # Return train, valid, and test sets
    return trainX, trainY, validX, validY, testX, testY


#%% UCI Adult data (Predict income >50k)
# Reference link: https://yanhan.github.io/posts/2017-02-15-analysis-of-the-adult-data-set-from-uci-machine-learning-repository.ipynb.html
 
def Data_Loading_Adult():
  
    # Data loading
    Data1 = pd.read_csv('data/adult.data', header = None, delimiter = ",")
    Data2 = pd.read_csv('data/adult.test', header = None, delimiter = ",")
  
    # Merge two datasets
    df = pd.concat((Data1, Data2), axis = 0)
  
    # Define column names
    df.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]
  
    # Index reset due to merging
    df = df.reset_index()
    df = df.drop("index", axis = 1)
  
    # Label define
    Y = np.ones([len(df),])
    
    # Set >50K as 1 and <=50K as 0
    Y[df["Income"].index[df["Income"] == " <=50K"]] = 0
    Y[df["Income"].index[df["Income"] == " <=50K."]] = 0
    
    # Drop feature which can directly infer label
    df.drop("Income", axis=1, inplace=True,)
  
    # Transform the type from string to float
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)
    df.CapitalGain = df.CapitalGain.astype(float)
    df.CapitalLoss = df.CapitalLoss.astype(float)
  
    # One hot incoding for categorical features
    df = pd.get_dummies(df, columns=["WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
    "Race", "Gender", "NativeCountry"])
    
    # Treat data as numpy array
    X = np.asarray(df)
        
    # Normalization with Minmax Scaler
    for i in range(len(X[0,:])):
        
        X[:,i] = X[:,i] - np.min(X[:,i])
        X[:,i] = X[:,i] / (np.max(X[:,i]) + 1e-8)
    
    # Divide the data into train, valid, and test set (1/3 each)
    idx = np.random.permutation(len(Y))
    
    # train
    trainX = X[idx[:int(len(Y)/3)],:]
    trainY = Y[idx[:int(len(Y)/3)]]    
    
    # valid
    validX = X[idx[int(len(Y)/3):(2*int(len(Y)/3))],:]
    validY = Y[idx[int(len(Y)/3):(2*int(len(Y)/3))]]
    
    # test
    testX = X[idx[(2*int(len(Y)/3)):],:]
    testY = Y[idx[(2*int(len(Y)/3)):]]

    # Return train, valid, and test sets
    return trainX, trainY, validX, validY, testX, testY
  
  
