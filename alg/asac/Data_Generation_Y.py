# -*- coding: utf-8 -*-

import numpy as np

#%% Corresponding Y generation
def Syn_Generation_Y1(X, W, sigma):
    
    # No of samples
    n = len(X) 
    t = len(X[0][:,0])
    
    # Initialization
    Output_Y = list()
    
    for i in range(n):
        Temp_X = X[i]
        Temp_Y = np.zeros([t,])
        for j in range(t):
            Temp_Y[j] =  np.exp(-np.abs(np.sum(W * Temp_X[j,:]))) + np.random.normal(loc = 0, scale = sigma)
            
        Output_Y.append(Temp_Y)    
    
    return Output_Y
  
  
  #%% Corresponding Y generation
def Syn_Generation_Y2(X, W, sigma):
    
    # No of samples
    n = len(X) 
    t = len(X[0][:,0])
    d = int(len(X[0][0,:])/2)
    
    # Initialization
    Output_Y = list()
    
    for i in range(n):
        Temp_X = X[i]
        Temp_Y = np.zeros([t,])
        for j in range(t):
            Temp_Y[j] =  np.exp(-np.abs(np.sum(W * Temp_X[j,:d]))) + np.random.normal(loc = 0, scale = sigma)
            
        Output_Y.append(Temp_Y)
    
    
    return Output_Y
  
#%% Corresponding Y generation
def Syn_Generation_Y3(X, W, sigma, eta):
    
    # No of samples
    n = len(X) 
    t = len(X[0][:,0])
    d = int(len(X[0][0,:])/2)
    
    # Initialization
    Output_Y = list()
    
    for i in range(n):
        Temp_X = X[i]
        Temp_Y = np.zeros([t,])
        for j in range(t):
            Temp_Y[j] =  np.exp(-np.abs(np.sum(W * Temp_X[j,:d]))) + np.random.normal(loc = 0, scale = sigma)
            
            
        Output_Y.append(Temp_Y)
    
    
    #%% Cost Generation
    
    Output_C = list()
    Output_G = list()
    
    for i in range(n):
        Temp = X[i].copy()
        Temp_Y = Output_Y[i]
        Temp_G = X[i].copy()
        
        for j in range(t):
            if (Temp_Y[j] < 0.5):
                Temp[j,:] = np.asarray([1,1,1,1,1,1,1,1,1,1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
                Temp_G[j,:] = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            else:
                Temp[j,:] = eta * np.asarray([1,1,1,1,1,1,1,1,1,1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
                Temp_G[j,:] = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
                
        Output_C.append(Temp)
        Output_G.append(Temp_G)
    
    return Output_Y, Output_C, Output_G
  
  
  
  
  
  
  
  
