'''
Y|X Generation function
Jinsung Yoon (9/27/2018)
'''

# Necessary packages
import numpy as np

#%% Corresponding Y|X generation with Logit
def Logit_Generation_Y(X, amp, num):    
        
    # No of samples
    n = len(X[:,0])  
    # No of dimensions
    d = len(X[0,:])
    
    # Initialization
    Output_Y = np.zeros([n,])
    
    # Logit generation
    # Weight
    Weight_Temp = ( 2 * ( np.random.uniform(-1,1,d) > 0 ) - 1 ) * amp     
    
    # Sample relevant features with (num) number 
    index = np.random.permutation(d)[:num]
    
    # Only put some weights on the selected features
    Weight = np.zeros([d,])
    Weight[index] = Weight_Temp[index]    
    
    # Compute Logit
    logit_temp = np.matmul(X, Weight)
    logit = np.exp(logit_temp) / (1 + np.exp(logit_temp) )  

    # Sample logit (Ber(logit))
    Temp = np.random.uniform(0,1,n)
    Output_Y = (Temp <= logit)*1    

    # Ground truth
    G = np.zeros([d,])
    G[index] = 1      
    
    return Output_Y, G


#%% Corresponding Y|X generation with Gaussian
def Gauss_Generation_Y(X, amp, num, sigma):
    
    # No of samples
    n = len(X[:,0])    
    d = len(X[0,:])
    
    # Initialization
    Output_Y = np.zeros([n,])
    
    # M Generation
    # Weight
    Weight_Temp = ( 2 * ( np.random.uniform(-1,1,d) > 0 ) - 1 ) * amp  
    
    # Sample relevant features with (num) number 
    index = np.random.permutation(d)[:num]
    
    # Only put some weights on the selected features
    Weight = np.zeros([d,])
    Weight[index] = Weight_Temp[index]    
    
    # Compute M
    M = np.matmul(X, Weight)   
    
    # Sample Y
    Output_Y = np.random.normal(M, sigma) 

    # Ground truth
    G = np.zeros([d,])
    G[index] = 1     
    
    return Output_Y, G
    