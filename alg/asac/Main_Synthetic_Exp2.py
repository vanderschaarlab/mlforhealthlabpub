'''
ASAC (Active Sensing using Actor-Critic Model) (12/18/2018)
Experiment 2 on Synthetic Data (Cheaper But Noisier Features) 
Jinsung Yoon
'''

#%% Necessary Packages
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

#%% Function Call
# X, Y Generation Functions
'''
X: Generated from AR_Gauss
Y: Generated from SYN1, SYN2, SYN3
'''
from Data_Generation_X import AR_Gauss_X2
from Data_Generation_Y import Syn_Generation_Y2

# ASAC
from ASAC import ASAC

# Predictor
from Predictor_G import Predictor_G

#%% Parameters
# n: No of samples, 
# d: No of dimensions, 
# t: No of time stamps, 
# Iteration_No: Experiment iterations

def main_exp2(cost_coef, gamma, Iteration_No):
  
    # Given Parameters
    n = 10000
    d = 10
    t = 9
    
    # Phi and W for X Generation    
    phi = np.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    W = np.asarray([0.1,0.2,0.3,0.4,0.0,0.0,0.0,0.0,0.0,0.0])
    
    # X, Y Variance
    sigma_X = 1
    sigma_Y = 0.1       
    
    # Cost
    cost = np.asarray([10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0]) + \
           cost_coef * np.asarray([0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10])
        
    # Output Initialization
    Output = np.zeros([Iteration_No,(2*d+1)])
    
    #%% Iterations
    for it in tqdm(range(Iteration_No)):
        
        # Data Generation
        trainX = AR_Gauss_X2 (n, d, t, phi, sigma_X, gamma)            
        testX = AR_Gauss_X2 (n, d, t, phi, sigma_X, gamma) 
        
        trainY = Syn_Generation_Y2(trainX, W, sigma_Y)
        testY = Syn_Generation_Y2(testX, W, sigma_Y)             
                  
        #%% ASAC
        '''
        Inputs: trainX, trainY, testX, cost
        Outputs: selected features in training and testing set
        '''
        trainG_hat, testG_hat = ASAC(trainX, trainY, testX, cost) 
        testY_hat = Predictor_G(trainX, testX, trainY, trainG_hat, testG_hat)  
            
        #%% Performance Metrics
            
        Temp_TPR  = np.zeros([2*d,])
        Temp_RMSE = 0        
            
        for i in range(n):
            for mm in range(2*d):
                
                Temp_TPR[mm] = Temp_TPR[mm] + np.sum(testG_hat[i][:,mm])
                    
            Temp_RMSE = Temp_RMSE + (mean_squared_error(testY[i],testY_hat[i]))
            
        for mm in range(2*d):
            Output[it,mm]  = Temp_TPR[mm]  / float(n*t)
                
        Output[it,2*d] = np.sqrt(Temp_RMSE / float(n))
        
    return Output    
              
#%% 
if __name__ == '__main__':
  
    # Parameter Setting
    cost_coef = 0.2
    gamma = 0.6
    Iteration_No = 1
    
    # Run Main Function
    Output = main_exp2(cost_coef, gamma, Iteration_No)
    
    # Print Results
    print('Results with Cost '+str(cost_coef) + ' and Gamma ' + str(gamma) + ' :')
    print('Selected Ratio for X: ')
    print(np.round(np.mean(Output[:,:4], axis = 0),3))
    print('Selected Ratio for X hat: ')
    print(np.round(np.mean(Output[:,10:14], axis = 0),3))
    
    
    