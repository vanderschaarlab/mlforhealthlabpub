'''
ASAC (Active Sensing using Actor-Critic Model) (12/18/2018)
Experiment 1 on Synthetic Data (Cost vs Selection) 
'''

#%% Necessary Packages
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import argparse
import json

#%% Function Call
# X, Y Generation
'''
X: Generated from AR_Gauss
Y: Generated from SYN1
'''
from Data_Generation_X import AR_Gauss_X1
from Data_Generation_Y import Syn_Generation_Y1

# ASAC
from ASAC import ASAC

# Predictor after selection
from Predictor_G import Predictor_G

#%% Parameters
# n: No of samples, 
# d: No of dimensions, 
# t: No of time stamps, 
# Iteration_No: Experiment iterations
# cost_coef: cost of each features


def main_exp1(cost_coef, Iteration_No, niter, learning_rate):
  
    # Parameters
    n = 10000
    d = 10
    t = 9
    
    # Phi and W for X Generation    
    phi = np.asarray([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    W = np.asarray([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    
    # X, Y Variance
    sigma_X = 1
    sigma_Y = 0.1       
    
    # Given Cost
    cost = cost_coef * np.asarray([1,1,1,1,1,1,1,1,1,1])
        
    # Output Initialization
    Output = np.zeros([Iteration_No,(d+1)])
    print(Iteration_No)

    #%% Iterations
    for it in tqdm(range(Iteration_No)):
        
        # Data Generation
        trainX = AR_Gauss_X1 (n, d, t, phi, sigma_X)            
        testX = AR_Gauss_X1 (n, d, t, phi, sigma_X) 
        
        trainY = Syn_Generation_Y1(trainX, W, sigma_Y)
        testY = Syn_Generation_Y1(testX, W, sigma_Y)
                  
        #%% ASAC
        '''
        Inputs: trainX, trainY, testX, cost
        Outputs: selected features in training and testing set
        '''
        trainG_hat, testG_hat = ASAC(trainX, trainY, testX, cost, niter) 
        
        # Prediction after selection
        testY_hat = Predictor_G(trainX, testX, trainY, trainG_hat, testG_hat, niter)   
            
        #%% Performance Metrics     
        
        Temp_TPR  = np.zeros([d,])
        Temp_RMSE = 0        
            
        for i in range(n):
            
            for mm in range(d):
                
                Temp_TPR[mm] = Temp_TPR[mm] + np.sum(testG_hat[i][:,mm])
                    
            Temp_RMSE = Temp_RMSE + (mean_squared_error(testY[i],testY_hat[i]))
            
        for mm in range(d):
            Output[it,mm]  = Temp_TPR[mm]  / float(n*t)
                
        Output[it,d] = np.sqrt(Temp_RMSE / float(n))  
      
    return Output


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numexp", default=10, type=int)
    parser.add_argument("--it", default=5001, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()
    cost_coef_lst = [2, 4, 6, 8, 10]
    Iteration_No = args.numexp
    niter = args.it
    learning_rate = args.lr
    result_lst = []
    for cost_coef in cost_coef_lst:

        # Run Main Function
        Output = main_exp1(cost_coef, Iteration_No, niter, learning_rate)

        # Print Results
        selected_ratio = np.round(np.mean(Output[:, :10], axis=0), 3)
        print('Results with # experiments {} Cost {} : Selected Ratio: {}'.format(
            Iteration_No, cost_coef, selected_ratio))
        rmse = np.round(np.mean(Output[:, 10], axis=0), 3)
        print('RMSE: {:0.3f}'.format(rmse))
        result_lst.append((cost_coef, selected_ratio, rmse))
    print('final results:')
    for cost_coef, selected_ratio, rmse in result_lst:
        print('coef {:0.3f} ratio:{} rmse:{:.3f} {}'.format(
            cost_coef, cost_coef/cost_coef_lst[0], rmse, selected_ratio))
