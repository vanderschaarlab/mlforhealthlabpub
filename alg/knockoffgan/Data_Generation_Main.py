'''
Data Geneation for KnockoffGAN Experiments 
Jinsung Yoon (9/27/2018)
'''

# Necessary packages
import numpy as np
from tqdm import tqdm

# Function calls
# X Distribution Generation
from Data_Generation_X import Normal_Generation_X, AR_Normal_Generation_X, Uniform_Generation_X, AR_Uniform_Generation_X

# Y Distribution Generation
from Data_Generation_Y import Logit_Generation_Y, Gauss_Generation_Y
import argparse
import os
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', default='.', help='output directory')
    parser.add_argument(
        '--xname', default='Normal', help='')
    parser.add_argument(
        '--yname', default='Logit', help='')
    return parser.parse_args()


args = init_arg()

odir = args.o


# Parameters
n = 3000
p = 1000

Replication = 100

#%% Settings
# 1. X Distribution
x_set = ['Normal','AR_Normal','Uniform','AR_Uniform']
x_name = x_set[0]
x_name = args.xname

# 2. Y|X Distribution
y_set = ['Logit','Gauss']
y_name = y_set[0]
y_name = args.yname

# Print Generating distribution 
print ('X: ' + x_name + ', Y: ' + y_name)

#%% Coefficients
if (x_name == 'AR_Normal') | (x_name == 'AR_Uniform'):
    Coef = np.asarray([0.0,0.1,0.2,0.3,0.4,0.6,0.8])
elif (x_name == 'Uniform') & (y_name == 'Logit'):
    Coef = np.asarray([1.0,2.0,3.0,4.0,5.0,6.0,7.0])
elif (x_name == 'Uniform') & (y_name == 'Gauss'):
    Coef = np.asarray([0.5,1.0,1.5,2.0,2.5,3.0,3.5])
elif (y_name == 'Logit'):
    Coef = np.asarray([5.0,6.0,7.0,8.0,9.0,10.0,11.0])
elif (y_name == 'Gauss'):
    Coef = np.asarray([2.0,2.5,3.0,3.5,4.0,4.5,5.0])
        
#%% Iterations
for it in tqdm(range(Replication)):

    # For each Coefficient
    for i in range(len(Coef)):

        # Set coefficient
        amp_coef = Coef[i]

        #%% 1. Normal     
        if x_name == 'Normal':
            
            # n: number of samples, p: dimensions, sigma = variance
            X = Normal_Generation_X (n, p, sigma = 1./n)
             
            # A. Normal-Logit
            if y_name == 'Logit':
                
                # Y: Output, G: Ground truth relevant features, amp: Logit amplitude, num: # of relevant features
                Y, G = Logit_Generation_Y(X, amp = amp_coef, num = 60)            
            
            # B. Normal-Gauss
            elif y_name == 'Gauss':
                # sigma is the noise variance of the output
                Y, G = Gauss_Generation_Y(X, amp = amp_coef, num = 60, sigma = 1)          

        #%% 2. AR_Normal
        elif x_name == 'AR_Normal':
            
            X = AR_Normal_Generation_X (n, p, coef = amp_coef, margin = 1./n)
    
            # A. AR_Normal-Logit
            if y_name == 'Logit':
                # amp is set to 10
                Y, G = Logit_Generation_Y(X, amp = 10, num = 60)            
    
            # B. AR_Normal-Gauss
            elif y_name == 'Gauss':
                # amp is set to 3.5
                Y, G = Gauss_Generation_Y(X, amp = 3.5, num = 60, sigma = 1)            
                   
            
        #%% 3. Uniform 
        elif x_name == 'Uniform':
            
            X = Uniform_Generation_X (n, p, sigma = 1./n)
    
            # A. Uniform-Logit
            if y_name == 'Logit':
                
                Y, G = Logit_Generation_Y(X, amp = amp_coef, num = 60)            
            # B. Uniform-Gauss
            elif y_name == 'Gauss':
                
                Y, G = Gauss_Generation_Y(X, amp = amp_coef, num = 60, sigma = 1)  
                
        #%% 4. AR_Uniform
        elif x_name == 'AR_Uniform':
            
            X = AR_Uniform_Generation_X (n, p, coef = amp_coef, margin = 1./n)
            
            # A. AR_Uniform-Logit
            if y_name == 'Logit':
                # amp is set to 5
                Y, G = Logit_Generation_Y(X, amp = 5, num = 60)            
            # B. AR_Uniform-Gauss
            elif y_name == 'Gauss':
                # amp is set to 2.5
                Y, G = Gauss_Generation_Y(X, amp = 2.5, num = 60, sigma = 1)    
                
                
                
        #%% Save files
        # file_name_X = '/home/vdslab/Documents/Jinsung/2019_Research/ICLR/KnockGAN/BitBucket_Final/Data/' + x_name + '_' + y_name +'/X_' + str(amp_coef) + '_' + str(it) + '.csv'

        file_name_X = odir + '/Data/' + x_name + '_' + y_name +'/X_' + str(amp_coef) + '_' + str(it) + '.csv'
        file_name_Y = odir + '/Data/' + x_name + '_' + y_name +'/Y_' + str(amp_coef) + '_' + str(it) + '.csv'
        file_name_G = odir + '/Data/' + x_name + '_' + y_name +'/G_' + str(amp_coef) + '_' + str(it) + '.csv'

        for el in [file_name_X, file_name_Y, file_name_G]:
            utilmlab.ensure_dir(os.path.dirname(el))

        # Save X, Y, G separately for each coefficient and each X distribution
        np.savetxt(file_name_X, X)
        np.savetxt(file_name_Y, Y)
        np.savetxt(file_name_G, G)
