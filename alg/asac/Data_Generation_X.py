# Necessary Packages
import numpy as np

#%% AR(1) Generation. 
'''
X_t = coef * X_t-1 + n 
n ~ N(0, sigma^2)
sigma = np.sqrt(margin*(1-coef*coef)) 
Therefore, Marginal distribution is N(0, margin)

Inputs
- n: Number of samples
- p: Number of features
- phi: Autoregressiveness
- margin: Std of the normal distribution 
'''
def AR_Gauss_X1 (n, d, t, phi, sigma):

    # Initialization
    Output_X = list()
                
    # For each sample
    for i in range(n):
        
        Temp_Output_X = np.zeros([t,d])
        
        # For each feature
        for j in range(d):
        
            for k in range(t):
              
                # Starting feature
                if (k == 0):            
                    Temp_Output_X[k,j] = np.random.normal(0,sigma)
                                
                # AR(1) Generation
                else:                
                    Temp_Output_X[k,j] = phi[j] * Temp_Output_X[k-1,j] + (1-phi[j])*np.random.normal(0,sigma)
    
        Output_X.append(Temp_Output_X)    
    
    return Output_X
  
#%% 
def AR_Gauss_X2 (n, d, t, phi, sigma, gamma):

    # Initialization
    Output_X = list()
                
    # For each sample
    for i in range(n):
        
        Temp_Output_X = np.zeros([t,2*d])
        
        # For each feature
        for j in range(d):
        
            for k in range(t):
              
                # Starting feature
                if (k == 0):            
                    Temp_Output_X[k,j] = np.random.normal(0,sigma)
                                
                # AR(1) Generation
                else:                
                    Temp_Output_X[k,j] = phi[j] * Temp_Output_X[k-1,j] + (1-phi[j])*np.random.normal(0,sigma)
                
                Temp_Output_X[k,d+j] = Temp_Output_X[k,j] + np.random.normal(0,gamma)       
        
    
        Output_X.append(Temp_Output_X)    
    
    return Output_X
  