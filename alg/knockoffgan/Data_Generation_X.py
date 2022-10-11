'''
X Generation function
Jinsung Yoon (9/27/2018)
'''

# Necessary Packages
import numpy as np

#%% 1. Normal Generation
# X ~ N(0, sigma)
def Normal_Generation_X (n, p, sigma):
    
    Output_X = np.random.normal(0,np.sqrt(sigma),[n,p])
    
    return Output_X

#%% 2. AR_Normal Generation. 
'''
X_t = coef * X_t-1 + n 
n ~ N(0, sigma^2)
sigma = np.sqrt(margin*(1-coef*coef)) 
Therefore, Marginal distribution is N(0, margin)
'''

def AR_Normal_Generation_X (n, p, coef, margin):

    # Initialization
    Output_X = np.zeros([n,p])
        
    # Standard Deviation of n
    sigma = np.sqrt(margin*(1-coef*coef)) 
        
    # For each sample
    for i in range(n):
        
        # For each feature
        for j in range(p):
        
            # Starting feature
            if (j == 0):            
                Output_X[i,j] = np.random.normal(0,np.sqrt(margin))
                            
            # AR(1) Generation
            else:                
                Output_X[i,j] = coef * Output_X[i,j-1] + np.random.normal(0,sigma)
    
    return Output_X
    
    
#%% 3. Uniform Generation
# X ~ U(-3/sqrt(n), 3/sqrt(n))

def Uniform_Generation_X (n, p, sigma):
    
    Output_X = np.random.uniform(-3*np.sqrt(sigma),3*np.sqrt(sigma),[n,p])
    
    return Output_X   
    
#%% 4. AR_Uniform Gneration
def AR_Uniform_Generation_X (n, p, coef, margin):

    # Initialization
    Output_X = np.zeros([n,p])
        
    # Standard Deviation of n
    sigma = np.sqrt(margin*(1-coef*coef)) 
        
    # For each sample
    for i in range(n):
        
        # For each feature
        for j in range(p):
        
            # Starting feature
            if (j == 0):            
                Output_X[i,j] = np.random.uniform(-3*np.sqrt(margin),3*np.sqrt(margin))
                            
            # AR(1) Generation
            else:                
                Output_X[i,j] = coef * Output_X[i,j-1] + np.random.uniform(-3*(sigma),3*(sigma))
    
    return Output_X
   