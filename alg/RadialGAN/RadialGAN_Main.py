
import numpy as np

from Data_Generation_Maggic import Data_Generation_Maggic
from RadialGAN import RadialGAN

Iteration = 5
alpha = 0.1
#%% Data Generation
AUC = list()
APR = list()
for i in range(Iteration):
  [Train_X, Train_M, Train_G, Train_Y, Test_X, Test_M, Test_G, Test_Y, Valid_X, Valid_M, Valid_G, Valid_Y, FSet] = Data_Generation_Maggic(Set_No = 3, fix = 0)
  
  Prediction, Output_Logit = RadialGAN(Train_X, Train_M, Train_G, Train_Y, Test_X, Test_M, Test_G, Test_Y, Valid_X, Valid_M, Valid_G, Valid_Y, FSet, alpha)
    
  #%%
  
  Output_Logit = np.asarray(Output_Logit)
  
  AUC_Imprv_Logit = Output_Logit[:3,0] - Output_Logit[3:,0]
  APR_Imprv_Logit = Output_Logit[:3,1] - Output_Logit[3:,1]
  
  AUC.append(np.mean(AUC_Imprv_Logit))
  APR.append(np.mean(APR_Imprv_Logit))

print('Logit')
print('AUC Improvement: ' + str(np.round(np.mean(AUC),4)))
print('APR Improvement: ' + str(np.round(np.mean(APR),4)))