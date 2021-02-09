'''
2019 NeurIPS Submission
Title: Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate
Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

DPBAG Function
- Use train and valid datasets to make Differentially Private Classification Model
- Use test set to measure the performances

Inputs
- train, valid, test sets
- Parameters (epsilon, delta, teacher_no, part number)

Outputs
- AUROC
- AUPRC
- Accuracy
- Budget
- Differentially Private Classification Model
'''

#%% Necessary Packages
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

#%% Function start

def DPBag (x_train, y_train, x_valid, x_test, y_test, parameters):      
  
    # Parameters
    No = len(x_train[:,0])
                    
    epsilon    = parameters['epsilon']
    delta      = parameters['delta'] 
    lamda      = parameters['lamda'] 
    teacher_no = parameters['teacher_no'] 
    part_no    = parameters['part_no']
    
    L = 80
    # Initialize alpha
    alpha = [[0 for i in range(No)] for j in range(L)]
        
    #%% Partition the data (Divide data into multiple partitions)

    # Initialize partitions
    Part_X = list()
    Part_Y = list()
    
    # Save the partition number and teacher number for each sample
    Part_Save = np.zeros([No, part_no, teacher_no])
    
    # For each partition
    for p in range(part_no):
          
        Data_X = list()
        Data_Y = list()
        
        # Divide them into multiple disjoint sets (# of teachers)
        idx = np.random.permutation(No)
        
        for i in range(teacher_no):
            
            # Index of samples in each disjoint set
            start_idx = i * int(No/teacher_no)
            end_idx = (i+1) * int(No/teacher_no)
            temp_idx = idx[start_idx:end_idx]
            
            # Divide the data
            Data_X.append(x_train[temp_idx,:])
            Data_Y.append(y_train[temp_idx])
            
            # Save the teacher number and partition number
            Part_Save[temp_idx,p,i] = 1
        
        # Save each partition
        Part_X.append(Data_X)
        Part_Y.append(Data_Y)
        
    #%% Teacher Training    
        
    # Initialize teacher models
    teacher_models = list()
    
    # For each partition
    for p_idx in (range(part_no)):
        
        # Initialize teacher models for each partition
        part_models = list()
      
        # For each teacher
        for t_idx in (range(teacher_no)):
              
            # Load data for each teacher in each partition
            x_temp = Part_X[p_idx][t_idx]
            y_temp = Part_Y[p_idx][t_idx]
            
            # Train the teacher model
            model = LogisticRegression()
            model.fit(x_temp, y_temp)
            
            # Save the teacher model
            part_models.append(model)
                
        # Save the teacher model in each partition
        teacher_models.append(part_models)
            
    #%% Student Training
    
    # Initialize some parameters
    # Tracking current epsilon
    epsilon_hat = 0
    # No of Privacy budget
    mb_idx = 0
    
    # Set the public data
    x_mb = x_valid[:No,:]
    
    # r_s initialize
    r_mb = np.zeros([No,1])
    
    # Output Initialization (Accuracy, AUROC, AUPRC, Privacy Budget, DP Classification Model)
    Output_ACC = list()
    Output_AUC = list()
    Output_APR = list()
    Output_Budget = list()
    Output_Model = list()

    #%% Get all the n_c, n_c(x), and m(x) to speed up the algorithm
    
    # Tx_all (T_i,j(x)) Initialization
    Tx_all = np.zeros([part_no, teacher_no, No])
               
    # Outputs of all teachers for public data
    for p_idx in range(part_no):
            
        for t_idx in range(teacher_no):
                  
            teacher_pred_result_temp = teacher_models[p_idx][t_idx].predict_proba(x_mb)[:,1]          
                    
            # Save them to the T_i,j(x)
            Tx_all[p_idx, t_idx, :] = np.reshape(1*(teacher_pred_result_temp>0.5), [-1])
    
    ## Compute nc_all (n_c)
    nc_all = np.zeros([No,2])
    
    nc_all[:,0] = np.sum(1-Tx_all, axis = (0,1)) / part_no
    nc_all[:,1] = np.sum(Tx_all, axis = (0,1)) / part_no
    
    # Compute ncx_all (n_c(x))
    ncx_all = np.zeros([No, No, 2])    
      
    ncx_all[:,:,1] = np.einsum('npt,ptc -> nc', Part_Save, Tx_all) / part_no
    ncx_all[:,:,0] = 1 - ncx_all[:,:,1]
    
    # Compute m(x) 
    mx_all = np.max(ncx_all, axis = 2)

    #%% Get access to the data until the epsilon is less than the threshold
    
    while ((epsilon_hat < epsilon) & (mb_idx < No)): 
              
        # PATE_lambda (x)
        r_mb[mb_idx,0] = np.argmax([nc_all[mb_idx,0] + np.random.laplace(scale=1/lamda), nc_all[mb_idx,1] + np.random.laplace(scale=1/lamda)])    
      
        # Compute alpha    
        for l_idx in range(L):
                  
            first_term  = (2 * ( (lamda*mx_all[:,mb_idx])**2 ) * (l_idx + 1) * (l_idx + 2)) 
              
            alpha[l_idx] = alpha[l_idx] + first_term
        
        # compute epsilon hat       
        min_list = list()
        
        for l_idx in range(L):
          
            temp_min_list = (np.max(alpha[l_idx]) + np.log(1/delta)) / (l_idx+1)        
            min_list.append(temp_min_list)
                 
        #%% If epsilon is 1,2,...,10
               
        if (int(epsilon_hat) < int(np.min(min_list))):
        
        #%% Student Training
      
            # Use entire data until int(epsilon_hat) < int(np.min(min_list))
            s_x_train = x_mb[:mb_idx,:]
            s_y_train = r_mb[:mb_idx,:]
                            
            # Train the DP classification model
            model = LogisticRegression()
            model.fit(s_x_train, s_y_train)
            
         #%% Evaluations
            student_y_final = model.predict_proba(x_test)[:,1]
            
            student_pred_result = roc_auc_score (y_test, student_y_final)  
            
            print('Student AUC: ' +str(np.round(student_pred_result,4)) + ', Epsilon: ' + str(np.round(epsilon_hat)))
            
            Output_AUC.append(np.round(roc_auc_score (y_test, student_y_final),4))
            Output_APR.append(np.round(average_precision_score (y_test, student_y_final),4))
            Output_ACC.append(np.round(accuracy_score(y_test, student_y_final > 0.5),4))
            Output_Budget.append(mb_idx+1)
            Output_Model.append(model)
            
        #%% Epsilon, mb_update Update
        
        epsilon_hat = np.min(min_list)
        
        # The number of accessed samples
        mb_idx = mb_idx + 1
        
        # Print current state (per 1000 accessed samples)
        if (mb_idx % 1000 == 0):            
            print('step: ' + str(mb_idx) + ', epsilon hat: ' + str(epsilon_hat))        
  
    # Return Accuracy, AUROC, AUPRC, Privacy Budget, and DP Classification Model
    return Output_ACC, Output_AUC, Output_APR, Output_Budget, Output_Model
        
        