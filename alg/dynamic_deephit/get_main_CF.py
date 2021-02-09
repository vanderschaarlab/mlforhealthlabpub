

'''
First implemented: 01/25/2018
  > For survival analysis on longitudinal dataset
By CHANGHEE LEE

Modifcation List:
    - (02/13/2018) C-index, B-score evaluation added (using pred_time and eval_time)
    - (02/13/2018) Valdiation Added (frist version is based on the mean of C-index w/ p_time=1 and e_time=3)
    - (02/14/2018) Prediction modified (divided by the denominator)
    - (02/15/2018) Cystic-Fibrosis Added
    - (02/21/2018) Comorbidity indes added (specific inidces can be selected among multiple features)
    - (02/22/2018) Burn-in training for RNN is added
    - (02/28/2018) Boosting training Set is added (N longitudinal measurements --> N samples with 1~N longitudinal measurements)
'''
_EPSILON = 1e-08


import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepLongitudinal import Model_Longitudinal_Attention
from utils_eval import c_index, brier_score


##### USER-DEFINED FUNCTIONS
def log(x): 
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask

def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask

def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask


def f_get_minibatch(mb_size, x, x_org, x_mi, label, time, mask1, mask2, mask3, mask4, mask5):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb     = x[idx, :, :].astype(float)
    x_org_mb = x_org[idx, :, :].astype(float)
    x_mi_mb  = x_mi[idx, :, :].astype(float)
    k_mb     = label[idx, :].astype(float) # censoring(0)/event(1,2,..) label
    t_mb     = time[idx, :].astype(float)
    m1_mb    = mask1[idx, :, :].astype(float) #rnn_mask
    m2_mb    = mask2[idx, :, :].astype(float) #rnn_mask
    m3_mb    = mask3[idx, :, :].astype(float) #fc_mask
    m4_mb    = mask4[idx, :, :].astype(float) #fc_mask
    m5_mb    = mask5[idx, :].astype(float) #fc_mask
    return x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb

###MODIFY
def f_get_boosted_trainset(x, x_org, x_mi, time, label, mask1, mask2, mask3, mask4, mask5):
    total_sample = 0
    for i in range(np.shape(x)[0]):
        total_sample += np.sum(np.sum(x[i], axis=1) != 0)

    new_label          = np.zeros([total_sample, np.shape(label)[1]])
    new_time           = np.zeros([total_sample, np.shape(time)[1]])
    new_x              = np.zeros([total_sample, np.shape(x)[1], np.shape(x)[2]])
    new_x_org          = np.zeros([total_sample, np.shape(x_org)[1], np.shape(x_org)[2]])
    new_x_mi           = np.zeros([total_sample, np.shape(x_mi)[1], np.shape(x_mi)[2]])
    new_mask1          = np.zeros([total_sample, np.shape(mask1)[1], np.shape(mask1)[2]])
    new_mask2          = np.zeros([total_sample, np.shape(mask2)[1], np.shape(mask2)[2]])
    new_mask3          = np.zeros([total_sample, np.shape(mask3)[1], np.shape(mask3)[2]])
    new_mask4          = np.zeros([total_sample, np.shape(mask4)[1], np.shape(mask4)[2]])
    new_mask5          = np.zeros([total_sample, np.shape(mask5)[1]])

    tmp_idx = 0
    for i in range(np.shape(x)[0]):
        max_meas = np.sum(np.sum(x[i], axis=1) != 0)

        for t in range(max_meas):
            new_label[tmp_idx+t, 0] = label[i,0]
            new_time[tmp_idx+t, 0]  = time[i,0]

            new_x[tmp_idx+t,:(t+1), :] = x[i,:(t+1), :]
            new_x_org[tmp_idx+t,:(t+1), :] = x_org[i,:(t+1), :]
            new_x_mi[tmp_idx+t,:(t+1), :] = x_mi[i,:(t+1), :]

            if t < (max_meas - 1):
                new_mask1[tmp_idx+(t+1),:(t+1), :] = mask1[i,:(t+1), :]

            new_mask2[tmp_idx+t, t, :] = mask2[i, (max_meas-1), :]
            new_mask3[tmp_idx+t, :, :] = f_get_fc_mask1(x_org[i,t,[1]].reshape([-1,1]), num_Event, num_Category) #age at the measurement
            new_mask4[tmp_idx+t, :, :] = mask4[i, :, :]
            new_mask5[tmp_idx+t, :]    = mask5[i, :]

        tmp_idx += max_meas
        
    return(new_x, new_x_org, new_x_mi, new_time, new_label, new_mask1, new_mask2, new_mask3, new_mask4, new_mask5)



def f_get_prediction_v5(sess, model, data, data_org, data_mi, time, label, mask2, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))
    new_time    = np.zeros(np.shape(time))
    new_label   = np.zeros(np.shape(label))

    new_mask2   = np.zeros(np.shape(mask2))
    last_meas   = np.zeros([np.shape(data)[0],1])

    for i in range(np.shape(data)[0]):
        if np.max(data_org[i,:,1]) <= pred_horizon:
            new_data[i, :, :]    = data[i, :, :]
            new_data_mi[i, :, :] = data_mi[i, :, :]
            new_mask2[i, :, :]   = mask2[i, :, :]
            last_meas[i, 0]      = np.max(data_org[i,:,1])
        elif np.min(data_org[i, data_org[i, :, 1] > 0, 1]) <= pred_horizon:
            for t in range(np.shape(data)[1]):
                if data_org[i, t, 1] <= pred_horizon:
                    new_data[i, t, :]    = data[i, t, :]
                    new_data_mi[i, t, :] = data_mi[i, t, :]
                    last_meas[i, 0]      = data_org[i, t, 1]
                else:
                    new_mask2[i, t-1, :] = 1
                    break

    last_meas   = last_meas[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :]
    new_time    = time[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :]
    new_label   = label[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :]
    new_mask2   = new_mask2[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :, :]
    new_data_mi = new_data_mi[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :, :]
    new_data    = new_data[np.where(np.sum(np.sum(new_data, axis=2), axis=1) !=0)[0], :, :]

    return model.predict(new_data, new_data_mi, new_mask2), new_time, new_label, last_meas


# continuous features first, binary features next.
cont_list = ['Age', 'Weight', 'Height', 'BMI', 
             'FEV1', 'FEV1 Predicted', 'Best FEV1', 'Best FEV1 Predicted', 'IV Antibiotic Days Hosp', 
             'IV Antibiotic Days Home', 'Non-IV Hospital Admission'] #1~11
bin_list  = ['Gender', 'Smoking Status', #12 ~ 13
             'Class I Mutation', 'Class II Mutation', 'Class III Mutation', 'Class IV Mutation',  'Class V Mutation', 'Class VI Mutation',
                'DF508 Mutation', 'G551D Mutation', 'Homozygous', 'Heterozygous', #14~23
             'Burkholderia Cepacia', 'Pseudomonas Aeruginosa', 'Haemophilus Influenza', 'Klebsiella Pneumoniae', 'Ecoli', 'ALCA',
                 'Aspergillus', 'NTM', 'Gram-Negative', 'Xanthomonas', 'Staphylococcus Aureus', #24~34
             'Liver Disease',  'Asthma', 'ABPA', 'Hypertension', 'Diabetes', 'Arthropathy', 'Bone fracture', 'Osteoporosis',  #35~42
                 'Osteopenia', 'Cancer', 'Cirrhosis', 'Kidney Stones', 'Depression', 'Hemoptysis', 'Pancreatitus', 'Hearing Loss', #43~50
                 'Gall bladder', 'Colonic structure', 'Intestinal Obstruction', 'GI bleeding non-var source', 'GI bleeding var source', #51~55
             'Dornase Alpha', 'Anti-fungals', 'Liver Enzymes', 'Lab Liver Enzymes', 'HyperSaline', 'HypertonicSaline',  #56~61
                 'Tobi Solution', 'Cortico Combo', 'Noninvasive Ventilation', 'Acetylcysteine', 'Aminoglycoside', 'iBuprofen', #62~67
                 'Drug Dornase', 'HDI Buprofen', 'Tobramycin', 'Leukotriene', 'Colistin', 'Diabetes Inter Insulin', #68~73
                 'Macrolida Antibiotics', 'Inhaled Broncho BAAC', 'Inhaled Broncho LAAC', 'Inhaled Broncho SAAC', #74~77
                 'Inhaled Broncho LABA', 'Inhaled Bronchodilators', 'Cortico Inhaled', 'Oral Broncho THEOPH', #78~81
                 'Oral Broncho BA', 'Oral Hypoglycemic Agents', 'Chronic Oral Antibiotic', 'Cortico Oral', 'O2 Prn', 'O2 Exc',
                 'O2 Noct', 'O2 Cont', 'Oxygen Therapy'] #82~90


feat_list = cont_list + bin_list

static_feature = [1,12,14,15,16,17,18,19,20,21,22,23]
time_varying_feature =[2,3,4,5,6,7,8,9,10,11,13,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
                       53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]



ACTIVATION_FN               = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'tanh': tf.nn.tanh}
data_mode                   = 'CysFib_v9' #PBC2, CysFib, CysFib_single, CysFib_v8
como_mode                   = 'como_all' #v1, v2, v3, v4, v5 ... all


##### MAIN SETTING
valid_mode                  = 'ON' #ON / OFF
burn_in_mode                = 'ON' #ON / OFF
boost_mode                  = 'ON' #ON / OFF

##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (1 + num_features)
    x_dim_cont              = dim of continuous features
    x_dim_bin               = dim of binary features
    mask1, mask2            = used for shared network (RNN structure)
    mask3, mask4, mask5     = used for cause-specific network (FCNet structure)
'''
if data_mode == 'CysFib_v9':
    (x_dim, x_dim_cont, x_dim_bin), (data, data_org, time, label), (mask1, mask2, mask3, mask4, mask5), (data_mi) = impt.import_dataset_cysfib_ver7_missing(num_Event=2, norm_mode = 'standard')
    data[:, :-1, 0] = data[:, 1:, 0] ### delta_{j} = t_{j+1} - t_{j}
    data[:, -1, 0]  = 0
  
    for i in range(np.shape(data)[0]):
        mask1[i, :, :] = mask1[i, 0, :] # this only indicates what feature to contain in loss3.
    
    time_interval           = 1./12. # time interval is 6 month
    if como_mode == 'como_v1':
        COMO_IDX = [5, 6]  # Como_v1   ('FEV1', 'FEV1 predicted')
    elif como_mode == 'como_v2':
        COMO_IDX = [5, 6, 9, 10]  # Como_v2   ('FEV1', 'FEV1 predicted', 'IV Antibiotic Days Hosp', 'IV Antibiotic Days Home')
    elif como_mode == 'como_v3':
        COMO_IDX = [5, 6, 9, 10, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45] # Como_v3  
        #('FEV1', 'FEV1 predicted', 'IV Antibiotic Days Hosp', 'IV Antibiotic Days Home', 'Liver Disease',  'Asthma', 'ABPA', 'Hypertension', 'Diabetes', 'Arthropathy', 'Bone fracture', 'Osteoporosis'
          #'Osteopenia', 'Cancer', 'Cirrhosis', 'Kidney Stones', 'Depression', 'Hemoptysis', 'Pancreatitus', 'Hearing Loss')
    elif como_mode == 'como_v6':
        COMO_IDX = [5, 6, 9, 10, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40] # Como_v4  
        #('FEV1', 'FEV1 predicted', 'IV Antibiotic Days Hosp', 'IV Antibiotic Days Home', 'Liver Disease',  'Asthma', 'ABPA', 'Hypertension', 'Diabetes', 'Arthropathy', 'Osteoporosis', 'Osteopenia', 'Cancer', 'Cirrhosis')
    elif como_mode == 'como_v5':
        COMO_IDX = [5, 6, 9, 10, 32, 33, 34, 35, 36, 41, 42, 43, 45, 46] # Como_v5  
        #('FEV1', 'FEV1 predicted', 'IV Antibiotic Days Hosp', 'IV Antibiotic Days Home', Liver Disease',  'Asthma', 'ABPA', 'Hypertension', 'Diabetes', 'Cancer', 'Cirrhosis', 'Kidney Stones', 'Hemoptysis', 'Pancreatitus')
    else:
        print ('ERROR: COMO_MODE NOT FOUND !!!')
else:
    print ('ERROR:  DATA_MODE NOT FOUND !!!')


_, num_Event, num_Category  = np.shape(mask3)  # dim of mask3: [subj, Num_Event, Num_Category]
max_length                  = np.shape(data)[1]

if como_mode != 'como_all':
    tmp_idx                 = np.arange(x_dim) 
    tmp_idx                 = np.delete(tmp_idx, COMO_IDX)
    mask1[:,:,tmp_idx]      = 0
else: #use all time_varying_features
    tmp_idx                 = np.arange(x_dim) 
    tmp_idx                 = np.delete(tmp_idx, (np.asarray(time_varying_feature)-1))
    mask1[:,:,tmp_idx]      = 0




def get_valid_performance(in_parser, out_itr, MAX_VALUE = -99, OUT_ITERATION=5, seed=1234):

    if out_itr == 0:
        selected_feat = [0, 1, 2, 5, 6, 9, 13, 24, 29, 34, 37, 39, 45, 47, 56, 60, 64, 66, 84, 86, 89, 90]
        x_dim      = len(selected_feat)
        x_dim_cont = 5
        x_dim_bin  = x_dim - 1 - x_dim_cont

    elif out_itr == 1:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 13, 24, 37, 39, 43, 44, 45, 56, 60, 62, 64, 67, 68, 70, 73, 84, 86, 90]
        x_dim      = len(selected_feat)
        x_dim_cont = 6
        x_dim_bin  = x_dim - 1 - x_dim_cont

    elif out_itr == 2:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 24, 31, 37, 56, 60, 62, 64, 67, 68, 73, 84]
        x_dim      = len(selected_feat)
        x_dim_cont = 6
        x_dim_bin  = x_dim - 1 - x_dim_cont

    elif out_itr == 3:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 24, 36, 37, 39, 45, 47, 56, 59, 60, 61, 64, 66, 73, 84]
        x_dim      = len(selected_feat)
        x_dim_cont = 6
        x_dim_bin  = x_dim - 1 - x_dim_cont

    elif out_itr == 4:
        selected_feat = [0, 1, 2, 5, 6, 9, 10, 12, 19, 24, 31, 37, 45, 56, 59, 62, 64, 66, 67, 73, 84]
        x_dim      = len(selected_feat)
        x_dim_cont = 6
        x_dim_bin  = x_dim - 1 - x_dim_cont
    
    tmp_data     = data[:, :, selected_feat]
    tmp_data_org = data_org[:, :, selected_feat]
    tmp_data_mi  = data_mi[:,:, selected_feat]
    tmp_mask1    = mask1[:,:, selected_feat]
    tmp_mask2    = mask2[:,:, selected_feat]

    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration_burn_in           = in_parser['iteration_burn_in']
    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']



    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))

    initial_W                   = tf.contrib.layers.xavier_initializer()


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'x_dim_cont'    : x_dim_cont,
                                    'x_dim_bin'     : x_dim_bin,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category,
                                    'max_length'    : max_length }

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_RNN'         : in_parser['h_dim_RNN'],
                                    'h_dim_FC'          : in_parser['h_dim_FC'],
                                    'num_layers_RNN'    : in_parser['num_layers_RNN'],
                                    'num_layers_ATT'    : in_parser['num_layers_ATT'],
                                    'num_layers_CS'     : in_parser['num_layers_CS'],
                                    'RNN_type'          : in_parser['RNN_type'],
                                    'BiRNN'             : in_parser['BiRNN'],
                                    'FC_active_fn'      : ACTIVATION_FN[in_parser['FC_active_fn']],
                                    'RNN_active_fn'     : ACTIVATION_FN[in_parser['RNN_active_fn']],
                                    'initial_W'         : initial_W }


    file_path = in_parser['out_path'] + '/valid/itr_' + str(out_itr)
    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)

    #change parameters...
    if not os.path.exists(file_path + '/results/'):
        os.makedirs(file_path + '/results/')
    if not os.path.exists(file_path + '/models/'):
        os.makedirs(file_path + '/models/')
    if not os.path.exists(file_path_final + '/models/'):
        os.makedirs(file_path_final + '/models/')


    pred_time = [30, 40, 50] # x-yr prediction time
    eval_time = [1, 3, 5, 10] # x-yr evaluation time (for C-index and Brier-Score)


    print ('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )

    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_Longitudinal_Attention(sess, "FHT_Landmarking", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data,te_data, tr_data_org,te_data_org, tr_data_mi, te_data_mi, tr_time,te_time, 
     tr_label,te_label, tr_mask1,te_mask1, tr_mask2,te_mask2, tr_mask3,te_mask3, tr_mask4,te_mask4, tr_mask5,te_mask5) = train_test_split(tmp_data, tmp_data_org, tmp_data_mi, time, label, tmp_mask1, tmp_mask2, mask3, mask4, mask5, test_size=0.2, random_state=seed+out_itr) 

    (tr_data,va_data, tr_data_org, va_data_org, tr_data_mi, va_data_mi, tr_time,va_time, 
     tr_label,va_label, tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3, tr_mask4,va_mask4, tr_mask5,va_mask5) = train_test_split(tr_data, tr_data_org, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5, test_size=0.2, random_state=seed) 

    if boost_mode == 'ON':
        tr_data, tr_data_org, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5 = f_get_boosted_trainset(tr_data, tr_data_org, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)
    
    ### TRAINING - BURN-IN
    if burn_in_mode == 'ON':
        print( "BURN-IN TRAINING ...")
        for itr in range(iteration_burn_in):
            x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb = f_get_minibatch(mb_size, tr_data, tr_data_org, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)
            DATA = (x_mb, x_org_mb, k_mb, t_mb)
            MASK = (m1_mb, m2_mb)
            MISSING = (x_mi_mb)

            _, loss_curr = model.train_burn_in(DATA, MASK, MISSING, keep_prob, lr_train)

            if (itr+1)%1000 == 0:
                print('|| Epoch: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(loss_curr)), 'green' , attrs=['bold']))

    max_valid = -99
    stop_flag = 0

    ### TRAINING - MAIN
    print( "MAIN TRAINING ...")
    for itr in range(iteration):
        if stop_flag > 5: #for faster early stopping
            break
        else:
            x_mb, x_org_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb, m5_mb = f_get_minibatch(mb_size, tr_data, tr_data_org, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, tr_mask4, tr_mask5)
            DATA = (x_mb, x_org_mb, k_mb, t_mb)
            MASK = (m1_mb, m2_mb, m3_mb, m4_mb, m5_mb)
            MISSING = (x_mi_mb)
            PARAMETERS = (alpha, beta, gamma)
            _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)
                
            if (itr+1)%1000 == 0:
                print('|| Epoch: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(loss_curr)), 'yellow' , attrs=['bold']))

            ### VALIDATION  (based on average C-index of our interest)
            if (itr+1)%1000 == 0:
                if valid_mode == 'ON':
                    for p, p_time in enumerate(pred_time):
                        ### PREDICTION
                        pred_horizon = int(p_time / time_interval)
                        pred, tmp_time, tmp_label, _ = f_get_prediction_v5(sess, model, va_data, va_data_org, va_data_mi, va_time, va_label, va_mask2, pred_horizon)

                        ### EVALUATION
                        val_result1 = np.zeros([num_Event, len(eval_time)])

                        for t, t_time in enumerate(eval_time):
                            eval_horizon = int(t_time/time_interval) + pred_horizon
                            if eval_horizon >= num_Category:
                                print ('ERROR: evaluation horizon is out of range')
                                val_result1[:, t] = 0 #-1 is too aggressive.
                            else:
                                # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                                risk = np.sum(pred[:,:,pred_horizon:(eval_horizon+1)], axis=2) #risk score until eval_time
                                risk = risk / (np.sum(np.sum(pred[:,:,pred_horizon:], axis=2), axis=1, keepdims=True) +_EPSILON) #conditioniong on t > t_pred
                                for k in range(num_Event):
                                    val_result1[k, t] = c_index(risk[:,k], tmp_time, (tmp_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

                        if p == 0:
                            val_final1 = val_result1
                        else:
                            val_final1 = np.append(val_final1, val_result1, axis=0)

                    tmp_valid = np.mean(val_final1)

                    if tmp_valid >  max_valid:
                        stop_flag = 0
                        max_valid = tmp_valid
                        saver.save(sess, file_path + '/models/model_itr_' + str(out_itr))
                        print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))

                        if max_valid > MAX_VALUE:
                            saver.save(sess, file_path_final + '/models/model_itr_' + str(out_itr))
                    else:
                        stop_flag += 1

    return max_valid
