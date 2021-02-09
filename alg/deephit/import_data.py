'''
This provide the dimension/data/mask to train/test the network.

Once must construct a function similar to "import_dataset_SYNTHETIC":
    - DATA FORMAT:
        > data: covariates with x_dim dimension.
        > label: 0: censoring, 1 ~ K: K competing(single) risk(s)
        > time: time-to-event or time-to-censoring
    - Based on the data, creat mask1 and mask2 that are required to calculate loss functions.
'''
import numpy as np
import pandas as pd
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab



##### DEFINE USER-FUNCTIONS #####
def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

### MASK FUNCTIONS
'''
    fc_mask2      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask3      : To calculate LOSS_2 (ranking loss)
'''
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


def import_dataset_SYNTHETIC(norm_mode='standard'):
    in_filename = '{}/synthetic/synthetic_comprisk.csv'.format(
        utilmlab.get_data_dir())
    df = pd.read_csv(in_filename, sep=',')
    
    label           = np.asarray(df[['label']])
    time            = np.asarray(df[['time']])
    data            = np.asarray(df.iloc[:,4:])
    data            = f_get_Normalization(data, norm_mode)

    num_Category    = int(np.max(time) * 1.2)  #to have enough time-horizon
    num_Event       = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)

    x_dim           = np.shape(data)[1]

    mask1           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim)
    DATA            = (data, time, label)
    MASK            = (mask1, mask2)

    return DIM, DATA, MASK


def import_dataset_METABRIC(norm_mode='standard'):
    in_filename1 = '{}/metabric/cleaned_features_final.csv'.format(
        utilmlab.get_data_dir())
    in_filename2 = '{}/metabric/label.csv'.format(
        utilmlab.get_data_dir())

    df1 = pd.read_csv(in_filename1, sep =',')
    df2 = pd.read_csv(in_filename2, sep =',')

    data  = np.asarray(df1)
    data  = f_get_Normalization(data, norm_mode)
    
    time  = np.asarray(df2[['event_time']])
    # time  = np.round(time/12.) #unit time = month
    label = np.asarray(df2[['label']])

    
    num_Category    = int(np.max(time) * 1.2)        #to have enough time-horizon
    num_Event       = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)

    x_dim           = np.shape(data)[1]

    mask1           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim)
    DATA            = (data, time, label)
    MASK            = (mask1, mask2)

    return DIM, DATA, MASK
