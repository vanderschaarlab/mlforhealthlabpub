'''
Jinsung Yoon (09/06/2018)
Data Loading
'''

#%% Necessary Packages
import numpy as np
import scipy.io as sio
#%% Google data loading

'''
1. train_rate: training / testing set ratio
2. missing_rate: the amount of introducing missingness
'''

def Data_Loader_ICU(train_rate = 0.8, missing_rate = 0.2, window = 48, start_end = 'Start'):
    
  #%%
    # Input data
    Ori_Data = sio.loadmat('../data/ICU_Data.mat')

    # Extract X, T, Y and N
    Temp_Train = Ori_Data['Time_Varying_Feature_Train']
    Temp_Test = Ori_Data['Time_Varying_Feature_Test']
    
    trainT = Temp_Train[:,1]
    testT = Temp_Test[:,1]
    
    trainX = Temp_Train[:,0]
    testX = Temp_Test[:,0]
    
    trainN = len(trainT)
    testN = len(testT)
    
    # Length of stay for each patient
    trainL = np.zeros([trainN,])
    for i in range(trainN):
        trainL[i] = len(trainT[i])
        
    # Remove patients with less than window length of stay
    idx = np.where(trainL >= window)[0]
    trainX = trainX[idx]
    trainT = trainT[idx]
    trainN = len(trainX)
    
    # Length of stay for each patient
    testL = np.zeros([testN,])
    for i in range(testN):
        testL[i] = len(testT[i])
        
    # Remove patients with less than window length of stay
    idx = np.where(testL >= window)[0]
    testX = testX[idx]
    testT = testT[idx]
    testN = len(testX)
    
    # Only use the admission to admission + window
    for i in range(trainN):
        if start_end == 'Start':
            Temp = trainX[i][:window,:]
            trainX[i] = Temp
            Temp = trainT[i][:window]
            trainT[i] = Temp
        if start_end == 'End':
            Temp = trainX[i][-window:,:]
            trainX[i] = Temp
            Temp = trainT[i][-window:]
            trainT[i] = Temp
        
    for i in range(testN):
        if start_end == 'Start':
            Temp = testX[i][:window,:]
            testX[i] = Temp
            Temp = testT[i][:window]
            testT[i] = Temp
        if start_end == 'End':
            Temp = testX[i][-window:,:]
            testX[i] = Temp
            Temp = testT[i][-window:]
            testT[i] = Temp
    
    #%% Normalization
    Col_No = len(trainX[0][0,:])
    Min_Val = np.ones([Col_No,]) * 100000
    Max_Val = np.ones([Col_No,]) * -100000
    
    for i in range(trainN):
        Temp = trainX[i]
        Temp_Max = np.max(Temp,0)
        Temp_Min = np.min(Temp,0)
    
        for j in range(Col_No):
            if (Temp_Max[j] > Max_Val[j]):
                Max_Val[j] = Temp_Max[j]
            if (Temp_Min[j] < Min_Val[j]):
                Min_Val[j] = Temp_Min[j]
    
    for i in range(testN):
        Temp = testX[i]
        Temp_Max = np.max(Temp,0)
        Temp_Min = np.min(Temp,0)
    
        for j in range(Col_No):
            if (Temp_Max[j] > Max_Val[j]):
                Max_Val[j] = Temp_Max[j]
            if (Temp_Min[j] < Min_Val[j]):
                Min_Val[j] = Temp_Min[j]
                
    #%%
    
    for i in range(trainN):
        Temp = trainX[i]
    
        for j in range(Col_No):
            Temp[:,j] = Temp[:,j] - Min_Val[j]
            if (Max_Val[j] - Min_Val[j] > 0):
                Temp[:,j] = Temp[:,j] / (Max_Val[j] - Min_Val[j])
            
        trainX[i] = Temp
    
    for i in range(testN):
        Temp = testX[i]
    
        for j in range(Col_No):
            Temp[:,j] = Temp[:,j] - Min_Val[j]
            if (Max_Val[j] - Min_Val[j] > 0):
                Temp[:,j] = Temp[:,j] / (Max_Val[j] - Min_Val[j])
            
        testX[i] = Temp
    
    #%%
    
    # Make object to list
    dataX = list()
    for i in range(trainN):
        dataX.append(trainX[i])
    
    for i in range(testN):
        dataX.append(testX[i])   
    
    #%%
    seq_length = window
    col_no = Col_No
    row_no = len(dataX)
    
    
    #%% Introduce Missingness (MCAR)
    
    dataZ = []
    dataM = []
    dataT = []
    
    for i in range(row_no):
        
        #%% Missing matrix construct
        temp_m = np.random.uniform(0,1,[seq_length, col_no]) 
        m = np.zeros([seq_length, col_no])
        m[np.where(temp_m >= missing_rate)] = 1
        
        dataM.append(m)
        
        #%% Introduce missingness to the original data
        z = np.copy(dataX[i])    
        z[np.where(m==0)] = 0
        
        dataZ.append(z)
        
        #%% Time gap generation
        t = np.ones([seq_length, col_no])
        for j in range(col_no):
            for k in range(seq_length):
                if (k > 0):
                    if (m[k,j] == 0):
                        t[k,j] = t[k-1,j] + 1
                        
        dataT.append(t)
        
    #%% Building the dataset
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    T: Time Gap
    '''
                
    #%% Train / Test Division   
    train_size = int(len(dataX) * train_rate)
    
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainZ, testZ = np.array(dataZ[0:train_size]), np.array(dataZ[train_size:len(dataX)])
    trainM, testM = np.array(dataM[0:train_size]), np.array(dataM[train_size:len(dataX)])
    trainT, testT = np.array(dataT[0:train_size]), np.array(dataT[train_size:len(dataX)])
    
    return [trainX, trainZ, trainM, trainT, testX, testZ, testM, testT]

