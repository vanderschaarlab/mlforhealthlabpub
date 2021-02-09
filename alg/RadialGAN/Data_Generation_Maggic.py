## Jinsung Yoon, GAN for Class Imbalance (Jinsung Yoon, 01/08/2017)
import numpy as np

def Data_Generation_Maggic(Set_No, fix):
        
    Data = np.loadtxt("/home/vdslab/Documents/Jinsung/2018_Research/ICML/Model_Transfer/Data/Maggic/Feature.csv")
    Label = np.loadtxt("/home/vdslab/Documents/Jinsung/2018_Research/ICML/Model_Transfer/Data/Maggic/Label.csv")
    FSet = np.loadtxt("/home/vdslab/Documents/Jinsung/2018_Research/ICML/Model_Transfer/Data/Maggic/FSet.csv")
    Group = np.loadtxt("/home/vdslab/Documents/Jinsung/2018_Research/ICML/Model_Transfer/Data/Maggic/Group.csv", delimiter=",")    
    
##### Use the group with less than 500    
    Temp_Group_No = int(np.max(Group)) 
    Temp = np.zeros([Temp_Group_No])
    for i in range(Temp_Group_No):
        idx = Group == (i+1)
        Temp[i] = np.sum(idx)
          
    poss_set = np.where(Temp > 200)[0]
        
    poss_set_no = len(poss_set)
    
    
    # Label Missing Change
    idx = Label == -1
    Label[idx] = 0
        
    set_idx = np.random.permutation(poss_set_no)
    set_idx = poss_set[set_idx[:Set_No]]
    
    if (fix == 1):
        set_idx = poss_set[range(Set_No)]    
    
    # Data Chop-up
    for i in range(Set_No):
        if (i == 0):
            idx = (Group == (set_idx[i] + 1 ))
        if (i > 0):
            idx = idx + (Group == (set_idx[i] + 1))
            
    Data = Data[idx,:]
    Label = Label[idx,0]
    FSet = FSet[(set_idx),:]
    Group = Group[idx]
    
    Final_Group = Group.copy()
    
    for i in range(Set_No):
        idx = (Group == (set_idx[i]+1))
        Final_Group[idx] = (i+1)
        
    Group = Final_Group
    
    No = len(Data)
    Dim = len(Data[0,:])
                
    # Mask generation
    Mask = np.zeros([No,Dim]) 
    
    idx = (Data > -1)
    Mask[idx] = 1
      
   # Train Test Division        
    
    idx = np.random.permutation(No)
    train_idx = idx[:int(0.64*No)]
    valid_idx = idx[int(0.64*No):int(0.8*No)]
    test_idx = idx[int(0.8*No):]
    
    Train_X = Data[train_idx,:]
    Valid_X = Data[valid_idx,:]
    Test_X = Data[test_idx,:]

    Train_M = Mask[train_idx,:]
    Valid_M = Mask[valid_idx,:]
    Test_M = Mask[test_idx,:]
    
    Train_Y = Label[train_idx]
    Valid_Y = Label[valid_idx]
    Test_Y = Label[test_idx]
    
    Train_G = Group[train_idx]
    Valid_G = Group[valid_idx]
    Test_G = Group[test_idx]

    return [Train_X, Train_M, Train_G, Train_Y, Test_X, Test_M, Test_G, Test_Y, Valid_X, Valid_M, Valid_G, Valid_Y, FSet]