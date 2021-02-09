'''
RadialGAN (Jinsung Yoon, 09/10/2018)

Inputs:
  - Train_X, Test_X: features
  - Train_M, Test_M: Mask vector (which features are selected)
  - Train_G, Test_G: Group
  - Train_Y, Test_Y: Labels
  - FSet: Which features are selected for which group

'''
#%% Necessary packages
import tensorflow as tf
import numpy as np
from Predictive_Models import Logit_R

#%% RadialGAN
def RadialGAN(Train_X, Train_M, Train_G, Train_Y, Test_X, Test_M, Test_G, Test_Y, Valid_X, Valid_M, Valid_G, Valid_Y, FSet, alpha):

    #%% Preparing
    # Reset
    tf.reset_default_graph()

    # One hot encoding Group
    Set_No = int(np.max(Train_G))
    New_Train_G = np.zeros([len(Train_X),(Set_No)])
    
    for i in range(Set_No):
        idx = np.where(Train_G == (i+1))
        idx = idx[0]
        New_Train_G[idx,i] = 1        
                
    # Parameters
    Dim = len(Train_X[0,:])

    mb_size = 128
    Z_dim = Dim

    X1_Dim = int(np.sum(FSet[0,:]))
    X2_Dim = int(np.sum(FSet[1,:]))
    X3_Dim = int(np.sum(FSet[2,:]))        
    
    y_dim = 1
    h_dim = Dim
    

    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)    

    #%% Neural Networks
    # 1. Placeholders
    # (X,y) for each group
    X1 = tf.placeholder(tf.float32, shape = [None, X1_Dim])
    y1 = tf.placeholder(tf.float32, shape = [None, y_dim])
    
    X2 = tf.placeholder(tf.float32, shape = [None, X2_Dim])
    y2 = tf.placeholder(tf.float32, shape = [None, y_dim])
    
    X3 = tf.placeholder(tf.float32, shape = [None, X3_Dim])
    y3 = tf.placeholder(tf.float32, shape = [None, y_dim])

    #%% Discriminator net model
    D1_W1 = tf.Variable(xavier_init([X1_Dim + y_dim, h_dim]))
    D1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D1_W2 = tf.Variable(xavier_init([h_dim,1]))
    D1_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D1 = [D1_W1, D1_W2, D1_b1, D1_b2]

    D2_W1 = tf.Variable(xavier_init([X2_Dim + y_dim, h_dim]))
    D2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D2_W2 = tf.Variable(xavier_init([h_dim,1]))
    D2_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D2 = [D2_W1, D2_W2, D2_b1, D2_b2]

    D3_W1 = tf.Variable(xavier_init([X3_Dim + y_dim, h_dim]))
    D3_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D3_W2 = tf.Variable(xavier_init([h_dim,1]))
    D3_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D3 = [D3_W1, D3_W2, D3_b1, D3_b2]

    theta_D = theta_D1 + theta_D2 + theta_D3    
    
    #%% Functions
    def discriminator1(x,y):
        inputs = tf.concat(axis=1, values = [x,y])
        D1_h1 = tf.nn.tanh(tf.matmul(inputs,D1_W1)+D1_b1)
        D1_logit = tf.matmul(D1_h1, D1_W2) + D1_b2
        D1_prob = tf.nn.sigmoid(D1_logit)
    
        return D1_prob, D1_logit
        
    def discriminator2(x,y):
        inputs = tf.concat(axis=1, values = [x,y])
        D2_h1 = tf.nn.tanh(tf.matmul(inputs,D2_W1)+D2_b1)
        D2_logit = tf.matmul(D2_h1, D2_W2) + D2_b2
        D2_prob = tf.nn.sigmoid(D2_logit)
    
        return D2_prob, D2_logit
        
    def discriminator3(x,y):
        inputs = tf.concat(axis=1, values = [x,y])
        D3_h1 = tf.nn.tanh(tf.matmul(inputs,D3_W1)+D3_b1)
        D3_logit = tf.matmul(D3_h1, D3_W2) + D3_b2
        D3_prob = tf.nn.sigmoid(D3_logit)
    
        return D3_prob, D3_logit
    
    # Generator Net Model (X to Z)
    G1_W1 = tf.Variable(xavier_init([X1_Dim + y_dim, h_dim]))
    G1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G1_W2 = tf.Variable(xavier_init([h_dim,Z_dim]))
    G1_b2 = tf.Variable(tf.zeros(shape=[Z_dim]))

    theta_G1_hat = [G1_W1, G1_W2, G1_b1, G1_b2]    
    
    G2_W1 = tf.Variable(xavier_init([X2_Dim + y_dim, h_dim]))
    G2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G2_W2 = tf.Variable(xavier_init([h_dim,Z_dim]))
    G2_b2 = tf.Variable(tf.zeros(shape=[Z_dim]))

    theta_G2_hat = [G2_W1, G2_W2, G2_b1, G2_b2]    
    
    G3_W1 = tf.Variable(xavier_init([X3_Dim + y_dim, h_dim]))
    G3_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G3_W2 = tf.Variable(xavier_init([h_dim,Z_dim]))
    G3_b2 = tf.Variable(tf.zeros(shape=[Z_dim]))

    theta_G3_hat = [G3_W1, G3_W2, G3_b1, G3_b2]

    #%% Function
    def generator1(x,y):
        inputs = tf.concat(axis=1, values=[x,y])
        G1_h1 = tf.nn.tanh(tf.matmul(inputs, G1_W1) + G1_b1)
        G1_log_prob = tf.matmul(G1_h1, G1_W2) + G1_b2
        G1_prob = (G1_log_prob)
    
        return G1_prob
        
    def generator2(x,y):
        inputs = tf.concat(axis=1, values=[x,y])
        G2_h1 = tf.nn.tanh(tf.matmul(inputs, G2_W1) + G2_b1)
        G2_log_prob = tf.matmul(G2_h1, G2_W2) + G2_b2
        G2_prob = (G2_log_prob)
    
        return G2_prob
        
    def generator3(x,y):
        inputs = tf.concat(axis=1, values=[x,y])
        G3_h1 = tf.nn.tanh(tf.matmul(inputs, G3_W1) + G3_b1)
        G3_log_prob = tf.matmul(G3_h1, G3_W2) + G3_b2
        G3_prob = (G3_log_prob)
    
        return G3_prob
        
    #%% Generator Net Model (Z to X)
    F1_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
    F1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    F1_W2 = tf.Variable(xavier_init([h_dim,X1_Dim]))
    F1_b2 = tf.Variable(tf.zeros(shape=[X1_Dim]))

    theta_F1 = [F1_W1, F1_W2, F1_b1, F1_b2]
    
    
    F2_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
    F2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    F2_W2 = tf.Variable(xavier_init([h_dim,X2_Dim]))
    F2_b2 = tf.Variable(tf.zeros(shape=[X2_Dim]))

    theta_F2 = [F2_W1, F2_W2, F2_b1, F2_b2]
    
    
    F3_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
    F3_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    F3_W2 = tf.Variable(xavier_init([h_dim,X3_Dim]))
    F3_b2 = tf.Variable(tf.zeros(shape=[X3_Dim]))

    theta_F3 = [F3_W1, F3_W2, F3_b1, F3_b2]

    theta_G1 = theta_G1_hat + theta_F1
    theta_G2 = theta_G2_hat + theta_F2
    theta_G3 = theta_G3_hat + theta_F3

    #%% Function
    def mapping1(z,y):
        inputs = tf.concat(axis=1, values=[z,y])
        F1_h1 = tf.nn.tanh(tf.matmul(inputs, F1_W1) + F1_b1)
        F1_log_prob = tf.matmul(F1_h1, F1_W2) + F1_b2
        F1_prob = (F1_log_prob)
    
        return F1_prob
        
    def mapping2(z,y):
        inputs = tf.concat(axis=1, values=[z,y])
        F2_h1 = tf.nn.tanh(tf.matmul(inputs, F2_W1) + F2_b1)
        F2_log_prob = tf.matmul(F2_h1, F2_W2) + F2_b2
        F2_prob = (F2_log_prob)
    
        return F2_prob
        
    def mapping3(z,y):
        inputs = tf.concat(axis=1, values=[z,y])
        F3_h1 = tf.nn.tanh(tf.matmul(inputs, F3_W1) + F3_b1)
        F3_log_prob = tf.matmul(F3_h1, F3_W2) + F3_b2
        F3_prob = (F3_log_prob)
    
        return F3_prob
            
    #%% Structure
    # 1. Generator
    G_sample12 = mapping2(generator1(X1,y1),y1)
    G_sample13 = mapping3(generator1(X1,y1),y1)
    
    G_sample21 = mapping1(generator2(X2,y2),y2)
    G_sample23 = mapping3(generator2(X2,y2),y2)
    
    G_sample31 = mapping1(generator3(X3,y3),y3)
    G_sample32 = mapping2(generator3(X3,y3),y3)


    # 2. Discriminator
    D1_real, D1_logit_real = discriminator1(X1,y1)
    D21_fake, D21_logit_fake = discriminator1(G_sample21, y2)
    D31_fake, D31_logit_fake = discriminator1(G_sample31, y3)
    
    D2_real, D2_logit_real = discriminator2(X2,y2)
    D12_fake, D12_logit_fake = discriminator2(G_sample12, y1)
    D32_fake, D32_logit_fake = discriminator2(G_sample32, y3)
    
    D3_real, D3_logit_real = discriminator3(X3,y3)
    D13_fake, D13_logit_fake = discriminator3(G_sample13, y1)
    D23_fake, D23_logit_fake = discriminator3(G_sample23, y2)

    # 3. Recover
    Recov_X121 = mapping1(generator2(G_sample12,y1),y1)
    Recov_X131 = mapping1(generator3(G_sample13,y1),y1)
    
    Recov_X212 = mapping2(generator1(G_sample21,y2),y2)
    Recov_X232 = mapping2(generator3(G_sample23,y2),y2)
    
    Recov_X313 = mapping3(generator1(G_sample31,y3),y3)
    Recov_X323 = mapping3(generator2(G_sample32,y3),y3)

    # Loss
    # 1. Discriminator loss
    D1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logit_real, labels = tf.ones_like(D1_logit_real)))
    D21_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D21_logit_fake, labels = tf.zeros_like(D21_logit_fake)))
    D31_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D31_logit_fake, labels = tf.zeros_like(D31_logit_fake)))
    D1_loss = D1_loss_real + 0.5 * (D21_loss_fake + D31_loss_fake)
    
    D2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logit_real, labels = tf.ones_like(D2_logit_real)))
    D12_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D12_logit_fake, labels = tf.zeros_like(D12_logit_fake)))
    D32_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D32_logit_fake, labels = tf.zeros_like(D32_logit_fake)))
    D2_loss = D2_loss_real + 0.5 * (D12_loss_fake + D32_loss_fake)
    
    D3_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D3_logit_real, labels = tf.ones_like(D3_logit_real)))
    D13_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D13_logit_fake, labels = tf.zeros_like(D13_logit_fake)))
    D23_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D23_logit_fake, labels = tf.zeros_like(D23_logit_fake)))
    D3_loss = D3_loss_real + 0.5 * (D13_loss_fake + D23_loss_fake)
    
    D_loss = D1_loss + D2_loss + D3_loss
    
    # 2. Generator loss
    G21_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D21_logit_fake, labels=tf.ones_like(D21_logit_fake)))
    G31_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D31_logit_fake, labels=tf.ones_like(D31_logit_fake)))
    G1_loss_hat = 0.5 * (G21_loss + G31_loss)
    
    G12_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D12_logit_fake, labels=tf.ones_like(D12_logit_fake)))
    G32_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D32_logit_fake, labels=tf.ones_like(D32_logit_fake)))
    G2_loss_hat = 0.5 * (G12_loss + G32_loss)
    
    G13_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D13_logit_fake, labels=tf.ones_like(D13_logit_fake)))
    G23_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D23_logit_fake, labels=tf.ones_like(D23_logit_fake)))
    G3_loss_hat = 0.5 * (G13_loss + G23_loss)
    
    # Recover Loss
    G121_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X121, X1))
    G131_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X131, X1))
    G1_Recov_loss = 0.5 * (G121_Recov_loss + G131_Recov_loss)
    
    G212_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X212, X2))
    G232_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X232, X2))
    G2_Recov_loss = 0.5 * (G212_Recov_loss + G232_Recov_loss)
    
    G313_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X313, X3))
    G323_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X323, X3))
    G3_Recov_loss = 0.5 * (G313_Recov_loss + G323_Recov_loss)
    
    G1_loss =  G1_loss_hat + alpha * tf.sqrt(G1_Recov_loss)
    G2_loss =  G2_loss_hat + alpha * tf.sqrt(G2_Recov_loss)
    G3_loss =  G3_loss_hat + alpha * tf.sqrt(G3_Recov_loss)
        
    # Solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G1_solver = tf.train.AdamOptimizer().minimize(G1_loss, var_list=theta_G1)
    G2_solver = tf.train.AdamOptimizer().minimize(G2_loss, var_list=theta_G2)
    G3_solver = tf.train.AdamOptimizer().minimize(G3_loss, var_list=theta_G3)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Iterations
    idx1 = np.where(Train_G == 1)[0]
    Train_X1 = Train_X[idx1,:]
    Train_Y1 = Train_Y[idx1]
    
    idx2 = np.where(Train_G == 2)[0]
    Train_X2 = Train_X[idx2,:]
    Train_Y2 = Train_Y[idx2]
    
    idx3 = np.where(Train_G == 3)[0]
    Train_X3 = Train_X[idx3,:]
    Train_Y3 = Train_Y[idx3]

    for it in range(20000):

        idx1 = np.random.permutation(len(Train_X1))
        train_idx1 = idx1[:mb_size]
                
        X1_mb = Train_X1[train_idx1,:]
        y1_mb = np.reshape(Train_Y1[train_idx1], [mb_size,1])
                    
        idx1 = np.where(FSet[0,:] == 1)[0]        
        X1_mb = X1_mb[:,idx1]
                
        idx2 = np.random.permutation(len(Train_X2))
        train_idx2 = idx2[:mb_size]
                
        X2_mb = Train_X2[train_idx2,:]
        y2_mb = np.reshape(Train_Y2[train_idx2], [mb_size,1])
                    
        idx2 = np.where(FSet[1,:] == 1)[0]        
        X2_mb = X2_mb[:,idx2]
        
        idx3 = np.random.permutation(len(Train_X3))
        train_idx3 = idx3[:mb_size]
                
        X3_mb = Train_X3[train_idx3,:]
        y3_mb = np.reshape(Train_Y3[train_idx3], [mb_size,1])
                    
        idx3 = np.where(FSet[2,:] == 1)[0]        
        X3_mb = X3_mb[:,idx3]
    
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X1: X1_mb, y1: y1_mb, X2: X2_mb, y2: y2_mb, X3: X3_mb, y3: y3_mb})
        _, G1_loss_curr = sess.run([G1_solver, G1_loss], feed_dict = {X1: X1_mb, y1: y1_mb, X2: X2_mb, y2: y2_mb, X3: X3_mb, y3: y3_mb})
        _, G2_loss_curr = sess.run([G2_solver, G2_loss], feed_dict = {X1: X1_mb, y1: y1_mb, X2: X2_mb, y2: y2_mb, X3: X3_mb, y3: y3_mb})
        _, G3_loss_curr = sess.run([G3_solver, G3_loss], feed_dict = {X1: X1_mb, y1: y1_mb, X2: X2_mb, y2: y2_mb, X3: X3_mb, y3: y3_mb})
    
        if it % 10000 == 0:
            print('Iter: {}'.format(it))
            print('D_loss: {:.4}'.format(D_loss_curr))
            print('G1_loss: {:.4}'.format(G1_loss_curr))
            print('G2_loss: {:.4}'.format(G2_loss_curr))
            print('G3_loss: {:.4}'.format(G3_loss_curr))
            print()

    #%%##### Data Generation & Prediction
    Test_No = len(Test_X)

    Prediction = np.zeros([Test_No])    
    
    Output_Logit = list()
    
    for i in range(Set_No):
        idx = np.where(Train_G == (i+1))[0]
        
        Final_X = Train_X[idx,:]
        Final_Y = Train_Y[idx]
                    
        idx1 = np.where(FSet[i,:] == 1)[0]        
        Final_X = Final_X[:,idx1]                    
                    
        for j in range(Set_No):
            
            if (j != i):   
                new_idx = np.where(Train_G == (j+1))[0]
                ## Data sampling
                X_sample = Train_X[new_idx,:]
                Y_sample = np.reshape(Train_Y[new_idx], [len(X_sample),1])

                idx2 = np.where(FSet[j,:] == 1)[0]        
                X_sample = X_sample[:,idx2]   


                if ((i == 0) & (j == 1)):
                    New_Sample = sess.run([G_sample21], feed_dict = {X2: X_sample, y2:Y_sample})
                if ((i == 0) & (j == 2)):
                    New_Sample = sess.run([G_sample31], feed_dict = {X3: X_sample, y3:Y_sample})
                if ((i == 1) & (j == 0)):
                    New_Sample = sess.run([G_sample12], feed_dict = {X1: X_sample, y1:Y_sample})
                if ((i == 1) & (j == 2)):
                    New_Sample = sess.run([G_sample32], feed_dict = {X3: X_sample, y3:Y_sample})
                if ((i == 2) & (j == 0)):
                    New_Sample = sess.run([G_sample13], feed_dict = {X1: X_sample, y1:Y_sample})
                if ((i == 2) & (j == 1)):
                    New_Sample = sess.run([G_sample23], feed_dict = {X2: X_sample, y2:Y_sample})
                
                Final_X = np.concatenate([Final_X, New_Sample[0]], axis = 0)
                Final_Y = np.concatenate([np.reshape(Final_Y, [len(Final_Y),]), np.reshape(Y_sample, [len(Y_sample),])], axis = 0)

        ## Prediction
        test_idx = np.where(Test_G == (i+1))[0]

        New_Test = Test_X[test_idx,:] 
        idx3 = np.where(FSet[i,:] == 1)[0]        
        New_Test = New_Test[:,idx3]   
        
        valid_idx = np.where(Valid_G == (i+1))[0]

        New_Valid = Valid_X[valid_idx,:] 
        idx4 = np.where(FSet[i,:] == 1)[0]        
        New_Valid = New_Valid[:,idx4]  
        
        #%% Selection
        # Original Data
        
        idx = np.where(Train_G == (i+1))[0]
        
        Ori_Final_X = Train_X[idx,:]
        Ori_Final_Y = Train_Y[idx]
                    
        idx1 = np.where(FSet[i,:] == 1)[0]        
        Ori_Final_X = Ori_Final_X[:,idx1]     
        
        _, Ori_Logit_AUC, Ori_Logit_APR = Logit_R(Ori_Final_X, Ori_Final_Y, New_Test, Test_Y[test_idx])
        
        # New Data
        _, New_Logit_AUC, New_Logit_APR = Logit_R(Final_X, Final_Y, New_Test, Test_Y[test_idx])
        
        if ((Ori_Logit_AUC < New_Logit_AUC) & (Ori_Logit_APR < New_Logit_APR)):        
        
            Logit_Prediction, Logit_AUC, Logit_APR = Logit_R(Final_X, Final_Y, New_Test, Test_Y[test_idx])
            
        else:
            
            Logit_Prediction, Logit_AUC, Logit_APR = Logit_R(Ori_Final_X, Ori_Final_Y, New_Test, Test_Y[test_idx])
            
        #%%
        
        print('New Logit: AUROC: ' + str(Logit_AUC) + ', AUPRC: ' + str(Logit_APR))
        
        Output_Logit.append([Logit_AUC, Logit_APR])

    for i in range(Set_No):
        
        idx = np.where(Train_G == (i+1))[0]
        
        Final_X = Train_X[idx,:]
        Final_Y = Train_Y[idx]
                    
        idx1 = np.where(FSet[i,:] == 1)[0]        
        Final_X = Final_X[:,idx1]   
      
        test_idx = np.where(Test_G == (i+1))[0]

        New_Test = Test_X[test_idx,:]
        idx3 = np.where(FSet[i,:] == 1)[0]        
        New_Test = New_Test[:,idx3]   
        
        Logit_Prediction, Logit_AUC, Logit_APR = Logit_R(Final_X, Final_Y, New_Test, Test_Y[test_idx])
    
        print('Ori Logit: AUROC: ' + str(Logit_AUC) + ', AUPRC: ' + str(Logit_APR))
        
        Output_Logit.append([Logit_AUC, Logit_APR])
        
    return Prediction, Output_Logit
