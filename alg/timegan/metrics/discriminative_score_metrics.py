'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Discriminative_Score_Metrics
- Use Post-hoc RNN to classify Original data and Synthetic data

Inputs
- dataX: Original data
- dataX_hat: Synthetic ata

Outputs
- Discriminative Score (np.abs(Classification Accuracy - 0.5))

'''

#%% Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

#%% Post-hoc RNN Classifier 

def discriminative_score_metrics (dataX, dataX_hat):
  
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Compute Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
     
    # Network Parameters
    hidden_dim = max(int(data_dim/2),1)
    iterations = 2000
    batch_size = 128
    
    #%% input place holders
    # Features
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    X_hat = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x_hat")
    
    # Times
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")
    T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
    #%% builde a RNN classification network 
    
    def discriminator (X, T):
      
        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
            
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'cd_cell')
                    
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                
            # Logits
            Y_hat = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
            
            # Sigmoid output
            Y_hat_Final = tf.nn.sigmoid(Y_hat)
            
            # Variables
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
        return Y_hat, Y_hat_Final, d_vars
    
    #%% Train / Test Division
    def train_test_divide (dataX, dataX_hat, dataT):
      
        # Divide train/test index
        No = len(dataX)
        idx = np.random.permutation(No)
        train_idx = idx[:int(No*0.8)]
        test_idx = idx[int(No*0.8):]
        
        # Train and Test X
        trainX = [dataX[i] for i in train_idx]
        trainX_hat = [dataX_hat[i] for i in train_idx]
        
        testX = [dataX[i] for i in test_idx]
        testX_hat = [dataX_hat[i] for i in test_idx]
        
        # Train and Test T
        trainT = [dataT[i] for i in train_idx]
        testT = [dataT[i] for i in test_idx]
      
        return trainX, trainX_hat, testX, testX_hat, trainT, testT
    
    #%% Functions
    # Variables
    Y_real, Y_pred_real, d_vars = discriminator(X, T)
    Y_fake, Y_pred_fake, _ = discriminator(X_hat, T_hat)
        
    # Loss for the discriminator
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_real, labels = tf.ones_like(Y_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_fake, labels = tf.zeros_like(Y_fake)))
    D_loss = D_loss_real + D_loss_fake
    
    # optimizer
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
        
    #%% Sessions    

    # Start session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Train / Test Division
    trainX, trainX_hat, testX, testX_hat, trainT, testT = train_test_divide (dataX, dataX_hat, dataT)
    
    # Training step
    for itt in range(iterations):
          
        # Batch setting
        idx = np.random.permutation(len(trainX))
        train_idx = idx[:batch_size]     
            
        X_mb = list(trainX[i] for i in train_idx)
        T_mb = list(trainT[i] for i in train_idx)
        
        # Batch setting
        idx = np.random.permutation(len(trainX_hat))
        train_idx = idx[:batch_size]     
            
        X_hat_mb = list(trainX_hat[i] for i in train_idx)
        T_hat_mb = list(trainT[i] for i in train_idx)
          
        # Train discriminator
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
        
        #%% Checkpoints
#        if itt % 500 == 0:
#            print("[step: {}] loss - d loss: {}".format(itt, np.round(step_d_loss,4)))
    
    #%% Final Outputs (ontTesting set)
    
    Y_pred_real_curr, Y_pred_fake_curr = sess.run([Y_pred_real, Y_pred_fake], feed_dict={X: testX, T: testT, X_hat: testX_hat, T_hat: testT})
    
    Y_pred_final = np.squeeze(np.concatenate((Y_pred_real_curr, Y_pred_fake_curr), axis = 0))
    Y_label_final = np.concatenate((np.ones([len(Y_pred_real_curr),]), np.zeros([len(Y_pred_real_curr),])), axis = 0)
    
    #%% Accuracy
    Acc = accuracy_score(Y_label_final, Y_pred_final>0.5)
    
    Disc_Score = np.abs(0.5-Acc)
    
    return Disc_Score
    