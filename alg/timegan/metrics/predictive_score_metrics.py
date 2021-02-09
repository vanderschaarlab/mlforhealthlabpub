'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Predictive_Score_Metrics
- Use Post-hoc RNN to predict one-step ahead (last feature)

Inputs
- dataX: Original data
- dataX_hat: Synthetic ata

Outputs
- Predictive Score (MAE of one-step ahead prediction)
'''

#%% Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

#%% Post-hoc RNN one-step ahead predictor

def predictive_score_metrics (dataX, dataX_hat):
  
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
     
    # Network Parameters
    hidden_dim = max(int(data_dim/2),1)
    iterations = 5000
    batch_size = 128
    
    #%% input place holders
    
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, data_dim-1], name = "myinput_x")
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
    Y = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, 1], name = "myinput_y")
    
    #%% builde a RNN discriminator network 
    
    def predictor (X, T):
      
        with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
            
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
                    
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
            
            Y_hat_Final = tf.nn.sigmoid(Y_hat)
            
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
        return Y_hat_Final, d_vars
    
    #%% Functions
    # Variables
    Y_pred, d_vars = predictor(X, T)
        
    # Loss for the predictor
    D_loss = tf.losses.absolute_difference(Y, Y_pred)
    
    # optimizer
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
        
    #%% Sessions    

    # Session start
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training using Synthetic dataset
    for itt in range(iterations):
          
        # Batch setting
        idx = np.random.permutation(len(dataX_hat))
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX_hat[i][:-1,:(data_dim-1)] for i in train_idx)
        T_mb = list(dataT[i]-1 for i in train_idx)
        Y_mb = list(np.reshape(dataX_hat[i][1:,(data_dim-1)],[len(dataX_hat[i][1:,(data_dim-1)]),1]) for i in train_idx)        
          
        # Train discriminator
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})            
        
        #%% Checkpoints
#        if itt % 500 == 0:
#            print("[step: {}] loss - d loss: {}".format(itt, np.sqrt(np.round(step_d_loss,4))))
    
    #%% Use Original Dataset to test
    
    # Make Batch with Original Data
    idx = np.random.permutation(len(dataX_hat))
    train_idx = idx[:No]     
    
    X_mb = list(dataX[i][:-1,:(data_dim-1)] for i in train_idx)
    T_mb = list(dataT[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(dataX[i][1:,(data_dim-1)], [len(dataX[i][1:,(data_dim-1)]),1]) for i in train_idx)
    
    # Predict Fugure
    pred_Y_curr = sess.run(Y_pred, feed_dict={X: X_mb, T: T_mb})
    
    # Compute MAE
    MAE_Temp = 0
    for i in range(No):
        MAE_Temp = MAE_Temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
    MAE = MAE_Temp / No
    
    return MAE
    