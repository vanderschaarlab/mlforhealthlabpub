'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

TGAN Function
- Use original data as training set to generater synthetic data (time-series)

Inputs
- Dataset
- Network Parameters

Outputs
- time-series synthetic data
'''

#%% Necessary Packages
import tensorflow as tf
import numpy as np

#%% Min Max Normalizer

def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val

#%% Start TGAN function (Input: Original data, Output: Synthetic Data)

def tgan (dataX, parameters):
  
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
        
    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0
     
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1
    
    #%% input place holders
    
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")
    
    #%% Basic RNN Cell
          
    def rnn_cell(module_name):
      # GRU
        if (module_name == 'gru'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM
        elif (module_name == 'lstm'):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM Layer Normalization
        elif (module_name == 'lstmLN'):
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        return rnn_cell
      
        
    #%% build a RNN embedding network      
    
    def embedder (X, T):      
      
        with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
            
            H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return H
      
    ##### Recovery
    
    def recovery (H, T):      
      
        with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
              
            r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
            
            X_tilde = tf.contrib.layers.fully_connected(r_outputs, data_dim, activation_fn=tf.nn.sigmoid) 

        return X_tilde
    
    
    
    #%% build a RNN generator network
    
    def generator (Z, T):      
      
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
            
            E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return E
      
    def supervisor (H, T):      
      
        with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
            
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers-1)])
                
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
            
            S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return S
      
      
      
    #%% builde a RNN discriminator network 
    
    def discriminator (H, T):
      
        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
            
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
            
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
    
        return Y_hat   
    
    
    #%% Random vector generation
    def random_generator (batch_size, z_dim, T_mb, Max_Seq_Len):
      
        Z_mb = list()
        
        for i in range(batch_size):
            
            Temp = np.zeros([Max_Seq_Len, z_dim])
            
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        
            Temp[:T_mb[i],:] = Temp_Z
            
            Z_mb.append(Temp_Z)
      
        return Z_mb
    
    #%% Functions
    
    # Embedder Networks
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    
    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)
    
    # Synthetic data
    X_hat = recovery(H_hat, T)
    
    # Discriminator
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)     
    Y_fake_e = discriminator(E_hat, T)
    
    # Variables        
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
    # Loss for the discriminator
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
    # Loss for the generator
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
    # 2. Supervised loss
    G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,1:,:])
    
    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(np.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(np.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
    G_loss_V = G_loss_V1 + G_loss_V2
    
    # Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
    # Loss for the embedder network
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10*tf.sqrt(E_loss_T0)
    E_loss = E_loss0  + 0.1*G_loss_S
    
    # optimizer
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
    #%% Sessions    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #%% Embedding Learning
    
    print('Start Embedding Network Training')
    
    for itt in range(iterations):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)
            
        # Train embedder        
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )        
            
    print('Finish Embedding Network Training')
    
    #%% Training Supervised Loss First
    
    print('Start Training with Supervised Loss Only')
    
    for itt in range(iterations):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)        
        
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
        
        # Train generator       
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
                           
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
                
    print('Finish Training with Supervised Loss Only')
    
    #%% Joint Training
    
    print('Start Joint Training')
    
    # Training step
    for itt in range(iterations):
      
        # Generator Training
        for kk in range(2):
          
            # Batch setting
            idx = np.random.permutation(No)
            train_idx = idx[:batch_size]     
            
            X_mb = list(dataX[i] for i in train_idx)
            T_mb = list(dataT[i] for i in train_idx)
            
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
              
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            
            # Train embedder        
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
        #%% Discriminator Training
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
        
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)
        
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
            
        
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        # Train discriminator
        
        if (check_d_loss > 0.15):        
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        #%% Checkpoints
        if itt % 1000 == 0:
            print('step: '+ str(itt) + 
                  ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                  ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
                  ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                  ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
   
    
    print('Finish Joint Training')
    
    #%% Final Outputs
    
    Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    
    X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})    
    
    #%% List of the final outputs
    
    dataX_hat = list()
    
    for i in range(No):
        Temp = X_hat_curr[i,:dataT[i],:]
        dataX_hat.append(Temp)
        
    # Renormalization
    if (Normalization_Flag == 1):
        dataX_hat = dataX_hat * max_val
        dataX_hat = dataX_hat + min_val
    
    return dataX_hat
    