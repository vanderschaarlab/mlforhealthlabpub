"""Anonymization through Data Synthesis using Generative Adversarial Networks:
A harmonizing advancement for AI in medicine (ADS-GAN) Codebase.

Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar, 
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine," 
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
Last updated Date: December 22th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
compute_wd.py
- Compare Wasserstein distance between original data and synthetic data
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

def compute_wd (orig_data, synth_data, params):
  """Compare Wasserstein distance between original data and synthetic data.
  
  Args:
    orig_data: original data
    synth_data: synthetically generated data
    params: Network parameters
      mb_size: mini-batch size
      h_dim: hidden state dimension
      iterations: training iterations
      
  Returns:
    WD_value: Wasserstein distance
  """
  
  # Preprocess the data
  orig_data = np.asarray(orig_data)
  synth_data = np.asarray(synth_data)
    
  no, x_dim = np.shape(orig_data)
    
  # Divide train / test
  orig_data_train = orig_data[:int(no/2),:]
  orig_data_test = orig_data[int(no/2):,:]
    
  synth_data_train = synth_data[:int(no/2),:]
  synth_data_test = synth_data[int(no/2):,:]
    
  #%% Parameters
  # Batch size    
  mb_size = params['mb_size']
  # Hidden unit dimensions
  h_dim = int(params['h_dim']/2)
  # Train iterations
  iterations = params['iterations']
    
  #%% Necessary Functions

  # Xavier Initialization Definition
  def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)   
        
  # Sample from the real data
  def sample_X(m, n):
    return np.random.permutation(m)[:n]  
     
  #%% Placeholder
  X = tf.placeholder(tf.float32, shape = [None, x_dim])   
  X_hat = tf.placeholder(tf.float32, shape = [None, x_dim])  
      
  #%% Discriminator
  # Discriminator
  D_W1 = tf.Variable(xavier_init([x_dim, h_dim]))
  D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    
  D_W2 = tf.Variable(xavier_init([h_dim,1]))
  D_b2 = tf.Variable(tf.zeros(shape=[1]))

  theta_D = [D_W1, D_W2, D_b1, D_b2]
    
  def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = (tf.matmul(D_h1, D_W2) + D_b2)
    return out
    
  # Structure
  D_real = discriminator(X)
  D_fake = discriminator(X_hat) 
    
  D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    
  D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
              .minimize(-D_loss, var_list=theta_D))
    
  clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]
            
  #%%
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
        
  # Iterations
  for it in tqdm(range(iterations)):                
            
    X_idx = sample_X(int(no/2),mb_size)        
    X_mb = orig_data_train[X_idx,:]   
    X_hat_mb = synth_data_train[X_idx,:]  
            
    _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict = {X: X_mb, X_hat: X_hat_mb})
       
  #%% Test
  WD_value = sess.run([D_loss], feed_dict = {X: orig_data_test, X_hat: synth_data_test})
    
  return WD_value[0]