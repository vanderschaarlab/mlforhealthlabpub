# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sklearn import metrics
  
#%%
def Logit_R (Train_X, Train_Y, Test_X, Test_Y):
  
  X_dim = len(Train_X[0,:])
  y_dim = 1
  
  mb_size = 128  
  
  X = tf.placeholder(tf.float32, shape = [None, X_dim])
  y = tf.placeholder(tf.float32, shape = [None, y_dim])
  
  def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)  
  
  G_W = tf.Variable(xavier_init([X_dim,1]))
  G_b = tf.Variable(tf.zeros(shape=[1]))

  theta_G = [G_W, G_b]

  #%% Function
  def generator(x):
    G1_logit = tf.matmul(x, G_W) + G_b
    G1_prob = tf.nn.sigmoid(G1_logit)
    
    return G1_logit, G1_prob
  
  G_logit, G_prob = generator(X)
  
  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logit, labels = y))
    
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  #%%
  
  # Sessions
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Iterations
  for it in range(50000):

    idx = np.random.permutation(len(Train_X))
    train_idx = idx[:mb_size]
                
    X_mb = Train_X[train_idx,:]
    y_mb = np.reshape(Train_Y[train_idx], [mb_size,1])
            
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {X: X_mb, y: y_mb})
  
  #%%
  Prediction = sess.run([G_prob], feed_dict = {X: Test_X})[0]

  AUC = metrics.roc_auc_score(Test_Y, Prediction)
  APR = metrics.average_precision_score(Test_Y, Prediction)
  
  if (AUC < 0.5):
    AUC = 0.5
  if (APR < np.mean(Test_Y)):
    APR = np.mean(Test_Y)
    
  return Prediction, AUC, APR
