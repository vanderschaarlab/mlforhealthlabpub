"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_semi.py
- Semi-supervised learning parts of the VIME framework
- Using both labeled and unlabeled data to train the predictor with the help of trained encoder
"""

# Necessary packages
import keras
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers

from vime_utils import mask_generator, pretext_generator


def vime_semi(x_train, y_train, x_unlab, x_test, parameters, 
              p_m, K, beta, file_name):
  """Semi-supervied learning part in VIME.
  
  Args:
    - x_train, y_train: training dataset
    - x_unlab: unlabeled dataset
    - x_test: testing features
    - parameters: network parameters (hidden_dim, batch_size, iterations)
    - p_m: corruption probability
    - K: number of augmented samples
    - beta: hyperparameter to control supervised and unsupervised loss
    - file_name: saved filed name for the encoder function
    
  Returns:
    - y_test_hat: prediction on x_test
  """
      
  # Network parameters
  hidden_dim = parameters['hidden_dim']
  act_fn = tf.nn.relu
  batch_size = parameters['batch_size']
  iterations = parameters['iterations']

  # Basic parameters
  data_dim = len(x_train[0, :])
  label_dim = len(y_train[0, :])
  
  # Divide training and validation sets (9:1)
  idx = np.random.permutation(len(x_train[:, 0]))
  train_idx = idx[:int(len(idx)*0.9)]
  valid_idx = idx[int(len(idx)*0.9):]
  
  x_valid = x_train[valid_idx, :]
  y_valid = y_train[valid_idx, :]
  
  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx, :]  

  # Input placeholder
  # Labeled data
  x_input = tf.placeholder(tf.float32, [None, data_dim])
  y_input = tf.placeholder(tf.float32, [None, label_dim])
  
  # Augmented unlabeled data
  xu_input = tf.placeholder(tf.float32, [None, None, data_dim])
  
  ## Predictor
  def predictor(x_input):
    """Returns prediction.
    
    Args: 
      - x_input: input feature
      
    Returns:
      - y_hat_logit: logit prediction
      - y_hat: prediction
    """
    with tf.variable_scope('predictor', reuse=tf.AUTO_REUSE):     
      # Stacks multi-layered perceptron
      inter_layer = contrib_layers.fully_connected(x_input, 
                                                   hidden_dim, 
                                                   activation_fn=act_fn)
      inter_layer = contrib_layers.fully_connected(inter_layer, 
                                                   hidden_dim, 
                                                   activation_fn=act_fn)

      y_hat_logit = contrib_layers.fully_connected(inter_layer, 
                                                   label_dim, 
                                                   activation_fn=None)
      y_hat = tf.nn.softmax(y_hat_logit)

    return y_hat_logit, y_hat

  # Build model
  y_hat_logit, y_hat = predictor(x_input)    
  yv_hat_logit, yv_hat = predictor(xu_input)
  
  # Defin losses
  # Supervised loss
  y_loss = tf.losses.softmax_cross_entropy(y_input, y_hat_logit)  
  # Unsupervised loss
  yu_loss = tf.reduce_mean(tf.nn.moments(yv_hat_logit, axes = 0)[1])
  
  # Define variables
  p_vars = [v for v in tf.trainable_variables() \
            if v.name.startswith('predictor')]    
  # Define solver
  solver = tf.train.AdamOptimizer().minimize(y_loss + \
                                 beta * yu_loss, var_list=p_vars)

  # Load encoder from self-supervised model
  encoder = keras.models.load_model(file_name)
  
  # Encode validation and testing features
  x_valid = encoder.predict(x_valid)  
  x_test = encoder.predict(x_test)

  # Start session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # Setup early stopping procedure
  class_file_name = './save_model/class_model.ckpt'
  saver = tf.train.Saver(p_vars)
    
  yv_loss_min = 1e10
  yv_loss_min_idx = -1
  
  # Training iteration loop
  for it in range(iterations):

    # Select a batch of labeled data
    batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
    x_batch = x_train[batch_idx, :]
    y_batch = y_train[batch_idx, :]    
    
    # Encode labeled data
    x_batch = encoder.predict(x_batch)  
    
    # Select a batch of unlabeled data
    batch_u_idx = np.random.permutation(len(x_unlab[:, 0]))[:batch_size]
    xu_batch_ori = x_unlab[batch_u_idx, :]
    
    # Augment unlabeled data
    xu_batch = list()
    
    for rep in range(K):      
      # Mask vector generation
      m_batch = mask_generator(p_m, xu_batch_ori)
      # Pretext generator
      _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)
      
      # Encode corrupted samples
      xu_batch_temp = encoder.predict(xu_batch_temp)
      xu_batch = xu_batch + [xu_batch_temp]
    # Convert list to matrix
    xu_batch = np.asarray(xu_batch)

    # Train the model
    _, y_loss_curr = sess.run([solver, y_loss], 
                              feed_dict={x_input: x_batch, y_input: y_batch, 
                                         xu_input: xu_batch})  
    # Current validation loss
    yv_loss_curr = sess.run(y_loss, feed_dict={x_input: x_valid, 
                                               y_input: y_valid})
  
    if it % 100 == 0:
      print('Iteration: ' + str(it) + '/' + str(iterations) + 
            ', Current loss: ' + str(np.round(yv_loss_curr, 4)))      
      
    # Early stopping & Best model save
    if yv_loss_min > yv_loss_curr:
      yv_loss_min = yv_loss_curr
      yv_loss_min_idx = it

      # Saves trained model
      saver.save(sess, class_file_name)
      
    if yv_loss_min_idx + 100 < it:
      break

  #%% Restores the saved model
  imported_graph = tf.train.import_meta_graph(class_file_name + '.meta')
  
  sess = tf.Session()
  imported_graph.restore(sess, class_file_name)
    
  # Predict on x_test
  y_test_hat = sess.run(y_hat, feed_dict={x_input: x_test})
  
  return y_test_hat
