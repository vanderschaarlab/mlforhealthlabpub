"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

supervised_models.py
- Train supervised model and return predictions on the testing data

(1) logit: logistic regression
(2) xgb_model: XGBoost model
(3) mlp: multi-layer perceptrons
"""

# Necessary packages
import numpy as np

from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from vime_utils import convert_matrix_to_vector, convert_vector_to_matrix

#%% 
def logit(x_train, y_train, x_test):
  """Logistic Regression.
  
  Args: 
    - x_train, y_train: training dataset
    - x_test: testing feature
    
  Returns:
    - y_test_hat: predicted values for x_test
  """
  # Convert labels into proper format
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)  
  
  # Define and fit model on training dataset
  model = LogisticRegression()
  model.fit(x_train, y_train)
  
  # Predict on x_test
  y_test_hat = model.predict_proba(x_test) 
  
  return y_test_hat

#%% 
def xgb_model(x_train, y_train, x_test):
  """XGBoost.
  
  Args: 
    - x_train, y_train: training dataset
    - x_test: testing feature
    
  Returns:
    - y_test_hat: predicted values for x_test
  """  
  # Convert labels into proper format
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)  
  
  # Define and fit model on training dataset
  model = xgb.XGBClassifier()
  model.fit(x_train, y_train)
  
  # Predict on x_test
  y_test_hat = model.predict_proba(x_test) 
  
  return y_test_hat

  
#%% 
def mlp(x_train, y_train, x_test, parameters):
  """Multi-layer perceptron (MLP).
  
  Args: 
    - x_train, y_train: training dataset
    - x_test: testing feature
    - parameters: hidden_dim, epochs, activation, batch_size
    
  Returns:
    - y_test_hat: predicted values for x_test
  """  
  
  # Convert labels into proper format
  if len(y_train.shape) == 1:
    y_train = convert_vector_to_matrix(y_train)
    
  # Divide training and validation sets (9:1)
  idx = np.random.permutation(len(x_train[:, 0]))
  train_idx = idx[:int(len(idx)*0.9)]
  valid_idx = idx[int(len(idx)*0.9):]
  
  # Validation set
  x_valid = x_train[valid_idx, :]
  y_valid = y_train[valid_idx, :]
  
  # Training set
  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx, :]  
  
  # Reset the graph
  K.clear_session()
    
  # Define network parameters
  hidden_dim = parameters['hidden_dim']
  epochs_size = parameters['epochs']
  act_fn = parameters['activation']
  batch_size = parameters['batch_size']
  
  # Define basic parameters
  data_dim = len(x_train[0, :])
  label_dim = len(y_train[0, :])

  # Build model
  model = Sequential()
  model.add(Dense(hidden_dim, input_dim = data_dim, activation = act_fn))
  model.add(Dense(hidden_dim, activation = act_fn))  
  model.add(Dense(label_dim, activation = 'softmax'))
  
  model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
                metrics = ['acc'])
  
  es = EarlyStopping(monitor='val_loss', mode = 'min', 
                     verbose = 1, restore_best_weights=True, patience=50)
  
  # Fit model on training dataset
  model.fit(x_train, y_train, validation_data = (x_valid, y_valid), 
            epochs = epochs_size, batch_size = batch_size, 
            verbose = 0, callbacks=[es])
  
  # Predict on x_test
  y_test_hat = model.predict(x_test)
  
  return y_test_hat