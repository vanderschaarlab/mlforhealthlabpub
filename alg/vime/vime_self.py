"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_self.py
- Self-supervised learning parts of the VIME framework
- Using unlabeled data to train the encoder
"""

# Necessary packages
from keras.layers import Input, Dense
from keras.models import Model
from keras import models

from vime_utils import mask_generator, pretext_generator


def vime_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in VIME.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
    
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  
  # Build model  
  inputs = Input(shape=(dim,))
  # Encoder  
  h = Dense(int(dim), activation='relu')(inputs)  
  # Mask estimator
  output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h)  
  # Feature estimator
  output_2 = Dense(dim, activation='sigmoid', name = 'feature')(h)
  
  model = Model(inputs = inputs, outputs = [output_1, output_2])
  
  model.compile(optimizer='rmsprop',
                loss={'mask': 'binary_crossentropy', 
                      'feature': 'mean_squared_error'},
                loss_weights={'mask':1, 'feature':alpha})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
            epochs = epochs, batch_size= batch_size)
      
  # Extract encoder part
  layer_name = model.layers[1].name
  layer_output = model.get_layer(layer_name).output
  encoder = models.Model(inputs=model.input, outputs=layer_output)
  
  return encoder
