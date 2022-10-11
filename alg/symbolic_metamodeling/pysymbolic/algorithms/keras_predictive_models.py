from __future__ import absolute_import, division, print_function

import keras
import numpy as np
from sklearn.neural_network import MLPClassifier

class pred_model(keras.models.Sequential):

    def predict_proba(self, x_):
        
        y_ = self.predict(x_)
        
        return np.float64(np.hstack((y_, 1-y_))) 

    def predict_(self, x_):
        
        y_ = self.predict(x_).reshape((-1,))
        
        return y_

    
def get_predictive_model(x_train, y_train, num_layers=2, num_hidden=200, 
                         activation='relu', out_activation='sigmoid', 
                         optimizer="adam", loss="binary_crossentropy", 
                         dropout_rate=0.5, model_type="keras"):
    
    # create a keras model
    
    if model_type=="keras":

    	model = keras.models.Sequential()
    
    elif model_type=="modified_keras":

    	model = pred_model()

    elif model_type=="sklearn":
        model = MLPClassifier(hidden_layer_sizes=(200,200, ))	


    if (model_type=="keras") or (model_type=="modified_keras"):

    	# Initiate the input layer
    
    	model.add(keras.layers.normalization.BatchNormalization(input_shape=tuple([x_train.shape[1]])))
    	model.add(keras.layers.core.Dense(num_hidden, activation=activation))
    	model.add(keras.layers.core.Dropout(rate=dropout_rate))
    
    	# Add all intermediate layers
    
    	for _ in range(num_layers-1):
    
        	model.add(keras.layers.normalization.BatchNormalization())
        	model.add(keras.layers.core.Dense(num_hidden, activation=activation))
        	model.add(keras.layers.core.Dropout(rate=dropout_rate))

    		# Add output layer    
    
    	model.add(keras.layers.core.Dense(1, activation=out_activation))
    
    	model.compile(loss=loss, optimizer=optimizer,metrics=["accuracy"])
    	#print(model.summary())

    	# Use Early-Stopping
    	callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    	# Train model
    	model.fit(x_train, y_train, batch_size=1024, epochs=200, verbose=0, callbacks=[callback_early_stopping])

    else:

    	model.fit(x_train, y_train)

    return model
    










