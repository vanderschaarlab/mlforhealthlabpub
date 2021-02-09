
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch import nn
from torch.autograd import Variable
import scipy.stats as st

from utils.data_padding import *

torch.manual_seed(1) 

from influence.influence_computation import *
from influence.influence_utils import * 


class RNN_uncertainty_wrapper():
    
    def __init__(self, model, mode="exact", damp=1e-4):
        
        self.model            = model
        self.IF               = influence_function(model, train_index=list(range(model.X.shape[0])), 
                                                   mode=mode, damp=damp)

        X_                    = [model.X[k][:int(torch.sum(model.masks[k, :]).detach().numpy())] for k in range(model.X.shape[0])]
        self.LOBO_residuals   = [] 

        for k in range(len(self.IF)):
    
            perturbed_models  = perturb_model_(self.model, self.IF[k])
            self.LOBO_residuals.append(np.abs(np.array(self.model.y[k]).reshape(-1, 1) - np.array(perturbed_models.predict(X_[k], padd=False)).T))
    
            del perturbed_models
        
        self.LOBO_residuals   = np.squeeze(np.array(self.LOBO_residuals))    

        #self.perturbed_models = [perturb_model_(model, self.IF[k]) for k in range(len(self.IF))]
        #self.LOBO_residuals   = np.abs(np.array(self.model.y) - np.array(self.model.predict(X_, padd=True)))

    def predict(self, X_test, coverage=0.95):

        variable_preds        = []
        num_sequences         = np.array(X_test).shape[0]
        
        for k in range(len(self.IF)):
    
            perturbed_models  = perturb_model_(self.model, self.IF[k])
            variable_preds.append(perturbed_models.predict(X_test))
    
            del perturbed_models

        variable_preds        = np.array(variable_preds)     
        #variable_preds        = np.array([self.perturbed_models[k].predict(X_test).reshape(-1,) for k in range(len(self.perturbed_models))])

        y_u_approx            = np.quantile(variable_preds + np.repeat(np.expand_dims(self.LOBO_residuals, axis=1), num_sequences, axis=1), 1 - (1-coverage)/2, axis=0, keepdims=False)
        y_l_approx            = np.quantile(variable_preds - np.repeat(np.expand_dims(self.LOBO_residuals, axis=1), num_sequences, axis=1), (1-coverage)/2, axis=0, keepdims=False)
        y_pred                = self.model.predict(X_test)
        
        return y_pred, y_l_approx, y_u_approx 


def model_loss_single(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.mean(torch.sum(single_loss, axis=0) / torch.sum(masks, axis=0)) 
    
    return loss


def single_losses(model):

    return model.masks * (model(model.X).view(-1, model.MAX_STEPS) - model.y)**2


def model_loss(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.sum(torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1)))
    
    return loss


def quantile_loss(output, target, masks, q):
    
    single_loss  = masks * ((output - target) * (output >= target)  * q + (target - output) * (output < target)  * (1-q))
    loss         = torch.mean(torch.sum(single_loss, axis=1) / torch.sum(masks, axis=1)) 
    
    return loss    


class RNN(nn.Module):
    
    def __init__(self, 
                 mode="RNN",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,  
                 INPUT_SIZE=30,     
                 LR=0.01,   
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50):
        

        super(RNN, self).__init__()

        self.EPOCH          = EPOCH      
        self.BATCH_SIZE     = BATCH_SIZE
        self.MAX_STEPS      = MAX_STEPS  
        self.INPUT_SIZE     = INPUT_SIZE     
        self.LR             = LR   
        self.OUTPUT_SIZE    = OUTPUT_SIZE
        self.HIDDEN_UNITS   = HIDDEN_UNITS
        self.NUM_LAYERS     = NUM_LAYERS 
        self.N_STEPS        = N_STEPS

        rnn_dict = {"RNN" : nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "LSTM": nn.LSTM(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "GRU" : nn.GRU(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,)
                    }

        self.mode = mode            
        self.rnn  = rnn_dict[self.mode]

        self.out  = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        else:

            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, :, :]) 
        
        return out
    
    def fit(self, X, Y):
        
        X_padded, _          = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[1], axis=2)
        
        X                    = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y                    = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks           = Variable(torch.tensor(loss_masks), volatile=True).type(torch.FloatTensor)
        
        self.X               = X
        self.y               = Y
        self.masks           = loss_masks
        
        optimizer            = torch.optim.Adam(self.parameters(), lr=self.LR)   # optimize all rnn parameters
        self.loss_fn         = model_loss #nn.MSELoss() 
        
        # training and testing
        for epoch in range(self.EPOCH):

            for step in range(self.N_STEPS):
                
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)
                
                x   = torch.tensor(X[batch_indexes, :, :])
                y   = torch.tensor(Y[batch_indexes])
                msk = torch.tensor(loss_masks[batch_indexes])
                
                b_x = Variable(x.view(-1, self.MAX_STEPS, self.INPUT_SIZE))   # reshape x to (batch, time_step, input_size)
                b_y = Variable(y)                                             # batch y
                b_m = Variable(msk)
                
                output    = self(b_x).view(-1, self.MAX_STEPS)     # rnn output
                
                self.loss = self.loss_fn(output, b_y, b_m)       # MSE loss  
                
                optimizer.zero_grad()                              # clear gradients for this training step
                self.loss.backward(retain_graph=True)              # backpropagation, compute gradients
                optimizer.step()                                   # apply gradients

                if step % 50 == 0:

                    print('Epoch: ', epoch, '| train loss: %.4f' % self.loss.data)
    
    def predict(self, X, padd=False, numpy_output=False):
        
        if type(X) is list:
            
            X_, masks  = padd_arrays(X, max_length=self.MAX_STEPS)
        
        else:
            
            X_, masks  = padd_arrays([X], max_length=self.MAX_STEPS)
        
        X_test     = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)
        predicts_  = self(X_test).view(-1, self.MAX_STEPS) 

        if padd:

            prediction = unpadd_arrays(predicts_.detach().numpy(), masks)

        else:
        
            prediction = predicts_.detach().numpy()
        
        return prediction
    
    def sequence_loss(self):

        return single_losses(self)


class DPRNN(nn.Module):
    
    def __init__(self, 
                 mode="RNN",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,  
                 INPUT_SIZE=30,     
                 LR=0.01,   
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50,
                 dropout_prob=0.5):
        
        super(DPRNN, self).__init__()
        
        self.EPOCH          = EPOCH      
        self.BATCH_SIZE     = BATCH_SIZE
        self.MAX_STEPS      = MAX_STEPS  
        self.INPUT_SIZE     = INPUT_SIZE     
        self.LR             = LR   
        self.OUTPUT_SIZE    = OUTPUT_SIZE
        self.HIDDEN_UNITS   = HIDDEN_UNITS
        self.NUM_LAYERS     = NUM_LAYERS 
        self.N_STEPS        = N_STEPS

        self.mode           = mode
        self.dropout_prob   = dropout_prob 
        self.dropout        = nn.Dropout(p=dropout_prob)

        rnn_dict = {"RNN" : nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True, dropout = self.dropout_prob,),
                    "LSTM": nn.LSTM(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True, dropout = self.dropout_prob,),
                    "GRU" : nn.GRU(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True, dropout = self.dropout_prob,)
                    }
        
        self.rnn            = rnn_dict[self.mode ] 
        self.out            = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        else:

            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(self.dropout(r_out[:, :, :])) 
        
        return out

    def fit(self, X, Y):
        
        X_padded, _          = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[1], axis=2)
        
        X                    = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y                    = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks           = Variable(torch.tensor(loss_masks), volatile=True).type(torch.FloatTensor)
        
        self.X         = X
        self.Y         = Y
        self.masks     = loss_masks
        
        optimizer      = torch.optim.Adam(self.parameters(), lr=self.LR)   # optimize all rnn parameters
        self.loss_func = model_loss #nn.MSELoss() 
        
        # training and testing
        for epoch in range(self.EPOCH):

            for step in range(self.N_STEPS):
                
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)
                
                x   = torch.tensor(X[batch_indexes, :, :])
                y   = torch.tensor(Y[batch_indexes])
                msk = torch.tensor(loss_masks[batch_indexes])
                
                b_x = Variable(x.view(-1, self.MAX_STEPS, self.INPUT_SIZE))   # reshape x to (batch, time_step, input_size)
                b_y = Variable(y)                                            # batch y
                b_m = Variable(msk)
                
                output = self(b_x).view(-1, self.MAX_STEPS)     # rnn output
                
                loss   = self.loss_func(output, b_y, b_m)       # MSE loss
                
                optimizer.zero_grad()                           # clear gradients for this training step
                loss.backward()                                 # backpropagation, compute gradients
                optimizer.step()                                # apply gradients

                if step % 50 == 0:

                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
    
    def predict(self, X, num_samples=100, alpha=0.05): 
        
        z_critical     = st.norm.ppf((1 - alpha) + (alpha)/2)
        
        if type(X) is list:
                
            X_, masks  = padd_arrays(X, max_length=self.MAX_STEPS)
        
        else:
            
            X_, masks  = padd_arrays([X], max_length=self.MAX_STEPS)
            
        
        predictions  = []
        X_test       = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)     
        
        for idx in range(num_samples):
        
            predicts_           = self(X_test).view(-1, self.MAX_STEPS) 
        
            predictions.append(predicts_.detach().numpy().reshape((-1, 1, self.MAX_STEPS)))

            
        pred_mean   = unpadd_arrays(np.mean(np.concatenate(predictions, axis=1), axis=1), masks)
        pred_std    = unpadd_arrays(z_critical * np.std(np.concatenate(predictions, axis=1), axis=1), masks)

        return pred_mean, pred_std


class QRNN(nn.Module):
    
    def __init__(self, 
                 mode="RNN",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,  
                 INPUT_SIZE=30,     
                 LR=0.01,   
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50,
                 alpha=0.05):
        
        super(QRNN, self).__init__()
        
        self.EPOCH          = EPOCH      
        self.BATCH_SIZE     = BATCH_SIZE
        self.MAX_STEPS      = MAX_STEPS  
        self.INPUT_SIZE     = INPUT_SIZE     
        self.LR             = LR   
        self.OUTPUT_SIZE    = OUTPUT_SIZE
        self.HIDDEN_UNITS   = HIDDEN_UNITS
        self.NUM_LAYERS     = NUM_LAYERS 
        self.N_STEPS        = N_STEPS
        self.q              = alpha
        self.mode           = mode

        rnn_dict = {"RNN" : nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "LSTM": nn.LSTM(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "GRU" : nn.GRU(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,)
                    }
        

        self.rnn = rnn_dict[self.mode]

        self.out = nn.Linear(self.HIDDEN_UNITS, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        else:

            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, :, :]) 
        
        return out
    
    def fit(self, X, Y):
        
        X_padded, _          = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[1], axis=2)
        
        X                    = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y                    = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks           = Variable(torch.tensor(loss_masks), volatile=True).type(torch.FloatTensor)
        
        self.X               = X
        self.Y               = Y
        self.masks           = loss_masks
        
        optimizer            = torch.optim.Adam(self.parameters(), lr=self.LR)   # optimize all rnn parameters
        self.loss_func       = quantile_loss 
        
        # training and testing
        for epoch in range(self.EPOCH):

            for step in range(self.N_STEPS):
                
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)
                
                x      = torch.tensor(X[batch_indexes, :, :])
                y      = torch.tensor(Y[batch_indexes])
                msk    = torch.tensor(loss_masks[batch_indexes])
                
                b_x    = Variable(x.view(-1, self.MAX_STEPS, self.INPUT_SIZE))   # reshape x to (batch, time_step, input_size)
                b_y    = Variable(y)                                             # batch y
                b_m    = Variable(msk)
                
                output = self(b_x).view(-1, self.MAX_STEPS, 2)                # rnn output
                
                loss   = self.loss_func(output[:, :, 0], b_y, b_m, self.q) + self.loss_func(output[:, :, 1], b_y, b_m, 1 - self.q)       # MSE loss
                
                optimizer.zero_grad()                           # clear gradients for this training step
                loss.backward()                                 # backpropagation, compute gradients
                optimizer.step()                                # apply gradients

                if step % 50 == 0:

                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
    
    def predict(self, X):
        
        if type(X) is list:
            
            X_, masks  = padd_arrays(X, max_length=self.MAX_STEPS)
        
        else:
            
            X_, masks  = padd_arrays([X], max_length=self.MAX_STEPS)
        
        X_test       = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)
        predicts_    = self(X_test).view(-1, self.MAX_STEPS, 2) 
        prediction_0 = unpadd_arrays(predicts_[:, :, 0].detach().numpy(), masks)
        prediction_1 = unpadd_arrays(predicts_[:, :, 1].detach().numpy(), masks)
        
        return prediction_0, prediction_1
