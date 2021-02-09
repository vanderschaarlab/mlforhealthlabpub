import numpy as np

from models.BNNLayer import BNNLayer
from models.BNN import BNN

import torch
from torch.autograd import Variable 

from scipy.stats import norm

import warnings
warnings.simplefilter("ignore")


class Bayes_backprop:
    
    def __init__(self, num_layers=2, input_dim=1, num_hidden=100):
        
        self.hidden_layers = [num_hidden for k in range(num_layers)]
        self.num_hidden    = num_hidden
        self.num_layers    = num_layers
        self.dim           = input_dim
        
        self.bnn           = BNN(BNNLayer(input_dim, self.num_hidden, activation='relu', prior_mean=0, prior_rho=0),
                                 BNNLayer(self.num_hidden, 1, activation='none', prior_mean=0, prior_rho=0))
        
    
    def fit(self, X, y, num_epochs=400, learning_rate=1e-1):
        
        Var     = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
        self.X  = Var(X)
        self.Y  = Var(y)
        
        self.optim = torch.optim.Adam(self.bnn.parameters(), lr=learning_rate)
        
        for i_ep in range(num_epochs):
        
            kl, lg_lklh  = self.bnn.Forward(self.X, self.Y, 1, 'Gaussian')
            self.loss    = BNN.loss_fn(kl, lg_lklh, 1)
        
            self.optim.zero_grad()
            self.loss.backward(retain_graph=True)
            self.optim.step()
        
    def predict(self, X, alpha=0.1, num_MC=100):
        
        Var       = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
        
        if self.dim == 1:
        
            X_new     = Var(X).unsqueeze(1)
        
        else:
            
            X_new     = Var(X.reshape((-1, self.dim))) 
    
        pred_lst  = [self.bnn.forward(X_new, mode='MC').data.numpy().squeeze(1) for _ in range(num_MC)]

        pred      = np.array(pred_lst).T
        pred_mean = pred.mean(axis=1)
        pred_std  = pred.std(axis=1)
        pred_up   = pred_mean + norm.ppf(1 - alpha/2) * pred_std 
        pred_lo   = pred_mean - norm.ppf(1 - alpha/2) * pred_std
        
        return pred_mean, pred_up, pred_lo
