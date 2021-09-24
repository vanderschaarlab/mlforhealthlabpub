# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains the implementation for the spectral filter module of the Fourier flow

"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal

import sys

import warnings

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from models.sequential import RNN


class SpectralFilter(nn.Module):

    """
    Spectral Filter torch module
    
    >> attributes <<
    
    :d: number of input dimensions
            
    :k: dimension of split in the input space
    
    :FFT: number of FFT components
    
    :hidden: number of hidden units in the spectral filter layer
    
    :flip: boolean indicator on whether to flip the split dimensions
    
    :RNN: boolean indicator on whether to use an RNN in spectral filtering
    
    """

    def __init__(self, d, k, FFT, hidden, flip=False, RNN=False):

        super().__init__()

        self.d, self.k = d, k

        if FFT:

            self.out_size = self.d - self.k + 1
            self.pz_size = self.d + 1
            self.in_size = self.k

        else:

            self.out_size = self.d - self.k
            self.pz_size = self.d
            self.in_size = self.k

        if flip:

            self.in_size, self.out_size = self.out_size, self.in_size

        self.sig_net = nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=False):

        """forward steps
        
           Similar to RealNVP, see:
           Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. 
           "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).
        
        """

        x1, x2 = x[:, : self.k], x[:, self.k :]

        if flip:

            x2, x1 = x1, x2

        # forward

        sig = self.sig_net(x1).view(-1, self.out_size)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1).view(-1, self.out_size)

        if flip:

            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z, flip=False):

        z1, z2 = Z[:, : self.k], Z[:, self.k :]

        if flip:
            z2, z1 = z1, z2

        x1 = z1

        sig_in = self.sig_net(z1).view(-1, self.out_size)
        x2 = (z2 - self.mu_net(z1).view(-1, self.out_size)) * torch.exp(-sig_in)

        if flip:

            x2, x1 = x1, x2

        return torch.cat([x1, x2], -1)


class AttentionFilter(nn.Module):

    """
    Attention Filter torch module
    
    >> attributes <<
    
    :d: number of input dimensions
            
    :k: dimension of split in the input space
    
    :FFT: number of FFT components
    
    :hidden: number of hidden units in the spectral filter layer
    
    :flip: boolean indicator on whether to flip the split dimensions
    
    :RNN: boolean indicator on whether to use an RNN in spectral filtering
    
    """

    def __init__(self, d, k, FFT, hidden, flip=False):

        super().__init__()

        self.d, self.k = d, k

        if FFT:

            self.out_size = self.d - self.k + 1
            self.pz_size = self.d + 1
            self.in_size = self.k

        else:

            self.out_size = self.d - self.k
            self.pz_size = self.d
            self.in_size = self.k

        if flip:

            self.in_size, self.out_size = self.out_size, self.in_size

        self.sig_net = nn.Sequential(
            RNN(mode="LSTM", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(
            RNN(mode="LSTM", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=False):

        """forward steps
        
           Similar to RealNVP, see:
           Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. 
           "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).
        
        """

        x1, x2 = x[:, : self.k], x[:, self.k :]

        if flip:

            x2, x1 = x1, x2

        # forward

        sig = self.sig_net(x1).view(-1, self.out_size)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1).view(-1, self.out_size)

        if flip:

            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z, flip=False):

        z1, z2 = Z[:, : self.k], Z[:, self.k :]

        if flip:
            z2, z1 = z1, z2

        x1 = z1

        sig_in = self.sig_net(z1).view(-1, self.out_size)
        x2 = (z2 - self.mu_net(z1).view(-1, self.out_size)) * torch.exp(-sig_in)

        if flip:

            x2, x1 = x1, x2

        return torch.cat([x1, x2], -1)
