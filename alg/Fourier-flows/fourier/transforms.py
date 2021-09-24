# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains functions for applying fourier transform and its inverse in a torch environment.
Used as the "Fourier Transform" step in a Fourier Flow, can be preceded by other torch modules.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal

import sys

import warnings

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Helper functions


def flip(x, dim):

    """flipping helper
    
       Takes a vector as an input, then flips its elements from left to right
       
       :x: input vector of size N x 1
       :dim: splitting dimension
        
    """

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


def reconstruct_DFT(x, component="real"):

    """prepares input to the DFT inverse
    
       Takes a cropped frequency and creates a symmetric or anti-symmetric mirror of it before applying inverse DFT
        
    """

    if component == "real":
        x_rec = torch.cat([x[0, :], flip(x[0, :], dim=0)[1:]], dim=0)

    elif component == "imag":
        x_rec = torch.cat([x[1, :], -1 * flip(x[1, :], dim=0)[1:]], dim=0)

    return x_rec


# Main DFT module


class DFT(nn.Module):

    """
    Discrete Fourier Transform (DFT) torch module
    
    >> attributes <<
    
    :N_fft: size of the DFT transform, conventionally set to length of the input time-series or a fixed 
            number of desired spectral components
            
    :crop_size: always equals to the size of non-redundant frequency components, i.e. N_fft / 2 since we deal with
                real-valued inputs, then the DFT is symmetric around 0 and half of the spectral components are redundant
    
    :base_dist: base distribution of the flow, always defined as a multi-variate normal distribution
    
    """

    def __init__(self, N_fft=100):

        super(DFT, self).__init__()

        self.N_fft = N_fft
        self.crop_size = int(self.N_fft / 2) + 1
        base_mu, base_cov = torch.zeros(self.crop_size * 2), torch.eye(self.crop_size * 2)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):

        """forward steps
        
        Step 1: Convert the input vector to numpy format
        
        Step 2: Apply FFT in numpy with FFTshift to center the spectrum around 0
        
        Step 3: Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size 
                is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size 
                remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft
        
        Step 4: Convert spectrum back to torch tensor format
        
        Step 5: Compute the flow likelihood and Jacobean. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0
        
        """

        if len(x.shape) == 1:

            x = x.reshape((1, -1))

        x_numpy = x.detach().float()
        X_fft = [np.fft.fftshift(np.fft.fft(x_numpy[k, :])) for k in range(x.shape[0])]
        X_fft_train = np.array(
            [
                np.array(
                    [np.real(X_fft[k])[: self.crop_size] / self.N_fft, np.imag(X_fft[k])[: self.crop_size] / self.N_fft]
                )
                for k in range(len(X_fft))
            ]
        )
        x_fft = torch.from_numpy(X_fft_train).float()

        log_pz = self.base_dist.log_prob(x_fft.view(-1, x_fft.shape[1] * x_fft.shape[2]))
        log_jacob = 0

        return x_fft, log_pz, log_jacob

    def inverse(self, x):

        """Inverse steps
        
        Step 1: Convert the input vector to numpy format with size NUM_SAMPLES x 2 x N_fft
                Second dimension indexes the real and imaginary components.
                
        Step 2: Apply FFT in numpy with FFTshift to center the spectrum around 0
        
        Step 3: Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size 
                is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size 
                remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft
        
        Step 4: Convert spectrum back to torch tensor format
        
        Step 5: Compute the flow likelihood and Jacobean. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0
        
        """

        x_numpy = x.view((-1, 2, self.crop_size))

        x_numpy_r = [
            reconstruct_DFT(x_numpy[u, :, :], component="real").detach().numpy() for u in range(x_numpy.shape[0])
        ]
        x_numpy_i = [
            reconstruct_DFT(x_numpy[u, :, :], component="imag").detach().numpy() for u in range(x_numpy.shape[0])
        ]

        x_ifft = [
            self.N_fft * np.real(np.fft.ifft(np.fft.ifftshift(x_numpy_r[u] + 1j * x_numpy_i[u])))
            for u in range(x_numpy.shape[0])
        ]
        x_ifft_out = torch.from_numpy(np.array(x_ifft)).float()

        return x_ifft_out
