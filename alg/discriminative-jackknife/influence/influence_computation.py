
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Code for influence functions computation in Pytorch
# ---------------------------------------------------------

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import warnings
warnings.simplefilter("ignore")

import torch

from influence.influence_utils import *

def influence_function(model, 
                       train_index,
                       W=None,
                       mode="stochastic",
                       batch_size=100, 
                       damp=1e-3, 
                       scale=1000, 
                       order=1,
                       recursion_depth=1000):

    """
    Computes the influence function defined as H^-1 dLoss/d theta. This is the impact that each
    training data point has on the learned model parameters. 
    """
    
    if mode=="stochastic":

        IF = influence_stochastic_estimation(model, train_index, batch_size, damp, scale, recursion_depth)

    if mode=="exact":    

        IF = exact_influence(model, train_index, damp, W, order)    
    
    return IF    


def influence_stochastic_estimation(model, 
                                    train_index,
                                    batch_size=100, 
                                    damp=1e-3, 
                                    scale=1000, 
                                    recursion_depth=1000):

    """
    This function applies the stochastic estimation approach to evaluating influence function based on the power-series
    approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:

    H^-1 = \sum^\infty_{i=0} (I - H) ^ j

    This series converges if all the eigen values of H are less than 1. 
    
    
    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        rnn: the model for which the Hessian of the loss is evaluated 
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """

    NUM_SAMPLES = model.X.shape[0]
    SUBSAMPLES  = batch_size
    
    loss        = [model.loss_fn(model.y[train_index[_]], model.predict(model.X[train_index[_], :], numpy_output=False)) for _ in range(len(train_index))]
    
    grads       = [stack_torch_tensors(torch.autograd.grad(loss[_], model.parameters(), create_graph=True)) for _ in range(len(train_index))]
     
    IHVP_       = [grads[_].clone().detach() for _ in range(len(train_index))]


    for j in range(recursion_depth):
        
        sampled_indx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES, replace=False)
      
        sampled_loss = model.loss_fn(model.y[sampled_indx], model.predict(model.X[sampled_indx, :], numpy_output=False))
        
        IHVP_prev    = [IHVP_[_].clone().detach() for _ in range(len(train_index))]
        
        hvps_        = [stack_torch_tensors(hessian_vector_product(sampled_loss, model, [IHVP_prev[_]])) for _ in range(len(train_index))]
     
        IHVP_        = [g_ + (1 - damp) * ihvp_ - hvp_/scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)] 
      
        
    return [-1 * IHVP_[_] / (scale * NUM_SAMPLES) for _ in range(len(train_index))] 
    
    


def exact_influence(model, train_index, damp=0, W=None, order=1):

    params_ = []

    for param in model.parameters():
    
        params_.append(param)
    
    num_par = stack_torch_tensors(params_).shape[0]   
    Hinv    = torch.inverse(exact_hessian(model) + damp * torch.eye(num_par))

    if order==2:

        H_ij   = [exact_hessian_ij(model, model.loss_fn(model.predict(model.X[_index], numpy_output=False), model.y[_index])) for _index in range(model.X.shape[0])]

    
    if W is None:

        y_preds = [model.predict(model.X[k, :], numpy_output=False) for k in train_index]

        if hasattr(model, 'masks'):

            losses   = [torch.sum(model.sequence_loss()[train_index[k], :]) for k in range(len(train_index))]
            n_factor = torch.sum(model.masks)

        else:

            losses   = [model.loss_fn(y_preds[k], model.y[train_index[k]]) for k in range(len(train_index))]
            n_factor = model.X.shape[0]
        
        grads   = [stack_torch_tensors(torch.autograd.grad(losses[k], model.parameters(), create_graph=True)) for k in range(len(losses))]
        
        if order==1:

            IFs_  = [-1 * torch.mm(Hinv, grads[k].reshape((grads[k].shape[0], 1))) / n_factor for k in range(len(grads))]  

        elif order==2:    

            IF_   = [-1 * torch.mm(Hinv, grads[k].reshape((grads[k].shape[0], 1))) / n_factor for k in range(len(grads))]
            IF2_  = [torch.mm(Hinv, torch.mm(H_ij[k], IF_[k])) * (2 / n_factor) for k in range(len(grads))]
            IFs_  = [IF_[k] + 0.5 * IF2_[k] for k in range(len(grads))]


    else:

        y_preds = model.predict(model.X, numpy_output=False)
        losses  = [model.loss_fn(y_preds[k], model.y[k]) * W[k] for k in range(len(y_preds))]
        grads   = stack_torch_tensors(torch.autograd.grad(torch.sum(stack_torch_tensors(losses)), model.parameters(), create_graph=True))
        IFs_    = [-1 * torch.mm(Hinv, grads.reshape((-1, 1))) / model.X.shape[0]] 
    
    return IFs_
