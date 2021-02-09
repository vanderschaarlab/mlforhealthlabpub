
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Helper functions for influence functions' 
# computation in Pytorch
# ---------------------------------------------------------

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import collections

import numpy as np

import warnings
warnings.simplefilter("ignore")

import torch

def stack_torch_tensors(input_tensors):
    
    '''
    Takes a list of tensors and stacks them into one tensor
    '''

    unrolled = [input_tensors[k].view(-1,1) for k in range(len(input_tensors))]

    return torch.cat(unrolled)


def get_numpy_parameters(model):

    '''
    Recovers the parameters of a pytorch model in numpy format
    '''	

    params = []

    for param in model.parameters():
    
        params.append(param)

    return stack_torch_tensors(params).detach().numpy() 


def exact_hessian(model):
    
    grad_params = torch.autograd.grad(model.loss, model.parameters(), retain_graph=True, create_graph=True)  
    grad_params = stack_torch_tensors(grad_params)
    hess_params = torch.zeros((len(grad_params), len(grad_params)))
    temp        = []

    for u in range(len(grad_params)):
    
        second_grad = torch.autograd.grad(grad_params[u], model.parameters(), retain_graph=True)
    
        temp.append(stack_torch_tensors(second_grad))

    Hessian     = torch.cat(temp, axis=1)   
        
    return Hessian


def exact_hessian_ij(model, loss_ij):
    
    grad_params = torch.autograd.grad(loss_ij, model.parameters(), retain_graph=True, create_graph=True)  
    grad_params = stack_torch_tensors(grad_params)
    hess_params = torch.zeros((len(grad_params), len(grad_params)))
    temp        = []

    for u in range(len(grad_params)):
    
        second_grad = torch.autograd.grad(grad_params[u], model.parameters(), retain_graph=True)
    
        temp.append(stack_torch_tensors(second_grad))

    Hessian     = torch.cat(temp, axis=1)   
        
    return Hessian


def hessian_vector_product(loss, model, v):
    
    """
    Multiplies the Hessians of the loss of a model with respect to its parameters by a vector v.
    Adapted from: https://github.com/kohpangwei/influence-release
    
    This function uses a backproplike approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians with O(p) complexity for p parameters.
    
    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        rnn: the model for which the Hessian of the loss is evaluated 
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """
    
    # First backprop
    first_grads       = stack_torch_tensors(torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True))

    """
    # Elementwise products
    elemwise_products = 0
    
    for grad_elem, v_elem in zip(first_grads, v):
        
        elemwise_products += torch.sum(grad_elem * v_elem)
    """

    elemwise_products = torch.mm(first_grads.view(-1, first_grads.shape[0]).float(), 
                                 v[0].view(first_grads.shape[0], -1).float())  
    
    # Second backprop
    HVP_              = torch.autograd.grad(elemwise_products, model.parameters(), create_graph=True) 
                         
    return HVP_



def perturb_model_(model, perturb):

    """
    Perturbs the parameters of a model by a given vector of influences 
    
    Arguments:
        model: a pytorch model with p parameters
        perturb: a tensors with size p designating the desired parameter-wise perturbation

    Returns:
        perturbed_model : a copy of the original model with perturbed parameters 
    """

    params            = []
    NUM_SAMPLES       = model.X.shape[0]

    for param in model.parameters():
    
        params.append(param.clone())


    param_            = stack_torch_tensors(params) 
    new_param_        = param_ - perturb

    # copy all model attributes

    perturbed_model   = type(model)()
    
    new_model_dict    = dict.fromkeys(model.__dict__.keys())
    new_model_state   = collections.OrderedDict.fromkeys(model.state_dict().keys())
    
    for key in new_model_dict.keys():
        
        if type(model.__dict__[key])==torch.Tensor:
            
            new_model_dict[key] = model.__dict__[key].clone()
        
        else:
            
            new_model_dict[key] = copy.deepcopy(model.__dict__[key])
            
     
    for key in new_model_state.keys(): 

        if type(model.state_dict()[key])==torch.Tensor:
            
            new_model_state[key] = model.state_dict()[key].clone()
        
        else:
            
            new_model_state[key] = copy.deepcopy(model.state_dict()[key])
        
    
    perturbed_model.__dict__.update(new_model_dict) 
    perturbed_model.load_state_dict(new_model_state) 
    
    #perturbed_model.__dict__.update(model.__dict__) 
    #perturbed_model.load_state_dict(model.state_dict())  # copy weights and stuff

    index             = 0

    for param in perturbed_model.parameters():
        
        if len(param.data.shape) > 1:
        
            new_size   = np.max((1, param.data.shape[0])) * np.max((1, param.data.shape[1]))
            param.data = new_param_[index: index + new_size].view(param.data.shape[0], param.data.shape[1])
        
        else:
            
            new_size   = param.data.shape[0]
            param.data = np.squeeze(new_param_[index: index + new_size])
        

        index += new_size

    return perturbed_model 

