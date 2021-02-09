
"""
This script contains functions for generating synthetic data. 
The code is based on https://github.com/Jianbo-Lab/L2X

""" 

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
import itertools

from mpmath import *
from sympy import *

from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

    
def create_rank(scores, k): 
    """
    Compute rank of each feature based on weight.
    
    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d) 
        permutated_weights = score[idx]  
        permutated_rank=(-permutated_weights).argsort().argsort()+1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)

    return np.array(ranks)

def compute_median_rank(scores, k, datatype_val = None):
    ranks = create_rank(scores, k)
    if datatype_val is None: 
        median_ranks = np.median(ranks[:,:k], axis = 1)
    else:
        datatype_val = datatype_val[:len(scores)]
        median_ranks1 = np.median(ranks[datatype_val == 'orange_skin',:][:,np.array([0,1,2,3,9])], 
            axis = 1)
        median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive',:][:,np.array([4,5,6,7,9])], 
            axis = 1)
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    return median_ranks 





