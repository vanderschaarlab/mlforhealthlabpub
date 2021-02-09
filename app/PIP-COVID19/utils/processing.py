
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
from datetime import datetime

from scipy.integrate import odeint

import warnings
warnings.filterwarnings('ignore')


def compute_loss(y_true, y_pred):
    
    return np.mean(np.abs(np.cumsum(y_true) - np.cumsum(y_pred)))


def smooth_curve_1d(x, d=7):
    
    y = []
    
    for u in range(len(x)):
        
        if u >= d:
            
            y.append(np.mean(x[u-d:u]))
            
        elif u==0:
            
            y.append(np.mean(x[0]))
            
        else:
            
            y.append(np.mean(x[:u]))
    
    return np.array(y)


def smoothen_mobility_scores(mobility_scores):
    
    sm_mob = []

    for i in range(mobility_scores.shape[1]):
        
        s  = mobility_scores[:, i]
        sl = smooth_curve_1d(s)
        #sl = np.concatenate([sl, np.ones(3) * sl[-1]])
        
        sm_mob.append(sl)
    
    mobility_scores = np.stack(sm_mob, axis=-1)
    
    return mobility_scores


def moving_average(a, n=3):
    
    ret     = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n

