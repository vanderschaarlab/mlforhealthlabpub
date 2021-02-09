
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pandas as pd
import numpy as np
import scipy

def compute_PEHE(T_true, T_est):
    
    return np.sqrt(np.mean((T_true.reshape((-1,1)) - T_est.reshape((-1,1)))**2))


def mean_confidence_interval(data, confidence=0.95):
 
    a     = 1.0 * np.array(data)
    n     = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h     = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return m, h
    

