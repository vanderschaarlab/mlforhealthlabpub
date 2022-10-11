
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function

import collections
import hashlib
import numbers
import itertools
import functools
import sets
import inspect
import pickle
import json

from sklearn.model_selection import *
from sklearn.metrics import *

import tensorflow as tf
from tensorflow.contrib.rnn import PhasedLSTMCell, MultiRNNCell, BasicRNNCell
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op, dtypes, ops, tensor_shape, tensor_util   
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import * 
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import shutil

    
def data_loader(dir_alg):

    with open('{}/data/feature_names.json'.format(dir_alg)) as json_file:
    
        feature_names = json.load(json_file)
    
    data = pickle.load(open("{}/data/CF_data.p".format(dir_alg), "rb"))

    X_obs  = []
    
    # Drop patient IDs

    for k in range(len(data)):

        X_obs.append(data[k][:, 1:])
        
    return X_obs, feature_names   


