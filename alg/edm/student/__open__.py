import os
import tensorflow as tf; tf.enable_v2_behavior()
import sys; sys.path.append(os.path.join(os.getcwd(), 'contrib'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ArgParse(dict):
    def __init__(self, *args, **kwargs):
        super(ArgParse, self).__init__(*args, **kwargs)
        self.__dict__ = self

import numpy as np

def one_hot(action, action_dim):
    return np.squeeze(np.eye(action_dim)[action.reshape(-1)])
