
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

from sklearn.model_selection import *
from sklearn.metrics import *

import os
from pathlib import Path
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import multivariate_normal
from hmmlearn import hmm


class HMM:
    
    def __init__(self, num_states):
        
        self.num_states = num_states
        self.hmm_model  = hmm.GaussianHMM(n_components=num_states)
        
        
    def fit(self, X_obs):
        
        X_obser = [[list(X_obs[u][k,:]) for k in range(len(X_obs[u]))] for u in range(len(X_obs))]
        X_obs_  = np.concatenate(X_obser)

        lengths = [len(X_obs[k]) for k in range(len(X_obs))]
        
        self.hmm_model.fit(X_obs_, lengths)
        
        
    def predict(self, X_new):
        
        preds       = np.concatenate([self.hmm_model.predict(X_new[k]) for k in range(len(X_new))])
        predictions = [np.sum(np.concatenate([(self.hmm_model.transmat_[preds[k],u] * self.hmm_model.means_[u]).reshape((1,-1)) for u in range(self.num_states)]), axis=0).reshape((1,-1)) for k in range(len(preds))] 
        
        return np.concatenate(predictions)
        


@tf_export("nn.rnn_cell.MultiRNNCell")
class MultiPhasedLSTMCell(MultiRNNCell):
    
    """RNN cell composed sequentially of multiple simple cells.

    Example:

    ```python
    num_units = [128, 64]
    cells = [BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = MultiRNNCell(cells)
    ```
    """

    def __init__(self, cells, state_is_tuple=True):
        
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(MultiRNNCell, self).__init__()
        
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError("cells must be a list or tuple, but saw: %s." % cells)

        self._cells = cells
        for cell_number, cell in enumerate(self._cells):
        
            # Add Checkpointable dependencies on these cells so their variables get
            # saved with this object when using object-based saving.
            if isinstance(cell, checkpointable.CheckpointableBase):
                # TODO(allenl): Track down non-Checkpointable callers.
                self._track_checkpointable(cell, name="cell-%d" % (cell_number,))
                self._state_is_tuple = state_is_tuple
            
            if not state_is_tuple:
                if any(nest.is_sequence(c.state_size) for c in self._cells):
                    raise ValueError("Some cells return tuples of states, but the flag "
                                     "state_is_tuple is not set.  State sizes are: %s"
                                     % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0

        times         = inputs[0]
        cur_inp       = inputs[1]
        
        new_states    = []
    
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError("Expected state to be a tuple of length %d, but received: %s" %(len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                
                cur_inp, new_state = cell((times, cur_inp), cur_state)
                new_states.append(new_state)

        new_states = (tuple(new_states) if self._state_is_tuple else array_ops.concat(new_states, 1))

        return cur_inp, new_states




def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def padd_data(X, padd_length):
    
    X_padded      = []
    
    for k in range(len(X)):
        
        if X[k].shape[0] < padd_length:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(np.vstack((np.array(X[k]), 
                                                    np.zeros((padd_length-X[k].shape[0],X[k].shape[1]))))))
            else:
                X_padded.append(np.array(np.vstack((np.array(X[k]).reshape((-1,1)),
                                                    np.zeros((padd_length-X[k].shape[0],1))))))
                
        else:
            
            if len(X[k].shape) > 1:
                X_padded.append(np.array(X[k]))
            else:
                X_padded.append(np.array(X[k]).reshape((-1,1)))
  

    X_padded      = np.array(X_padded)

    return X_padded


def flatten_sequences_to_numpy(sequence_list):
    
    seqLists   = [list(itertools.chain.from_iterable(sequence_list[k].tolist())) for k in range(len(sequence_list))]
    flat_seqs  = np.array(list(itertools.chain.from_iterable(seqLists)))
    
    return flat_seqs


def state_to_array(state_index, number_of_states):
    
    state_array = np.zeros(number_of_states)
    state_array[state_index] = 1
    
    return state_array
    
    

def get_transitions(state_array, num_states):
    
    trans_matrix = np.zeros((num_states, num_states))
    each_state   = [np.sum((state_array==k)*1) for k in range(num_states)]
    
    for k in range(num_states):
        
        where_states       = np.where(state_array==k)[0]
        where_states_      = where_states[where_states < len(state_array) - 1]
        
        after_states       = [state_array[where_states_[k] + 1] for k in range(len(where_states_))]
        trans_matrix[k, :] = np.array([(np.where(np.array(after_states)==k))[0].shape[0] for k in range(num_states)])
        
    return trans_matrix, each_state


class attentive_state_space_model:
    
    '''
    Class for the "Attentive state space model" implementation. Based on the paper: 
    "Attentive state space model for disease progression" by Ahmed M. Alaa and Mihaela van der Schaar.
    
    ** Key arguments **
    
    :param maximum_seq_length: Maximum allowable length for any trajectory in the training data. 
    :param input_dim: Dimensionality of the observations (emissions).
    :param num_states: Cardinality of the state space.
    :param inference_network: Configuration of the inference network. Default is: 'Seq2SeqAttention'.
    :param rnn_type: Type of RNN cells to use in the inference network. Default is 'LSTM'.
    :param unsupervised: Boolean for whether the model is supervised or unsupervised. Default is True. 
                         Supervised is NOT IMPLEMENTED.
    :param generative: Boolean for whether to enable sampling from the model. 
    :param irregular: Whether the trajectories are in continuous time. NOT IMPLEMENTED.
    :param multitask: Boolean for whether multi-task output layers are used in inference network. NOT IMPLEMENTED
    :param num_iterations: Number of iterations for the stochastic variational inference algorithm.
    :param num_epochs: Number of epochs for the stochastic variational inference algorithm.
    :param batch_size: Size of the batch subsampled from the training data.
    :param learning_rate: Learning rate for the ADAM optimizer. (TO DO: enable selection of the optimizer)
    :param num_rnn_hidden: Size of the RNN layers used in the inference network.
    :param num_rnn_layers: Number of RNN layers used in the inference network.
    :param dropout_keep_prob: Dropout probability. Default is None.
    :param num_out_hidden: Size of output layer in inference network.
    :param num_out_layers: Size of output layer in inference network.

    ** Key attributes **
    
    After fitting the model, the key model parameters are stored in the following attributes:
    
    :attr states_mean: Mean of each observation in each of the num_states states.
    :attr states_covars: Covariance matrices of observations.
    :attr transition_matrix: Baseline Markov transition matrix for the attentive state space.
    :attr intial probabilities: Initial distribution of states averaged accross all trajectories in training data.
    
    ** Key methods **
    
    Three key methods are implemented in the API:
    
    :method fit: Takes a list of observations and fits an attentive state space model in an unsupervised fashion.
    :method predict: Takes a new observation and returns three variables:
                     - Prediction of the next state at every time step.
                     - Expected observation at the next time tep.
                     - List of attention weights assigned to previous states at every time step.
    :method sample: This method samples synthetic trajectories from the model.

    '''
    
    def __init__(self, 
                 maximum_seq_length, 
                 input_dim, 
                 num_states=3,
                 inference_network='Seq2SeqAttention', 
                 rnn_type='LSTM',
                 unsupervised=True,
                 generative=True,
                 irregular=False,
                 multitask=False,
                 input_name="Input", 
                 output_name="Output",
                 model_name="SeqModel",
                 num_iterations=50, 
                 num_epochs=10, 
                 batch_size=100, 
                 learning_rate=5*1e-4, 
                 num_rnn_hidden=100, 
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=100, 
                 num_out_layers=1,
                 verbosity=True,
                 **kwargs
                ):
        
        # Set all model variables

        self.maximum_seq_length = maximum_seq_length 
        self.input_dim          = input_dim
        self.num_states         = num_states
        self.inference_network  = inference_network
        self.rnn_type           = rnn_type
        self.unsupervised       = unsupervised
        self.generative         = generative
        self.irregular          = irregular
        self.multitask          = multitask
        self.input_name         = input_name 
        self.output_name        = output_name 
        self.model_name         = model_name
        self.num_iterations     = num_iterations
        self.num_epochs         = num_epochs
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.num_rnn_hidden     = num_rnn_hidden
        self.num_rnn_layers     = num_rnn_layers
        self.dropout_keep_prob  = dropout_keep_prob
        self.num_out_hidden     = num_out_hidden
        self.num_out_layers     = num_out_layers
        self.verbosity          = verbosity
        
        
        self.build_attentive_inference_network()
        
        tf.reset_default_graph()
        
        self.build_attentive_inference_graph()
        
        
    
    def build_attentive_inference_network(self):
        
        # replace this with dictionary style indexing
        
        model_options_names     = ['RNN','LSTM','GRU','PhasedLSTM']
        
        optimizer_options_names = []
        
        
        model_options   = [BasicRNNCell(self.num_rnn_hidden), 
                           rnn_cell.LSTMCell(self.num_rnn_hidden), 
                           rnn_cell.GRUCell(self.num_rnn_hidden), 
                           PhasedLSTMCell(self.num_rnn_hidden)]
        
        self._rnn_model = model_options[np.where(np.array(model_options_names)==self.rnn_type)[0][0]]
        
        if self.dropout_keep_prob is not None:
            
            self._rnn_model = tf.nn.rnn_cell.DropoutWrapper(self._rnn_model, output_keep_prob=self.dropout_keep_prob)
        
        self._Losses = []
        

    def build_attentive_inference_graph(self):
        
        self.observation = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.input_dim], 
                                     name=self.input_name)
            
        self.state_guess = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.num_states]) 
        
        if self.irregular:
            
            self.times      = tf.placeholder(tf.float32, [None, self.maximum_seq_length, 1])
            self.rnn_input  = (self.times, self.observation)
        
        else:
            
            self.rnn_input  = self.observation

            
    @lazy_property
    def length(self):
        
        used   = tf.sign(tf.reduce_max(tf.abs(self.observation), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        
        return length

    @lazy_property
    def forward(self):
        
        self.attentive_inference_network_inputs()
        
        # Recurrent network.   
        if self.inference_network != 'Seq2SeqAttention': 
            
            rnn_output, _  = rnn.dynamic_rnn(self._rnn_model, 
                                             self.rnn_input_, 
                                             dtype=tf.float32, 
                                             sequence_length=self.length_,)
        else:
            
            try:
                
                tf.nn.seq2seq = tf.contrib.legacy_seq2seq
                tf.nn.rnn_cell = tf.contrib.rnn
                tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell

                if self.verbosity:
                    
                    print("TensorFlow version : >= 1.0")
            
            except: 
            
                print("TensorFlow version : 0.12")
            
            if self.verbosity:
                
                print("---------------------------")
            
            self.enc_inp    = [self.rnn_input_[:, t, :] for t in range(self.maximum_seq_length)]

            self.dec_output = [tf.placeholder(tf.float32, shape=(None, 1), 
                                              name="dec_output_".format(t)) for t in range(self.maximum_seq_length)]

            self.dec_inp    = [tf.zeros_like(self.enc_inp[0], dtype=np.float32, name="GO")] + self.enc_inp[:-1] 

            self.cells = []
    
            for i in range(self.num_rnn_layers):
                
                with tf.variable_scope('RNN_{}'.format(i)):
                    
                    self.cells.append(tf.nn.rnn_cell.LSTMCell(self.num_rnn_hidden))
            
            
            self.cell  = tf.nn.rnn_cell.MultiRNNCell(self.cells)
            self.dec_outputs, self.dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell) 
            
            self.weight_dec, self.bias_dec = self._weight_and_bias(self.num_rnn_hidden, 1, ["w_dec", "b_dec"])
            
            self.seq2seq_attn = [(tf.matmul(i, self.weight_dec) + self.bias_dec) for i in self.dec_outputs]
            self.seq2seq_attn = tf.nn.softmax(tf.reshape(tf.stack(self.seq2seq_attn), 
                                                         [-1, self.maximum_seq_length, 1]), axis=1)
            
        
        # Softmax layer.
        self.combiner_func_weight, self.combiner_func_bias = self._weight_and_bias(self.input_dim, 
                                                                                   self.num_out_hidden, 
                                                                                   ["w_0", "b_0"])
        
        self.weight, self.bias     = self._weight_and_bias(self.num_out_hidden, 
                                                           self.num_states, 
                                                           ["w", "b"])
            
        # Flatten to apply same weights to all time steps.
        
        if self.inference_network not in ['RETAIN', 'Seq2SeqAttention']: 
            
            rnn_output  = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            forward     = tf.nn.softmax(tf.matmul(rnn_output, self.weight) + self.bias)
        
        elif self.inference_network == 'RETAIN':
            
            self.weight_a, self.bias_a = self._weight_and_bias(self.num_out_hidden, 1, ["w_a", "b_a"])
            
            rnn_output      = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            self.attention  = tf.nn.softmax(tf.reshape(tf.matmul(rnn_output, self.weight_a) + self.bias_a, 
                                                       [-1, self.maximum_seq_length, 1]), axis=1)
            

            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            
            context_layer   = tf.matmul(self.context, self.combiner_func_weight ) + self.combiner_func_bias
            forward         = tf.nn.softmax(tf.matmul(context_layer, self.weight) + self.bias)
        
        elif self.inference_network == 'Seq2SeqAttention':
            
            self.attention  = self.seq2seq_attn
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)            
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.combiner_func_weight ) + self.combiner_func_bias
            forward         = tf.nn.softmax(tf.matmul(context_layer, self.weight) + self.bias)

        forward         = tf.reshape(forward, [-1, self.maximum_seq_length, self.num_states])
        
        self.predicted  = forward
        self.predicted  = tf.identity(self.predicted, name=self.output_name)
        
        return forward

    
    def attentive_inference_network_inputs(self):
        
        if self.inference_network in ['RETAIN', 'Seq2SeqAttention']: 
            
            self.num_samples = tf.shape(self.observation)[0] 
             
            Lengths_         = np.repeat(self.length, self.maximum_seq_length)
            
            conv_data        = tf.reshape(tf.tile(self.observation, [1, self.maximum_seq_length, 1]), 
                                          [self.maximum_seq_length * self.num_samples, 
                                           self.maximum_seq_length, self.input_dim])
            
            conv_mask_       = tf.ones([self.maximum_seq_length, self.maximum_seq_length], tf.float32)
            
            conv_mask        = tf.tile(tf.expand_dims(tf.tile(tf.matrix_band_part(conv_mask_, -1, 0), 
                                                              [self.num_samples, 1]), 2), 
                                                              [1, 1, self.input_dim])
            
            masked_data   = tf.multiply(conv_data, conv_mask)
            
            Seq_lengths_  = tf.tile(tf.range(1, self.maximum_seq_length + 1, 1), [self.num_samples])
            
            if self.inference_network == 'RETAIN':
            
                self.rnn_input_  = tf.reverse_sequence(masked_data, batch_axis=0, seq_dim=1, 
                                                       seq_lengths=Seq_lengths_, seq_axis=None)
            else:
                
                self.rnn_input_  = masked_data
                
            
            used         = tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2))
            length       = tf.reduce_sum(used, reduction_indices=1)
            self.length_ = tf.cast(length, tf.int32)

            
        else:    
            
            self.rnn_input_ = self.rnn_input
            self.length_    = self.length

            
    @lazy_property
    def ELBO(self):

        mask                = tf.sign(tf.reduce_max(tf.abs(self.observation), reduction_indices=2))
        flat_mask           = tf.reshape(mask, [-1,1])
        
        flat_state_guess    = tf.reshape(self.state_guess, [-1, self.num_states])      
        flat_forward        = tf.reshape(self.forward, [-1, self.num_states])

        likelihood_loss     = tf.reduce_sum(-1*(flat_state_guess * tf.log(flat_forward)))
        
        self.mask           = mask
        
        # Average over actual sequence lengths. << Did I forget masking padded ELBOs? >>
        likelihood_loss     /= tf.reduce_sum(tf.cast(self.length, tf.float32),reduction_indices=0)
        
        return likelihood_loss        
            
        
    def initialize_hidden_states(self, X):
        
        self.init_states = GaussianMixture(n_components=self.num_states, 
                                           covariance_type='full')
        
        self.init_states.fit(np.concatenate(X).reshape((-1, self.input_dim))) 
        
        
    def get_likelihood(self, X, pred):
        
        likelihoods_ = []
        
        #for u in range(X.shape[0]):
    
            #likel_ = np.log(np.array([multivariate_normal.pdf(X[u,:], self.state_means[k], self.state_covars[k])*pred[u,k] for k in range(self.num_states)]))
        #    likel_ = np.array([multivariate_normal.logpdf(X[u,:], self.state_means[k], self.state_covars[k])*pred[u,k] for k in range(self.num_states)])
        
        #likel_ = [np.array([multivariate_normal.logpdf(X[u,:], self.state_means[k], self.state_covars[k])*pred[u,k] for k in range(self.num_states)]) for u in range(X.shape[0])]
        #likelihoods_  = np.array(likel_)
        
        #Faster vectorized implementation
 
        XX    = X.reshape((-1, self.input_dim))
        
        lks_  = np.array([multivariate_normal.logpdf(XX, self.state_means[k], self.state_covars[k]).reshape((-1,1))*pred[:,k].reshape((-1,1)) for k in range(self.num_states)])

        likelihoods_  = lks_
        
        return np.mean(likelihoods_[np.isfinite(likelihoods_)])
    
    
    def sample_posterior_states(self, q_posterior):
        
        sampled_list = [state_to_array(np.random.choice(self.num_states, 1, p=q_posterior[k,:])[0], self.num_states) for k in range(q_posterior.shape[0])]
        self.state_trajectories = np.array(sampled_list)
    
    def fit(self, X, T=None):
        
        self.state_trajectories_ = []
        
        self.initialize_hidden_states(X)

        state_inferences_init  = [np.argmax(self.init_states.predict_proba(X[k]), axis=1) for k in range(len(X))]
        self.all_states        = state_inferences_init

        for v in range(len(state_inferences_init)):
            
            state_list = [state_to_array(state_inferences_init[v][k], self.num_states) for k in range(len(state_inferences_init[v]))]
            delayed_traject = np.vstack((np.array(state_list)[1:, :], np.array(state_list)[-1, :]))
            
            self.state_trajectories_.append(delayed_traject)
            
            
        self.normalizer   = StandardScaler()
        self.normalizer.fit(np.concatenate(X))

        self.X_normalized  = []

        for k in range(len(X)):
            
            self.X_normalized.append(self.normalizer.transform(X[k])) 
    

        self.stochastic_variational_inference(self.X_normalized)

        
    def stochastic_variational_inference(self, X, T=None):
        
        X_, state_update = padd_data(X, self.maximum_seq_length), padd_data(self.state_trajectories_, self.maximum_seq_length)
        
        if T is not None:
            T_   = padd_data(T, self.maximum_seq_length)
            
            
        # Baseline transition matrix

        initial_states = np.array([self.all_states[k][0] for k in range(len(self.all_states))])
        init_probs     = [np.where(initial_states==k)[0].shape[0] / len(initial_states) for k in range(self.num_states)]

        transits   = np.zeros((self.num_states, self.num_states))
        each_state = np.zeros(self.num_states)

        for _ in range(len(self.all_states)):
    
            new_trans, new_each_state = get_transitions(self.all_states[_], self.num_states)
    
            transits   += new_trans
            each_state += new_each_state
    
        for _ in range(self.num_states):
    
            transits[_, :] = transits[_, :] / each_state[_]
            transits[_, :] = transits[_, :] / np.sum(transits[_, :])
    
        self.initial_probabilities = np.array(init_probs)
        self.transition_matrix     = np.array(transits)
        
        # -----------------------------------------------------------
        # Observational distribution
        # -----------------------------------------------------------
        
        self.state_means  = self.init_states.means_
        self.state_covars = self.init_states.covariances_    
        
        
        sess = tf.InteractiveSession()
        
        opt      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.ELBO)
        init     = tf.global_variables_initializer()
        
        sess.run(init)

        saver = tf.train.Saver()
        
        for epoch in range(self.num_epochs):
                
            for _ in range(self.num_iterations):
                
                batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train   = X_[batch_samples,:,:]
                batch_states  = state_update[batch_samples,:,:]
                
                train_dict    = {self.observation : batch_train,
                                 self.state_guess : batch_states}
                
                batch_preds   = sess.run(self.forward, feed_dict=train_dict)
                
                # sample and update posterior states
                self.sample_posterior_states(batch_preds.reshape((-1, self.num_states)))

                sess.run(opt, feed_dict=train_dict)
                
                Loss = sess.run(self.ELBO, feed_dict=train_dict)
                
                log_likelihood_ = np.array([self.get_likelihood(batch_train[k,:,:], batch_preds[k,:,:]) for k in range(batch_train.shape[0])]) 
                log_likelihood_ = np.sum(log_likelihood_)/self.batch_size
                
                self._Losses.append(log_likelihood_)
 
                # Verbosity function
    
                if self.verbosity:
            
                    print('Epoch %d \t----- \tBatch %d \t----- \tLog-Likelihood %10.6e' % (epoch, _, log_likelihood_))
                
        
        # Save model
        saver.save(sess, "./mlaimRNN_model") 
        
        if os.path.exists("attentive_state_space"):
            
            shutil.rmtree("attentive_state_space")
        
        tf.saved_model.simple_save(sess, export_dir='attentive_state_space', inputs={"myInput": self.observation}, 
                                   outputs={"myOutput": self.predicted})    
        
           

    def predict(self, X):
        
        X_normalized  = []

        for k in range(len(X)):
            
            X_normalized.append(self.normalizer.transform(X[k])) 
        
        with tf.Session() as sess:
            
            saver           = tf.train.import_meta_graph("mlaimRNN_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            
            preds_lengths   = [len(X_normalized[k]) for k in range(len(X_normalized))]
            
            X_pred          = padd_data(X_normalized, padd_length=self.maximum_seq_length)
            pred_dict       = {self.observation  : X_pred}
            
            prediction_     = sess.run(self.forward, pred_dict).reshape([-1, self.maximum_seq_length, self.num_states])         

            preds_          = []
            obs_            = []
            
            for k in range(len(X)):
                
                preds_.append(prediction_[k, 0 : preds_lengths[k]])
                obs_.append(self.get_observations(preds_[-1]))

            if self.inference_network in ['RETAIN', 'Seq2SeqAttention']:
                
                
                attn_                  = sess.run(self.attention, pred_dict) 
                attn_per_patient       = [attn_[u * self.maximum_seq_length : u * self.maximum_seq_length + self.maximum_seq_length, :, :] for u in range(len(X))]
                attn_lists_per_patient = [[attn_per_patient[u][k, 0 : k + 1, :] for k in range(self.maximum_seq_length)] for u in range(len(X))]
                
                all_preds_             = (preds_, obs_, attn_lists_per_patient)    

            else:
                
                all_preds_             = (preds_, obs_) 

                
        return all_preds_ 
    
    
    def get_observations(self, preds):
        
        pred_obs     = []
    
        for v in range(preds.shape[0]):
        
            observations = np.zeros(self.input_dim)
    
            for k in range(self.num_states):
            
                observations += self.state_means[k] * preds[v, k] 
    
            pred_obs.append(observations) 
    
    
        return np.array(pred_obs)
    
    
    def sample(self, trajectory_length=5):
        
        initial_state    = np.random.choice(self.num_states, 1, 
                                            p=self.initial_probabilities)[0]
    
        State_trajectory      = [initial_state]
        first_observation     = np.random.multivariate_normal(self.state_means[initial_state], 
                                                              self.state_covars[initial_state]) 
    
        Obervation_trajectory = [first_observation.reshape((1,-1))]
    
        for _ in range(trajectory_length):
        
            next_state_pred  = self.predict(Obervation_trajectory)[0][0][0]
            next_state       = np.random.choice(self.num_states, 1, p=next_state_pred)[0]
        
            State_trajectory.append(next_state)
        
            next_observation = np.random.multivariate_normal(self.state_means[next_state], 
                                                         self.state_covars[next_state]).reshape((1,-1))
        
        
            Obervation_trajectory[0] = np.vstack((Obervation_trajectory[0], next_observation))
        
    
        return State_trajectory, Obervation_trajectory   
    
    
    @staticmethod
    def _weight_and_bias(in_size, out_size, wnames):
    
        weight = tf.get_variable(wnames[0], 
                                 shape=[in_size, out_size], 
                                 initializer=tf.contrib.layers.xavier_initializer())
        
        bias   = tf.get_variable(wnames[1], 
                                 shape=[out_size], 
                                 initializer=tf.contrib.layers.xavier_initializer())
        
        return weight, bias
        



class SeqModel:
    
    '''
    Parent class for all RNN models.

    '''
    
    def __init__(self, 
                 maximum_seq_length, 
                 input_dim, 
                 output_dim=1,
                 model_type='RNN',
                 rnn_type='RNN',
                 latent=False,
                 generative=False,
                 irregular=False,
                 multitask=False,
                 prediction_mode='Sequence_labeling',
                 input_name="Input", 
                 output_name="Output",
                 model_name="SeqModel",
                 num_iterations=20, 
                 num_epochs=10, 
                 batch_size=100, 
                 learning_rate=0.0005, 
                 num_rnn_hidden=200, 
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=200, 
                 num_out_layers=1,
                 **kwargs
                ):
        
        # Set all model variables

        self.maximum_seq_length = maximum_seq_length 
        self.input_dim          = input_dim
        self.output_dim         = output_dim
        self.model_type         = model_type
        self.rnn_type           = rnn_type
        self.latent             = latent
        self.generative         = generative
        self.irregular          = irregular
        self.multitask          = multitask
        self.prediction_mode    = prediction_mode
        self.input_name         = input_name 
        self.output_name        = output_name 
        self.model_name         = model_name
        self.num_iterations     = num_iterations
        self.num_epochs         = num_epochs
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.num_rnn_hidden     = num_rnn_hidden
        self.num_rnn_layers     = num_rnn_layers
        self.dropout_keep_prob  = dropout_keep_prob
        self.num_out_hidden     = num_out_hidden
        self.num_out_layers     = num_out_layers
        
        
        self.build_rnn_model()
        tf.reset_default_graph()
        self.build_rnn_graph()
        
        
    
    def build_rnn_model(self):
        
        # replace this with dictionary style indexing
        
        model_options_names     = ['RNN','LSTM','GRU','PhasedLSTM']
        
        optimizer_options_names = []
        
        
        model_options   = [BasicRNNCell(self.num_rnn_hidden), rnn_cell.LSTMCell(self.num_rnn_hidden), 
                                   rnn_cell.GRUCell(self.num_rnn_hidden), PhasedLSTMCell(self.num_rnn_hidden)]
        
        self._rnn_model = model_options[np.where(np.array(model_options_names)==self.rnn_type)[0][0]]
        
        if self.dropout_keep_prob is not None:
            
            self._rnn_model = tf.nn.rnn_cell.DropoutWrapper(self._rnn_model, output_keep_prob=self.dropout_keep_prob)
        
        self._Losses = []
        

    def build_rnn_graph(self):
        
        self.data   = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.input_dim], 
                                     name=self.input_name)
            
        self.target = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.output_dim]) 
        
        if self.irregular:
            
            self.times      = tf.placeholder(tf.float32, [None, self.maximum_seq_length, 1])
            self.rnn_input  = (self.times, self.data)
        
        else:
            
            self.rnn_input  = self.data 

            
    @lazy_property
    def length(self):
        
        used   = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        
        return length

    @lazy_property
    def prediction(self):
        
        self.process_rnn_inputs()
        
        # Recurrent network.   
        if self.model_type != 'Seq2SeqAttention': 
            
            rnn_output, _  = rnn.dynamic_rnn(self._rnn_model, 
                                             self.rnn_input_, 
                                             dtype=tf.float32, 
                                             sequence_length=self.length_,)
        else:
            
            try:
                
                tf.nn.seq2seq = tf.contrib.legacy_seq2seq
                tf.nn.rnn_cell = tf.contrib.rnn
                tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
                print("TensorFlow version : >= 1.0")
            
            except: 
            
                print("TensorFlow version : 0.12")
            
            self.enc_inp    = [self.rnn_input_[:, t, :] for t in range(self.maximum_seq_length)]

            self.dec_output = [tf.placeholder(tf.float32, shape=(None, 1), 
                                              name="dec_output_".format(t)) for t in range(self.maximum_seq_length)]

            self.dec_inp    = [tf.zeros_like(self.enc_inp[0], dtype=np.float32, name="GO")] + self.enc_inp[:-1] 

            self.cells = []
    
            for i in range(self.num_rnn_layers):
                
                with tf.variable_scope('RNN_{}'.format(i)):
                    
                    self.cells.append(tf.nn.rnn_cell.GRUCell(self.num_rnn_hidden))
            
            
            self.cell  = tf.nn.rnn_cell.MultiRNNCell(self.cells)
            self.dec_outputs, self.dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell) 
            
            self.weight_dec, self.bias_dec = self._weight_and_bias(self.num_rnn_hidden, self.output_dim, ["w_dec", "b_dec"])
            
            self.seq2seq_attn = [(tf.matmul(i, self.weight_dec) + self.bias_dec) for i in self.dec_outputs]
            self.seq2seq_attn = tf.nn.softmax(tf.reshape(tf.stack(self.seq2seq_attn), 
                                                         [-1, self.maximum_seq_length, 1]), axis=1)
            
        
        # Softmax layer.
        self.weight_0, self.bias_0 = self._weight_and_bias(self.input_dim, 
                                                           self.num_out_hidden, 
                                                           ["w_0", "b_0"])
        
        self.weight, self.bias     = self._weight_and_bias(self.num_out_hidden, 
                                                           self.output_dim, 
                                                           ["w", "b"])
            
        # Flatten to apply same weights to all time steps.
        
        if self.model_type not in ['RETAIN', 'Seq2SeqAttention']: 
            
            rnn_output  = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            prediction  = tf.nn.softmax(tf.matmul(rnn_output, self.weight) + self.bias)
        
        elif self.model_type == 'RETAIN':
            
            self.weight_a, self.bias_a = self._weight_and_bias(self.num_out_hidden, 1, ["w_a", "b_a"])
            
            rnn_output      = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            self.attention  = tf.nn.softmax(tf.reshape(tf.matmul(rnn_output, self.weight_a) + self.bias_a, 
                                                       [-1, self.maximum_seq_length, 1]), axis=1)
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.nn.softmax(tf.matmul(context_layer, self.weight) + self.bias)
        
        elif self.model_type == 'Seq2SeqAttention':
            
            self.attention  = self.seq2seq_attn
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.nn.softmax(tf.matmul(context_layer, self.weight) + self.bias)

        prediction      = tf.reshape(prediction, [-1, self.maximum_seq_length, self.output_dim])
        self.predicted  = prediction
        self.predicted  = tf.identity(self.predicted, name=self.output_name)
        
        return prediction

    
    def process_rnn_inputs(self):
        
        if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
            
            self.num_samples = tf.shape(self.data)[0]
             
            Lengths_         = np.repeat(self.length, self.maximum_seq_length)
            
            conv_data        = tf.reshape(tf.tile(self.data, [1, self.maximum_seq_length, 1]), 
                                          [self.maximum_seq_length * self.num_samples, 
                                           self.maximum_seq_length, self.input_dim])
            
            conv_mask_       = tf.ones([self.maximum_seq_length, self.maximum_seq_length], tf.float32)
            
            conv_mask        = tf.tile(tf.expand_dims(tf.tile(tf.matrix_band_part(conv_mask_, -1, 0), 
                                                              [self.num_samples, 1]), 2), 
                                                              [1, 1, self.input_dim])
            
            masked_data   = tf.multiply(conv_data, conv_mask)
            
            Seq_lengths_  = tf.tile(tf.range(1, self.maximum_seq_length + 1, 1), [self.num_samples])
            
            if self.model_type == 'RETAIN':
            
                self.rnn_input_  = tf.reverse_sequence(masked_data, batch_axis=0, seq_dim=1, 
                                                       seq_lengths=Seq_lengths_, seq_axis=None)
            else:
                
                self.rnn_input_  = masked_data
                
            
            used         = tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2))
            length       = tf.reduce_sum(used, reduction_indices=1)
            self.length_ = tf.cast(length, tf.int32)
            
            self.target_ = tf.tile(self.target, [self.maximum_seq_length, 1, 1]) 
            
        else:    
            
            self.rnn_input_ = self.rnn_input
            self.target_    = self.target 
            self.length_    = self.length

    @lazy_property
    def loss(self):
        
        # Compute cross entropy for each frame.
        cross_entropy  = tf.reduce_sum(-1*(self.target * tf.log(self.prediction) + (1-self.target)*(tf.log(1-self.prediction))),
                                       reduction_indices=2) 

        mask           = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        cross_entropy *= mask
        self.mask      = mask
        
        # Average over actual sequence lengths.
        cross_entropy  = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        
        return tf.reduce_mean(cross_entropy)
        
    
    
    def train(self, X, Y, T=None):
        
        X_, Y_   = padd_data(X, self.maximum_seq_length), padd_data(Y, self.maximum_seq_length)
        
        if T is not None:
            T_   = padd_data(T, self.maximum_seq_length)
        
        
        sess = tf.InteractiveSession()
        
        opt      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init     = tf.global_variables_initializer()
        
        sess.run(init)

        saver = tf.train.Saver()
        
        for epoch in range(self.num_epochs):
                
            for _ in range(self.num_iterations):
                
                batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train   = X_[batch_samples,:,:]
                batch_targets = Y_[batch_samples,:,:] 
                
                if T is not None:
                    batch_times = T_[batch_samples,:,:]
                    
                    train_dict  = {self.data   : batch_train,
                                   self.target : batch_targets,
                                   self.times  : batch_times}
                else:
                    
                    train_dict  = {self.data   : batch_train,
                                   self.target : batch_targets}
                
                
                sess.run(opt, feed_dict=train_dict)
                
                Loss          = sess.run(self.loss, feed_dict=train_dict)
                
                self._Losses.append(Loss)
 
                # Visualize function
                print('Epoch {} \t----- \tBatch {} \t----- \tLoss {}'.format(epoch, _, self._Losses[-1]))
  
        # change names
        saver.save(sess, "./mlaimRNN_model") 
        
        if os.path.exists("modelgraph"):
            
            shutil.rmtree("modelgraph")
        
        tf.saved_model.simple_save(sess, export_dir='modelgraph', inputs={"myInput": self.data}, 
                                   outputs={"myOutput": self.predicted})    
        
            
    def predict(self, X, T=None):
        
        with tf.Session() as sess:
            
            saver           = tf.train.import_meta_graph("mlaimRNN_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            
            preds_lengths   = [len(X[k]) for k in range(len(X))]
            
            X_pred          = padd_data(X, padd_length=self.maximum_seq_length)
            
            if T is not None:
                T_pred      = padd_data_enforce(T, padd_length=self.maximum_seq_length)
                pred_dict   = {self.data   : X_pred, self.times   : T_pred}
            else:
                pred_dict   = {self.data   : X_pred}
            
            prediction_     = sess.run(self.prediction, pred_dict).reshape([-1, self.maximum_seq_length, 1])         

            preds_          = []
            
            for k in range(len(X)):
                
                preds_.append(prediction_[k, 0 : preds_lengths[k]])
                
                
            if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
                
                attn_                  = sess.run(self.attention, pred_dict) 
                attn_per_patient       = [attn_[u * self.maximum_seq_length : u * self.maximum_seq_length + self.maximum_seq_length, :, :] for u in range(len(X))]
                attn_lists_per_patient = [[attn_per_patient[u][k, 0 : k + 1, :] for k in range(self.maximum_seq_length)] for u in range(len(X))]
                
                preds_                 = (preds_, attn_lists_per_patient)
            
        return preds_    
    
    
    def evaluate(self, preds, Y_test):
        
        flat_preds   = flatten_sequences_to_numpy(preds)
        flat_Y_test  = np.array(list(itertools.chain.from_iterable([Y_test[k].tolist() for k in range(len(Y_test))])))
        
        _performance = roc_auc_score(flat_Y_test, flat_preds)
        
        return _performance
    
    @staticmethod
    def _weight_and_bias(in_size, out_size, wnames):
    
        weight = tf.get_variable(wnames[0], shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.get_variable(wnames[1], shape=[out_size], initializer=tf.contrib.layers.xavier_initializer())
        
        return weight, bias
        


class sequence_prediction:
    
    '''
    Parent class for all RNN models.

    '''
    
    def __init__(self, 
                 maximum_seq_length, 
                 input_dim, 
                 model_type='RNN',
                 rnn_type='RNN',
                 input_name="Input", 
                 output_name="Output",
                 model_name="SeqModel",
                 num_iterations=20, 
                 num_epochs=10, 
                 batch_size=100, 
                 learning_rate=0.0005, 
                 num_rnn_hidden=200, 
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=200, 
                 num_out_layers=1,
                 verbosity=True,
                 **kwargs
                ):
        
        # Set all model variables

        self.maximum_seq_length = maximum_seq_length 
        self.input_dim          = input_dim
        self.output_dim         = input_dim
        self.model_type         = model_type
        self.rnn_type           = rnn_type
        
        self.input_name         = input_name 
        self.output_name        = output_name 
        self.model_name         = model_name
        self.num_iterations     = num_iterations
        self.num_epochs         = num_epochs
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.num_rnn_hidden     = num_rnn_hidden
        self.num_rnn_layers     = num_rnn_layers
        self.dropout_keep_prob  = dropout_keep_prob
        self.num_out_hidden     = num_out_hidden
        self.num_out_layers     = num_out_layers
        self.verbosity          = verbosity
        
        
        self.build_rnn_model()
        tf.reset_default_graph()
        self.build_rnn_graph()
        
        
    
    def build_rnn_model(self):
        
        # replace this with dictionary style indexing
        
        model_options_names     = ['RNN','LSTM','GRU','PhasedLSTM']
        
        optimizer_options_names = []
        
        
        model_options   = [BasicRNNCell(self.num_rnn_hidden), rnn_cell.LSTMCell(self.num_rnn_hidden), 
                                   rnn_cell.GRUCell(self.num_rnn_hidden), PhasedLSTMCell(self.num_rnn_hidden)]
        
        self._rnn_model = model_options[np.where(np.array(model_options_names)==self.rnn_type)[0][0]]
        
        if self.dropout_keep_prob is not None:
            
            self._rnn_model = tf.nn.rnn_cell.DropoutWrapper(self._rnn_model, output_keep_prob=self.dropout_keep_prob)
        
        self._Losses = []
        

    def build_rnn_graph(self):
        
        self.data   = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.input_dim], 
                                     name=self.input_name)
            
        self.target = tf.placeholder(tf.float32, 
                                     [None, self.maximum_seq_length, self.output_dim]) 
        
            
        self.rnn_input  = self.data 

            
    @lazy_property
    def length(self):
        
        used   = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        
        return length

    @lazy_property
    def prediction(self):
        
        self.process_rnn_inputs()
        
        # Recurrent network.   
        if self.model_type != 'Seq2SeqAttention': 
            
            rnn_output, _  = rnn.dynamic_rnn(self._rnn_model, 
                                             self.rnn_input_, 
                                             dtype=tf.float32, 
                                             sequence_length=self.length_,)
        else:
            
            try:
                
                tf.nn.seq2seq = tf.contrib.legacy_seq2seq
                tf.nn.rnn_cell = tf.contrib.rnn
                tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
                print("TensorFlow version : >= 1.0")
            
            except: 
            
                print("TensorFlow version : 0.12")
            
            self.enc_inp    = [self.rnn_input_[:, t, :] for t in range(self.maximum_seq_length)]

            self.dec_output = [tf.placeholder(tf.float32, shape=(None, 1), 
                                              name="dec_output_".format(t)) for t in range(self.maximum_seq_length)]

            self.dec_inp    = [tf.zeros_like(self.enc_inp[0], dtype=np.float32, name="GO")] + self.enc_inp[:-1] 

            self.cells = []
    
            for i in range(self.num_rnn_layers):
                
                with tf.variable_scope('RNN_{}'.format(i)):
                    
                    self.cells.append(tf.nn.rnn_cell.GRUCell(self.num_rnn_hidden))
            
            
            self.cell  = tf.nn.rnn_cell.MultiRNNCell(self.cells)
            self.dec_outputs, self.dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell) 
            
            self.weight_dec, self.bias_dec = self._weight_and_bias(self.num_rnn_hidden, self.output_dim, ["w_dec", "b_dec"])
            
            self.seq2seq_attn = [(tf.matmul(i, self.weight_dec) + self.bias_dec) for i in self.dec_outputs]
            self.seq2seq_attn = tf.nn.softmax(tf.reshape(tf.stack(self.seq2seq_attn), 
                                                         [-1, self.maximum_seq_length, 1]), axis=1)
            
        
        # Softmax layer.
        self.weight_0, self.bias_0 = self._weight_and_bias(self.input_dim, 
                                                           self.num_out_hidden, 
                                                           ["w_0", "b_0"])
        
        self.weight, self.bias     = self._weight_and_bias(self.num_out_hidden, 
                                                           self.output_dim, 
                                                           ["w", "b"])
            
        # Flatten to apply same weights to all time steps.
        
        if self.model_type not in ['RETAIN', 'Seq2SeqAttention']: 
            
            rnn_output  = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            prediction  = tf.matmul(rnn_output, self.weight) + self.bias
        
        elif self.model_type == 'RETAIN':
            
            self.weight_a, self.bias_a = self._weight_and_bias(self.num_out_hidden, 1, ["w_a", "b_a"])
            
            rnn_output      = tf.reshape(rnn_output, [-1, self.num_out_hidden])
            
            self.attention  = tf.nn.softmax(tf.reshape(tf.matmul(rnn_output, self.weight_a) + self.bias_a, 
                                                       [-1, self.maximum_seq_length, 1]), axis=1)
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn #/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.matmul(context_layer, self.weight) + self.bias
        
        elif self.model_type == 'Seq2SeqAttention':
            
            self.attention  = self.seq2seq_attn
            
            attn_mask       = tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2)), axis=2)
            masked_attn     = tf.multiply(attn_mask, self.attention)
            attn_norms      = tf.expand_dims(tf.tile(tf.reduce_sum(masked_attn, axis=1), [1, self.maximum_seq_length]), axis=2)
            self.attention  = masked_attn/attn_norms
            self.attention_ = tf.tile(self.attention, [1, 1, self.input_dim])
            self.context    = tf.reduce_sum(tf.multiply(self.attention_, self.rnn_input_), reduction_indices=1)
            context_layer   = tf.matmul(self.context, self.weight_0) + self.bias_0
            prediction      = tf.matmul(context_layer, self.weight) + self.bias

        prediction      = tf.reshape(prediction, [-1, self.maximum_seq_length, self.output_dim])
        self.predicted  = prediction
        self.predicted  = tf.identity(self.predicted, name=self.output_name)
        

        return prediction

    
    def process_rnn_inputs(self):
        
        if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
            
            self.num_samples = tf.shape(self.data)[0]
             
            Lengths_         = np.repeat(self.length, self.maximum_seq_length)
            
            conv_data        = tf.reshape(tf.tile(self.data, [1, self.maximum_seq_length, 1]), 
                                          [self.maximum_seq_length * self.num_samples, 
                                           self.maximum_seq_length, self.input_dim])
            
            conv_mask_       = tf.ones([self.maximum_seq_length, self.maximum_seq_length], tf.float32)
            
            conv_mask        = tf.tile(tf.expand_dims(tf.tile(tf.matrix_band_part(conv_mask_, -1, 0), 
                                                              [self.num_samples, 1]), 2), 
                                                              [1, 1, self.input_dim])
            
            masked_data   = tf.multiply(conv_data, conv_mask)
            
            Seq_lengths_  = tf.tile(tf.range(1, self.maximum_seq_length + 1, 1), [self.num_samples])
            
            if self.model_type == 'RETAIN':
            
                self.rnn_input_  = tf.reverse_sequence(masked_data, batch_axis=0, seq_dim=1, 
                                                       seq_lengths=Seq_lengths_, seq_axis=None)
            else:
                
                self.rnn_input_  = masked_data
                
            
            used         = tf.sign(tf.reduce_max(tf.abs(self.rnn_input_), reduction_indices=2))
            length       = tf.reduce_sum(used, reduction_indices=1)
            self.length_ = tf.cast(length, tf.int32)
            
            self.target_ = tf.tile(self.target, [self.maximum_seq_length, 1, 1]) 
            
        else:    
            
            self.rnn_input_ = self.rnn_input
            self.target_    = self.target 
            self.length_    = self.length

    @lazy_property
    def loss(self):

        MSE            = (self.target - self.prediction)**2
        
        return tf.reduce_mean(MSE) 
        
    
    def fit(self, X):
        
        self.normalizer   = MinMaxScaler()
        self.normalizer.fit(np.concatenate(X))

        self.X_normalized  = []
        self.Y_normalized  = []

        for k in range(len(X)):
            
            self.X_normalized.append(self.normalizer.transform(X[k])[:len(X[k])-1,:]) 
            self.Y_normalized.append(self.normalizer.transform(X[k])[1:,:]) 
        
        
        X_  = padd_data(self.X_normalized, self.maximum_seq_length)
        Y_  = padd_data(self.Y_normalized, self.maximum_seq_length)
        
        sess = tf.InteractiveSession()
        
        opt      = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init     = tf.global_variables_initializer()
        
        sess.run(init)

        saver = tf.train.Saver()
        
        for epoch in range(self.num_epochs):
                
            for _ in range(self.num_iterations):
                
                batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train   = X_[batch_samples,:,:]
                batch_targets = Y_[batch_samples,:,:]
                    
                train_dict    = {self.data   : batch_train,
                                 self.target : batch_targets}
                
                
                sess.run(opt, feed_dict=train_dict)
                
                Loss          = sess.run(self.loss, feed_dict=train_dict)
                
                self._Losses.append(Loss)
 
                # Visualize function
                if self.verbosity:
                    print('Epoch {} \t----- \tBatch {} \t----- \tLoss {}'.format(epoch, _, self._Losses[-1]))
  
        # change names
        saver.save(sess, "./mlaimRNN_model") 
        
        if os.path.exists("modelgraph"):
            
            shutil.rmtree("modelgraph")
        
        tf.saved_model.simple_save(sess, export_dir='modelgraph', inputs={"myInput": self.data}, 
                                   outputs={"myOutput": self.predicted})    
        
            
    def predict(self, X, T=None):
        
        with tf.Session() as sess:
            
            saver           = tf.train.import_meta_graph("mlaimRNN_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            
            preds_lengths   = [len(X[k]) for k in range(len(X))]

            self.X_normalized  = []


            for k in range(len(X)):
            
                self.X_normalized.append(self.normalizer.transform(X[k])[:len(X[k])-1,:]) 

            
            X_pred          = padd_data(self.X_normalized, padd_length=self.maximum_seq_length)

            pred_dict       = {self.data   : X_pred}
            
            prediction_     = sess.run(self.prediction, pred_dict).reshape([-1, self.maximum_seq_length, self.output_dim])         

            preds_          = []
            
            for k in range(len(X)):
                
                preds_.append(prediction_[k, 0 : preds_lengths[k], :])
                
                
            if self.model_type in ['RETAIN', 'Seq2SeqAttention']: 
                
                attn_                  = sess.run(self.attention, pred_dict) 
                attn_per_patient       = [attn_[u * self.maximum_seq_length : u * self.maximum_seq_length + self.maximum_seq_length, :, :] for u in range(len(X))]
                attn_lists_per_patient = [[attn_per_patient[u][k, 0 : k + 1, :] for k in range(self.maximum_seq_length)] for u in range(len(X))]
                
                preds_                 = (preds_, attn_lists_per_patient)
            
        return preds_    
    
    
    @staticmethod
    def _weight_and_bias(in_size, out_size, wnames):
    
        weight = tf.get_variable(wnames[0], shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.get_variable(wnames[1], shape=[out_size], initializer=tf.contrib.layers.xavier_initializer())
        
        return weight, bias
        
