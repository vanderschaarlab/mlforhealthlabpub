import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.models import load_model, Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, EarlyStopping
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Input, Dense, Dropout
from keras import backend as K
from keras import optimizers
from keras.utils import to_categorical
import keras.optimizers
import pydot
import networkx as nx
from IPython.display import SVG
import glob, os
import pandas as pd
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, accuracy_score
from pycausal import search as s
from pycausal.pycausal import pycausal as pc
from collections import defaultdict
from pycausal import prior as p

pc = pc()
pc.start_vm(java_max_heap_size = '20000M')
tetrad = s.tetradrunner()


# Returns a keras model
# dense is an ordered list of the number of dense neurons like [1024, 2048, 1024]
# dropouts is an ordered list of the dropout masks like [0.2, 0.3, 0.4]
def get_model(dense, dropouts, inputs):
    inputs = keras.Input(shape = (inputs,))
    x = keras.layers.Dense(dense[0], activation = 'relu')(inputs)
    x = keras.layers.Dropout(dropouts[0])(x, training=False)
    for den, drop in zip(dense[1:], dropouts[1:]):
        x = keras.layers.Dense(den, activation = 'relu')(x)
        x = keras.layers.Dropout(drop)(x, training=False)
    outputs = keras.layers.Dense(1, activation = 'linear')(x)
    model = keras.Model(inputs, outputs)
    return model


# This function returns the BIC score, which for graphs and datasets of identical size is proportional to the LL(G|D)
# It takes both a dataframe and a prior (see Tetrad manual)
def get_ll_continuous(df, prior, penalty = 2, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df,  scoreId = 'sem-bic', dataType = 'continuous',
                   structurePrior = 1.0, samplePrior = 1, maxDegree = -1, maxPathLength = -1, priorKnowledge = prior,
                   completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
                   )
    else:
        tetrad.run(algoId = 'gfci', dfs = df, testId = 'sem-bic', scoreId = 'sem-bic', dataType = 'continuous',
                   structurePrior = 1.0, samplePrior = 1, maxDegree = -1, maxPathLength = -1, priorKnowledge = prior,
                   completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
                   )
    ll = tetrad.getTetradGraph().getAllAttributes().toString()
    ll = float(ll.split('=')[-1].split('}')[0])
    return ll 

# This function returns the BIC score, which for graphs and datasets of identical size is proportional to the LL(G|D)
# It takes both a dataframe and a prior (see Tetrad manual)
def get_ll_mixed(df, prior, penalty = 2, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df, scoreId = 'cond-gauss-bic',
               priorKnowledge = prior, dataType = 'mixed', numCategoriesToDiscretize = 5,
               structurePrior = 1.0, maxDegree = -1, faithfulnessAssumed = True, verbose = True)
    else:
        tetrad.run(algoId = 'gfci', dfs = df, scoreId = 'cond-gauss-bic',testId = 'cond-gauss-lrt',
               priorKnowledge = prior, dataType = 'mixed', numCategoriesToDiscretize = 5,
               structurePrior = 1.0, maxDegree = -1, faithfulnessAssumed = True, verbose = True)
    ll = tetrad.getTetradGraph().getAllAttributes().toString()
    ll = float(ll.split('=')[-1].split('}')[0])
    return ll

# This function returns the BIC score, which for graphs and datasets of identical size is proportional to the LL(G|D)
# It takes both a dataframe and a prior (see Tetrad manual)
def get_ll_discrete(df, prior, penalty = 2, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df, scoreId = 'bdeu', priorKnowledge = prior, dataType = 'discrete',
                  structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)
    else:
        tetrad.run(algoId = 'gfci', dfs = df, scoreId = 'bdeu', priorKnowledge = prior, dataType = 'discrete', testId = 'bdeu',
                  structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)
    ll = tetrad.getTetradGraph().getAllAttributes().toString()
    ll = float(ll.split('=')[-1].split('}')[0])
    return ll 

# Normalizes an np array
def normalize(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


# Generates random DAG with nodes and up to 'edges' of edges
def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G


# Examine causal graph
# set method to choose between gfci or fges
# prior is used for prior knowledge - self explanatory below (or see Tetrad documentation and user's manual)
def examine_graph_continuous(df, prior = None, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df,  scoreId = 'sem-bic', dataType = 'continuous',
                   structurePrior = 1.0, samplePrior = 1, maxDegree = -1, maxPathLength = -1, priorKnowledge = prior,
                   completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
                   )
    else:
        tetrad.run(algoId = 'gfci', dfs = df, testId = 'sem-bic', scoreId = 'sem-bic', dataType = 'continuous',
                   structurePrior = 1.0, samplePrior = 1, maxDegree = -1, maxPathLength = -1, priorKnowledge = prior,
                   completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
                   )
    return tetrad.getTetradGraph()

# Examine causal graph continuous and discrete
# set method to choose between gfci or fges
# prior is used for prior knowledge - self explanatory below (or see Tetrad documentation and user's manual)
def examine_graph_mixed(df, prior = None, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df, scoreId = 'cond-gauss-bic',
               priorKnowledge = prior, dataType = 'mixed', numCategoriesToDiscretize = 5,
               structurePrior = 1.0, maxDegree = -1, faithfulnessAssumed = True, verbose = True)
    else:
        tetrad.run(algoId = 'gfci', dfs = df, scoreId = 'cond-gauss-bic',testId = 'cond-gauss-lrt',
               priorKnowledge = prior, dataType = 'mixed', numCategoriesToDiscretize = 5,
               structurePrior = 1.0, maxDegree = -1, faithfulnessAssumed = True, verbose = True)

    return tetrad.getTetradGraph()

# Examine causal graph discrete only
# set method to choose between gfci or fges
# prior is used for prior knowledge - self explanatory below (or see Tetrad documentation and user's manual)
def examine_graph_discrete(df, prior = None, method = 0):
    if method == 0:
        tetrad.run(algoId = 'fges', dfs = df, scoreId = 'bdeu', priorKnowledge = prior, dataType = 'discrete',
                  structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)
    else:
        tetrad.run(algoId = 'gfci', dfs = df, scoreId = 'bdeu', priorKnowledge = prior, dataType = 'discrete', testId = 'bdeu',
                  structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)
    return tetrad.getTetradGraph()


# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It will apply a perturbation at each node provided in perturb.
def gen_data1(list_vertex = [], list_edges = [], mean = 0, var = 1, SIZE = 10000, perturb = []):
    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean,var,SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            g.append(np.random.normal(0,1,SIZE))

    for edge in list_edges:
        g[edge[1]] += g[edge[0]]
    g = np.swapaxes(g,0,1)
    return pd.DataFrame(g, columns = list(map(str, list_vertex)))

# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It does not apply perturbation at a single node, but uniformly by setting the means and variance values
def gen_data2(list_vertex = [], list_edges = [], mean = 0, var = 1, SIZE = 10000):
    g = []
    for v in list_vertex:
        g.append(np.random.normal(mean,var,SIZE))


    for edge in list_edges:
        g[edge[1]] += g[edge[0]]
    g = np.swapaxes(g,0,1)
    return pd.DataFrame(g, columns = list(map(str, list_vertex)))


# Makes the variables in df categorical according to our complete_df, categoricals is the variables to make categorical
def make_categorical(df, complete_df, categoricals):   
    retval = None
    for key in df.columns:
        if retval is not None:
            if key in categoricals:
                retval = np.concatenate((retval, to_categorical(df[key], len(complete_df[key].unique()))), axis = 1)
            else:
                retval = np.concatenate((retval, df[key].values[...,np.newaxis]), axis = 1)
        else:
            if key in categoricals:
                retval = to_categorical(df[key], len(complete_df[key].unique()))
            else:
                retval = df[key]
    return retval
    
    

