
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import io
import json
import os
import sys

import pandas as pd
import numpy as np
import scipy

from sklearn import preprocessing, random_projection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
#SelectFdr, SelectFpr, SelectFwe, SelectPercentile, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

import warnings
warnings.filterwarnings("ignore")



class basePreprocessors:
    
    """ A parent class for all preprocessors. Particular preprocessors. models will inherit from this class."""
    
    def __init__(self):
        
        self.model_type    = 'preprocessor'
        
    def fit_transform(self, X, Y=None):
        
        if Y is None:
            
            X_transformed = self.model.fit_transform(X)
        
        else:
            
            X_transformed = self.model.fit_transform(X, Y)
        
        return X_transformed    
     
    
class Scaler(basePreprocessors): 
    """ Scaler model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Scaler'
        self.model         = preprocessing.StandardScaler()
        self.takes_label   = False
    
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_ 
    
    
class MinMaxScaler(basePreprocessors): 
    """ Min-max scaler model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Min-max Scaler'
        self.model         = preprocessing.MinMaxScaler()
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_     
    
            
class MaxAbsScaler(basePreprocessors): 
    """ Max-abs scaler model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Max-abs Scaler'
        self.model         = preprocessing.MaxAbsScaler()
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_         
        
        
class UniformTransform(basePreprocessors): 
    """ Uniform-Transformer scaler model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Uniform Transformer'
        self.model         = preprocessing.QuantileTransformer()
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_         
                
        
class GaussianTransform(basePreprocessors): 
    """ Gaussian-Transformer scaler model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Gaussian Transformer'
        self.model         = preprocessing.QuantileTransformer(output_distribution='normal')
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_           

    
class FeatureNormalizer(basePreprocessors): 
    """ L2 normalization model for every sample."""
    
    def __init__(self):
        
        super().__init__()
   
        self.name          = 'Normalization'
        self.model         = preprocessing.Normalizer()
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = []

        return hyp_    
    
    
class PolynomialTransform(basePreprocessors): 
    """ Model for generating polynomial features."""
    
    def __init__(self, ndegree=2):
        
        super().__init__()
   
        self.name          = 'Polynomial features'
        self.ndegree       = ndegree
        self.model         = PolynomialFeatures(ndegree)
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'PolynomialFeatures.ndegree', 'type': 'discrete', 'domain': tuple(range(2,5)),'dimensionality': 1}]

        return hyp_  
    
    
class GaussProjection(basePreprocessors): 
    """ Gaussian random projections."""
    
    def __init__(self, maxcomponents=5, ncomponents=2):
        
        super().__init__()
   
        self.name          = 'Gaussian random projections'
        self.ncomponents   = ncomponents
        self.maxcomponents = maxcomponents
        self.model         = random_projection.GaussianRandomProjection(n_components=ncomponents)
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'GaussProjection.ncomponents', 'type': 'discrete', 
                   'domain': tuple(range(2,self.maxcomponents)),'dimensionality': 1}]

        return hyp_      
    
    
class PrincipalComponentAnalysis(basePreprocessors): 
    """ PCA dimensionality reduction."""
    
    def __init__(self, maxcomponents=5, ncomponents=2):
        
        super().__init__()
   
        self.name          = 'PCA'
        self.ncomponents   = ncomponents
        self.maxcomponents = maxcomponents
        self.model         = PCA(n_components=ncomponents)
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'PCA.ncomponents', 'type': 'discrete', 
                   'domain': tuple(range(2, self.maxcomponents)),'dimensionality': 1}]

        return hyp_       

    
class FeatureSelectionVariance(basePreprocessors): 
    """ Feature selection based on variance threshold for dimensionality reduction."""
    
    def __init__(self, threshold=0.001):
        
        super().__init__()
   
        self.name          = 'Feature selection by variance threshold'
        self.threshold     = threshold
        self.model         = VarianceThreshold(threshold=threshold)
        self.takes_label   = False
        
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'FeatureSelectionVariance.threshold', 'type': 'continuous', 
                   'domain': (0,1), 'dimensionality': 1}]

        return hyp_     
    
    
class FeatureSelection(basePreprocessors): 
    """ Select K best dimensionality reduction."""
    
    def __init__(self, maxfeatures=5, nfeatures=2):
        
        super().__init__()
   
        self.name          = 'SelectKBest'
        self.nfeatures     = nfeatures
        self.maxfeatures   = maxfeatures
        self.model         = SelectKBest(chi2, k=nfeatures)
        self.takes_label   = True
        
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'SelectKBest.nfeatures', 'type': 'discrete', 
                   'domain': tuple(range(2, self.maxfeatures)),'dimensionality': 1}]

        return hyp_       

