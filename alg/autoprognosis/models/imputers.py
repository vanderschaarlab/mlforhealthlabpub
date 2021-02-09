
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np

from sklearn import preprocessing

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

import warnings
warnings.filterwarnings("ignore")


class baseImputer:
    """
    Base class for constructing an imputation method on the pipeline. Default imputer is the mean imputation strategy.
    Includes R wrappers for various imputation methods.
    
    Available imputers:
    ------------------
    sk-learn: mean, median and most frequent imputation strategies.
    R wrappers: MICE, bootstrapped EMB (from AMELIA package), missForest, matrix completion.
    
    Attributes:
    ----------
    _hyperparameters: a dictionary of hyperparameters for the class instance along with their current values.
    _mode: the selected imputation algorithm for this class instance.
    model: imputation model object.
    
    Methods:
    -------
    fit: a method for imputing missing data using the selected algorithm and hyperparameters
    
    """

    def __init__(self,**kwargs):
        """
        Class constructor. 
        Initialize an Imputer object. 
    
        :_mode: imputation algorithm (options: 'mean', 'median', 'most_frequent', 'missForest', 'MICE', 'EMB', 'matrix_completion')
        :_hyperparameters: hyper-parameter setting for the imputer 
        
        """ 
        self._model_list       = ['mean', 'median', 'most_frequent', 'missForest', 'MICE', 'EMB', 'matrix_completion']
        self._hyperparameters  = {} 
        self._mode             = 'mean' 
        self.MI                = False   # multiple-imputation flag, default is FALSE
        self.package_dict      = {'missForest': 'missForest', 'MICE': 'mice',
                                  'EMB': 'Amelia', 'matrix_completion': 'softImpute'}
        self.kwargs            = kwargs

        self.is_init_r_system = False
        
        # Set defaults and catch exceptions
        self.__acceptable_keys_list = ['_mode', '_hyperparameters']
        
        try:
            if(len(kwargs) > 0):
                
                [self.__setattr__(key, kwargs.get(key)) for key in self.__acceptable_keys_list]
            
            #if self._mode not in self._model_list:
                
                #raise ValueError("Unrecognized imputation model! Default mean imputation will be used.")
        
        except ValueError as valerr:
            
            print(valerr)
            self._hyperparameters  = {} 
            self._mode             = 'mean'
        
        # Set imputation model object
        
        self.set_model()
        self.set_hyperparameters()
        
        
    def set_model(self):
        """
        Creates an imputation model object and assigns it to the "model" attribute.
        
        """
        
        if self._mode in ['mean','median','most_frequent']:
            
            self.model   = preprocessing.Imputer(strategy=self._mode)
        
        else:    
            self.collect_Rpackage_imputation_()
            
            r_verbose_library = 'library(' + self.package_dict[self._mode] + ')'
            
            # invoke relevant R library
            r(r_verbose_library)
            
            # Create an imputation object, fill it with some toy data
            _r_init_commands = {'missForest': 'missForest(data.frame(Data = matrix(c(2, 4, 4, 1, 5, 7))))', 
                                'MICE': 'mice(data = data.frame(Data = matrix(c(2, 4, NaN, 1, 5, 7), nrow=3, ncol=2)))', 
                                'EMB': 'amelia(data.frame(Data = matrix(c(2, 4, NaN, 1, 5, 7))), m=5, parallel = "multicore", noms = "Species")',
                                'matrix_completion': 'softImpute(matrix(rnorm(200*50),200,50)%*%matrix(rnorm(50*100),50,100)+matrix(rnorm(200*100),200,100)/5,rank=2,lambda=0)'}

            r('if (!sink.number()) { sink(file="sinkr.txt")}')  # shut up missForest

            r('model <-' + _r_init_commands[self._mode])
            
            self.model   = r.model
            
    
    def set_hyperparameters(self):
        """
        Set the imputation model hyper-parameters.
        
        """  
        
        hyper_dict   = {'mean': [], 'median': [], 'most_frequent': [], 
                        'MICE': ['Number of Multiple Imputations'], 'missForest': ['Number of Trees'],
                        'EMB': ['Number of Multiple Imputations'], 'matrix_completion': ['Max Rank', 'Lambda']}
        
        _hyp__input  = (self.kwargs.__contains__('_hyperparameters')) # Hyperparameters input FLAG
        
        default_dict = {'MICE': {'Number of Multiple Imputations': 5},
                        'missForest': {'Number of Trees': 100},
                        'EMB': {'Number of Multiple Imputations': 5},
                        'matrix_completion': {'Max Rank': 2, 'Lambda': 0}}
        
        missing_hyp     = []
        missing_flg     = False
        self.hyper_dict = hyper_dict 
        
        # cleanup inputs by deleting wrongly provided hyperparameters
        
        if _hyp__input:
            
            hyper_keys = list(self._hyperparameters.keys())
            
            # clean wrong inputs
            
            for u in range(len(hyper_keys)):
                
                if hyper_keys[u] not in hyper_dict[self._mode]:
                    self._hyperparameters.pop(hyper_keys[u], None)
            
            for u in range(len(hyper_dict[self._mode])):
                
                if hyper_dict[self._mode][u] not in hyper_keys:
                    missing_hyp.append(hyper_dict[self._mode][u])
                    missing_flg = True
        else:
            missing_hyp = hyper_dict[self._mode]
            missing_flg = True
            
        # set defaults if no input provided
        
        if self._mode in ['mean','median','most_frequent']:
            self._hyperparameters  = {}
            
        elif missing_flg:
            
            # create an empty hyperparameters dictionary 
            self._hyperparameters = dict.fromkeys(hyper_dict[self._mode])
            
            for v in range(len(missing_hyp)):
                self._hyperparameters[missing_hyp[v]] = default_dict[self._mode][missing_hyp[v]]
            
    def collect_Rpackage_imputation_(self):
        """
        Creates an R wrapper for imputation alorithms (MICE, EMB, missForest, matrix_completion). 
        Searches the R packages required, installs them, and assign a model instance in an R object to
        "model" attribute.
        
        """  
        r_verbose    = 'require("' + self.package_dict[self._mode] + '")'
    
        if (r(r_verbose)==0): # Check whether the grf package is installed, if not, install it.
            
            # import R's utility package
            utils    = rpackages.importr('utils')
            
            # select a CRAN mirror for R packages
            utils.chooseCRANmirror(ind=1)
            utils.install_packages(self.package_dict[self._mode])
            
    def get_r_command(self, r_names_dict):
        """
        Build the syntax for the imputation R command to execute in the R object.
        
        :r_names_dict: names of R variables.
        """  
        if self._mode == 'MICE':
            
            [rpy2.robjects.r.assign(r_names_dict['MICE'][indx], self._hyperparameters[indx]) for indx in self.hyper_dict['MICE']] 
            r_command =  'X <- as.matrix(complete(mice(data = X,  m=' + r_names_dict['MICE']['Number of Multiple Imputations'] + ')))'
        
        elif self._mode == 'missForest':
            
            [rpy2.robjects.r.assign(r_names_dict['missForest'][indx], self._hyperparameters[indx]) for indx in self.hyper_dict['missForest']] 
            r_command =  'X <- missForest(X, ntree=' + r_names_dict['missForest']['Number of Trees'] + ')$ximp'
         
        elif self._mode == 'matrix_completion':
            
            [rpy2.robjects.r.assign(r_names_dict['matrix_completion'][indx], self._hyperparameters[indx]) for indx in self.hyper_dict['matrix_completion']] 
            r_command =  'X <- complete(X,softImpute(X, rank.max = ' + r_names_dict['matrix_completion']['Max Rank'] + ', lambda =' + r_names_dict['matrix_completion']['Lambda'] + '))'
        
        elif self._mode == 'EMB':
            
            [rpy2.robjects.r.assign(r_names_dict['EMB'][indx], self._hyperparameters[indx]) for indx in self.hyper_dict['EMB']] 
            r_command =  'X <- amelia(X, m=' + r_names_dict['EMB']['Number of Multiple Imputations'] + ')'
            
        return r_command        
        
    def fit(self, X):
        """
        Impute missing data using the current instance of the imputation model. 
        
        :X: Input features with missing data.
        """  
        
        X = np.array(X)
        
        # export the data into the r object
        if self._mode in ['mean','median','most_frequent']:
            
            X = self.model.fit_transform(X)
        
        else:
            
            r_names_dict = {'MICE': {'Number of Multiple Imputations': 'mi'}, 'missForest': {'Number of Trees': 'num_trees'},
                            'EMB': {'Number of Multiple Imputations': 'mi'}, 'matrix_completion': {'Max Rank': 'rank', 'Lambda':'lambda'}}
            
            rpy2.robjects.r.assign('X', X)
            r_command    = self.get_r_command(r_names_dict)

            self.init_r_sytem()
            
            r(r_command)
            X            = r.X
        
        return X

    def init_r_sytem(self):

        if not self.is_init_r_system:
            self.collect_Rpackage_imputation_()
            self.is_init_r_system = True


class MICE:
    """ Multiple imputation via chained equations model."""
    
    
    def __init__(self, num_impute=1): 

        self.model_type    = 'imputer'
        self.name          = 'MICE'
        self.MI            = bool(num_impute > 1)
        self.num_impute    = num_impute
        self.model         = baseImputer(_mode=self.name, 
                                         _hyperparameters={'Number of Multiple Imputations': self.num_impute})
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = [{'name': 'MICE.numimpute', 
                               'type': 'discrete', 'domain': tuple(range(1,10)), 
                               'dimensionality': 1}]
        
        return hyp_         



class missForest:
    """ missForest imputation algorithm."""
    
    
    def __init__(self, num_trees=50): 
        
        self.model_type    = 'imputer'
        self.name          = 'missForest'
        self.MI            = False
        self.num_trees     = num_trees
        self.model         = baseImputer(_mode=self.name, 
                                         _hyperparameters={'Number of Trees': self.num_trees})
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = [{'name': 'missForest.ntrees', 
                               'type': 'discrete', 'domain': tuple(range(10,1000)), 
                               'dimensionality': 1}]
        
        return hyp_ 



class EMB:
    """ Bootsrapped EM imputation algorithm."""
    
    
    def __init__(self, num_impute=1): 
        
        self.model_type    = 'imputer'
        self.name          = 'EMB'
        self.MI            = bool(num_impute > 1)
        self.num_impute    = num_impute
        self.model         = baseImputer(_mode=self.name, 
                                         _hyperparameters={'Number of Multiple Imputations': self.num_impute})
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = [{'name': 'EMB.numimpute', 
                               'type': 'discrete', 'domain': tuple(range(1,10)), 
                               'dimensionality': 1}]
        
        return hyp_ 


class matrix_completion:
    """ Matrix completion imputation algorithm."""
    
    
    def __init__(self, Max_rank=2, Lambda=0): 
        
        self.model_type    = 'imputer'
        self.name          = 'matrix_completion'
        self.MI            = False
        self.Lambda        = Lambda
        self.Max_rank      = Max_rank
        self.model         = baseImputer(_mode=self.name, 
                                         _hyperparameters={'Max Rank': self.Max_rank, 'Lambda': self.Lambda})
        
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = [{'name': 'mcompletion.maxrank', 
                               'type': 'discrete', 'domain': tuple(range(2,10)), 
                               'dimensionality': 1}, {'name': 'Lambda', 
                               'type': 'discrete', 'domain': tuple(range(0,5)), 
                               'dimensionality': 1}]
        
        return hyp_ 



class mean:
    """ Mean imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'mean'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_
    

class median:
    """ Median imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'median'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_     
    
    
class most_frequent:
    """ Most frequent imputation algorithm."""
    
    
    def __init__(self): 
        
        self.model_type    = 'imputer'
        self.name          = 'most_frequent'
        self.MI            = False
        self.model         = baseImputer(_mode=self.name)
        
    def fit_transform(self, X):
        
        return self.model.fit(X)
        
    def get_hyperparameter_space(self):
            
        hyp_               = []
        
        return hyp_         

