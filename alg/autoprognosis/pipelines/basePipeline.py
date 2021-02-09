
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import
import itertools
import warnings
import time
import logging
import numpy as np
warnings.filterwarnings("ignore")

logger = logging.getLogger()


class basePipeline:
    
    """
    Base class for pipeline objects.

    """

    def __init__(self, 
                 analysis_mode=None,
                 analysis_type=None, 
                 model_list=None,
                 hyperparameter_list=None,
                 **kwargs):

        self.model_list = model_list
        self.explained = '-todo: explain basepipeline-'
        self.image_name = None
        self.classes = None
        if self.model_list is not None:

            self.num_stages      = len(self.model_list)
            self.pipeline_stages = [self.model_list[k].model_type for k in range(len(self.model_list))]

            self.name = '[ '
            self.explained = '[ '
            for u in range(self.num_stages):

                if hasattr(self.model_list[u], 'explained'):
                    self.explained += self.model_list[u].explained
                self.name += self.model_list[u].name

                if u != self.num_stages-1:

                    self.name += ', '
                    self.explained += ', '

            self.name = self.name + ' ]'
            self.explained += ' ]'
        
        self.analysis_mode       = analysis_mode
        self.analysis_type       = analysis_type
        
        # check this is a valid pipeline
        # self.set_hyperparameters
        # order pipeline stages for fit and predict
    
    def CHECK_VALIDITY(self, pipeline):
        
        self.VALIDITY_FLAG = True
        mandatory_stages   = ['classifier']
        
        stage_unique       = len(np.unique(np.array(self.pipeline_stages))) < len(self.pipeline_stages)
        mandatory_stage    = True
        
        if not (stage_unique and mandatory_stage):
            
            self.VALIDITY_FLAG = False
    
        return self.VALIDITY_FLAG
    

    def fit(self, X, Y, **kwargs):
        
        """
        Fit the selected algorithm to the training data.
        Parameters
        ----------
        X : array-like training data with shape = (n_samples, n_features)
        Y : array-like labels/targets/resposes
        
        **kwargs: extra inputs may be needed for analysis types other than classification and regression.
        
        Returns
        -------
        All the stages in the pipeline are fit with (X, Y) if the pipeline stage has attribute "fit" 
        and (X) only if the pipeline stage has attribute "fit_transform"
            
        """
        fit_start = time.time()
        
        X_temp = X

        logger.info('+basepipeline::fit y:{},{} {} nstages:{} {} prop:{}'.format(
            len(Y),
            list(set(np.ravel(Y))),
            self.name,
            self.num_stages,
            self.pipeline_stages,
            self.get_properties()))

        for u in range(self.num_stages):

            if hasattr(self.model_list[u], 'fit_transform'): # This should be just a transform

                X_temp = np.array(self.model_list[u].fit_transform(X_temp)).copy()

            else:

                self.model_list[u].fit(X_temp, Y)

            if self.pipeline_stages[u] == 'classifier':

                classes = list(self.model_list[u].model.classes_)
                logger.debug('clf classes:{} ordered:{}'.format(classes, sorted(classes)))
                assert sorted(classes) == classes # this assumption must hold for multiclass when calculating the score
            
        logger.info('-basepipeline::fit {:0.0f}s'.format(time.time() - fit_start))



    def predict(self, X):
        
        """
        
        Predict the target output using the selected model.
        Parameters
        ----------
        X     : array-like testing input with shape = (n_samples, n_features)

        Returns
        -------
        Y_pred: Predicted target with shape=(n_samples,) 
            
        """
        X_temp = X
        
        for u in range(self.num_stages):
            
            if hasattr(self.model_list[u], 'fit_transform'):

                X_temp = self.model_list[u].fit_transform(X_temp)
            
            else:
 
                X_temp = self.model_list[u].predict(X_temp)
        
        Y_pred = X_temp 

        return Y_pred


    def get_hyperparameter_space(self):
        
        """
        
        Return the hyperparameter space for the pipeline object.
        
        Returns
        -------
        hyp_ : The standardized hyperparameter dictionary following the same format as the base models.
               This dictionary object will be passed to the Bayesian optimization function.
        
        """
        
        hyp_= []
        
        if self.model_list is not None:
            
            for model_ in self.model_list:
                
                if hasattr(model_, 'get_hyperparameter_space'):
                    
                    hyp_.append(model_.get_hyperparameter_space())
            
            
            hyp_ = list(itertools.chain.from_iterable(hyp_))
            
        return hyp_

    def get_is_pred_proba(self):
        print(self.name)
        for u in range(self.num_stages):
            if self.pipeline_stages[u] == 'classifier':
                return self.model_list[u].is_pred_proba
        
    def get_properties(self):
        name_lst = list()
        for u in range(self.num_stages):
            if hasattr(self.model_list[u], 'get_properties'): 
                name_lst.append(self.model_list[u].get_properties())
            else:
                name_lst.append({'name': self.model_list[u].name})
        return name_lst
