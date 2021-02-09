'''
    << New Release >>    
    For stability issues, R packages are replaced by recent python packages (if available) or removed (otherwise).
    
'''

### SCIKIT-SURVIVAL
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter


import numpy as np
import pandas as pd
import sys, os, warnings, time

#=================================================================

class sksurvSurvival( object ):
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self):     
        exec('1+1') # dummy instruction
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        y = [(Y.iloc[i,0], T.iloc[i,0]) for i in range(len(Y))]
        y = np.array(y, dtype=[('status', 'bool'),('time','<f8')])
        # print(self.name)
        self.model.fit(X,y)

    def predict(self,X, time_horizons): 
        if self.name in ['CoxPH', 'CoxPHRidge']:
            surv = self.model.predict_survival_function(X)  #returns StepFunction object
            preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])
            
            for t, eval_time in enumerate(time_horizons):
                if eval_time > np.max(surv[0].x):  #all have the same maximum surv.x
                    eval_time = np.max(surv[0].x)
                preds_[:, t] = np.asarray([(1. - surv[i](eval_time)) for i in range(len(surv))])  #return cif at self.median_tte
                
        elif self.name in ['RandomSurvForest']:
            surv       = self.model.predict_survival_function(X) #returns numpy array
            surv_times = self.model.event_times_
            preds_     = np.zeros([np.shape(surv)[0], len(time_horizons)])

            for t, eval_time in enumerate(time_horizons):
                tmp_time = np.where(eval_time <= surv_times)[0]

                if len(tmp_time) == 0:
                    preds_[:, t] = 1. - surv[:, 0]
                else:
                    preds_[:, t] = 1. - surv[:, tmp_time[0]]
        
        else:
            preds_ = self.model.predict(X)

        return float(self.direction)*preds_



#-----------------------------------------------------------------
class CoxPH(sksurvSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self):

        super(CoxPH, self).__init__()
        # super().__init__()

        self.name          = 'CoxPH'

        self.model         = CoxPHSurvivalAnalysis(alpha=0.01)  #otherwise error occured
        self.direction     = 1
        self.prob_FLAG     = True

        self.explained     = "*Cox proportional model"
        self.image_name    = "Cox.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. this is not used in CoxPH (c.f. CoxPHRidge)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None used in CoxPH
        
    def get_hyperparameter_space(self):
        return []

#-----------------------------------------------------------------
class CoxPHRidge(sksurvSurvival): 
    """ Cox proportional hazard model with ridge regression penalty. """
    
    def __init__(self,alpha=10.0):

        super(CoxPHRidge, self).__init__()
        # super().__init__()

        self.alpha         = alpha
        self.name          = 'CoxPHRidge'

        self.model         = CoxPHSurvivalAnalysis(alpha=self.alpha) #ridge regression penalty
        self.direction     = 1
        self.prob_FLAG     = True
        
        self.explained     = "*Cox proportional model with ridge regression"
        self.image_name    = "CoxPHRidge.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def get_hyperparameter_space(self):
        hyp_   = [{'name': 'CoxPHRidge.alpha', 'type': 'continuous', 'domain': (0.001,10),'dimensionality': 1}]

        return hyp_
    
    
class RandomSurvForest(sksurvSurvival): 
    """ Cox proportional hazard model with ridge regression penalty. """
    
    def __init__(self, n_estimators=100):

        super(RandomSurvForest, self).__init__()
        # super().__init__()
        
        self.n_estimators  = n_estimators
        self.name          = 'RandomSurvForest'

        self.model         = RandomSurvivalForest(n_estimators=self.n_estimators)
        self.direction     = 1
        self.prob_FLAG     = True
        
        self.explained     = "*Random Survival Forest"
        self.image_name    = "RandomSurvForest.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def get_hyperparameter_space(self):
        hyp_   = [{'name': 'RandomSurvForest.n_estimators', 'type': 'continuous', 'domain': (50,500), 'dimensionality': 1}]

        return hyp_
#=================================================================



class lifelinesSurvival( object ):
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self):     
        exec('1+1') # dummy instruction
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        df = pd.concat([X, T, Y], axis=1)
        df.columns = [x for x in X.columns] + ['time', 'label']
        
        self.model.fit(df, duration_col='time', event_col='label')

    def predict(self, X, time_horizons): 
        if self.name in ['Weibull', 'LogNormal', 'LogLogistic']:
            surv       = self.model.predict_survival_function(X)
            surv_times = np.asarray(surv.index).astype(int)
            surv       = np.asarray(surv.T)
            preds_     = np.zeros([np.shape(surv)[0], len(time_horizons)])

            for t, eval_time in enumerate(time_horizons):
                tmp_time = np.where(eval_time <= surv_times)[0]
                if len(tmp_time) == 0:
                    preds_[:, t] = 1. - surv[:, 0]
                else:
                    preds_[:, t] = 1. - surv[:, tmp_time[0]]
                    
        else:
            raise ValueError('Wrong Model Choice! (Weibull, LogNormal, LogLogistic)')
        
        return float(self.direction)*preds_



#-----------------------------------------------------------------
class Weibull(lifelinesSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self):

        super(Weibull, self).__init__()
        # super().__init__()

        self.name          = 'Weibull'

        self.model         = WeibullAFTFitter()  #otherwise error occured
        self.direction     = 1
        self.prob_FLAG     = True

        self.explained     = "*Parameteric model - Weibull"
        self.image_name    = "Weibull.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. this is not used in CoxPH (c.f. CoxPHRidge)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None used in CoxPH
        
    def get_hyperparameter_space(self):
        return []
    

#-----------------------------------------------------------------
class LogNormal(lifelinesSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self):

        super(LogNormal, self).__init__()
        # super().__init__()

        self.name          = 'LogNormal'

        self.model         = LogNormalAFTFitter()  #otherwise error occured
        self.direction     = 1
        self.prob_FLAG     = True

        self.explained     = "*Parameteric model - LogNormal"
        self.image_name    = "LogNormal.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. this is not used in CoxPH (c.f. CoxPHRidge)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None used in CoxPH
        
    def get_hyperparameter_space(self):
        return []

#-----------------------------------------------------------------
class LogLogistic(lifelinesSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self):

        super(LogLogistic, self).__init__()
        # super().__init__()

        self.name          = 'LogLogistic'

        self.model         = LogLogisticAFTFitter()  #otherwise error occured
        self.direction     = 1
        self.prob_FLAG     = True

        self.explained     = "*Parameteric model - LogLogistic"
        self.image_name    = "LogLogistic.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ((alpha)) : ridge regression penalty. this is not used in CoxPH (c.f. CoxPHRidge)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None used in CoxPH
        
    def get_hyperparameter_space(self):
        return []
    
    
