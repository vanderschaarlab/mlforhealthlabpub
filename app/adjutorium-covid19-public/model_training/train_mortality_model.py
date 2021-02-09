
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
import scipy.stats
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from data.data_processing import *


def monotonize(curve):
    
    new_curve = []
    
    for i in range(len(curve)):
        if i > 0:
            if i == 1:
                new_curve.append(np.maximum(curve[i], curve[i-1]))
            else:    
                new_curve.append(np.maximum(curve[i], np.max(curve[:i])))
        else:        
            new_curve.append(curve[i])
            
    return new_curve  


# def prepare_CHESS_data(data_collection_date, feature_groups):
#
#     CHESS_data, feature_names = get_data(curr_date=data_collection_date)
#
#     for u in range(len(feature_groups)):
#
#         if u==0:
#
#             feature_set = feature_names[feature_groups[u]]
#
#         else:
#
#             feature_set = feature_set + feature_names[feature_groups[u]]
#
#
#     imputer                   = IterativeImputer(max_iter=10, random_state=0)
#
#     imputer.fit(CHESS_data[feature_set])
#
#     X                         = np.float32(imputer.transform(CHESS_data[feature_set]))
#
#     return CHESS_data, X, feature_set, feature_names, imputer


def prepare_CHESS_data(data_collection_date, feature_groups, imputer=None):
  CHESS_data, feature_names, aux_data = get_data(curr_date=data_collection_date)

  for u in range(len(feature_groups)):

    if u == 0:

      feature_set = feature_names[feature_groups[u]]

    else:

      feature_set = feature_set + feature_names[feature_groups[u]]

  if imputer is None:
    imputer = IterativeImputer(max_iter=10, random_state=0)

    imputer.fit(CHESS_data[feature_set])

  X = np.float32(imputer.transform(CHESS_data[feature_set]))

  return CHESS_data, X, feature_set, feature_names, imputer, aux_data


class adjutorium_mortality_model:
    
    def __init__(self, number_of_days=7, n_bootstraps=100, outcome="Dead", surrogate_outcome="Discharged", 
                 outcome_time="Follow up time", data_collection_date="2020-04-04", 
                 feature_groups=["Personal info", "Comorbidity info"], train_frac=.8):
        
        self.horizons       = list(range(1, number_of_days + 1))
        self.n_samples      = n_bootstraps
        self.outcome        = outcome
        self.surrogate_out  = surrogate_outcome
        self.outcome_time   = outcome_time
        self.train_frac     = train_frac
        self.sampled_models = []

        self.data, self.X, self.feature_set, self.feature_names, self.imputer, _ = prepare_CHESS_data(
            data_collection_date=data_collection_date,
            feature_groups=feature_groups)
        
    def train(self):
        
        for _ in range(self.n_samples): 

            models_           = []
            
            print("Model number: %s" % str(_+1))
    
            for horizon in self.horizons:
            
                print("Prediction horizon: %s days" % str(horizon))
                
                if self.surrogate_out is not None:
                
                    included      = list(((self.data[self.outcome]==1) | (self.data[self.surrogate_out]==1) | (self.data[self.outcome_time] >= horizon)))   
        
                else:
                
                    included      = list(((self.data[self.outcome]==1) | (self.data[self.outcome_time] >= horizon)))   
        
                train_size    = int(np.floor(self.train_frac * np.sum(included)))
                train_indexes = np.random.choice(np.sum(included), train_size, replace=False)
                test_indexes  = list(set(list(range(np.sum(included)))) - set(list(train_indexes)))

                X_            = self.X[included, :]
                Y_            = np.array(((self.data.loc[included, self.outcome]==1) & (self.data.loc[included, self.outcome_time]<=horizon)) * 1)
                
                X_train       = X_[train_indexes, :]
                Y_train       = Y_[train_indexes]
                        
                X_test        = X_[test_indexes, :]
                Y_test        = Y_[test_indexes]
                
                # replace this with AutoPrognosis
                base_model    = GradientBoostingClassifier(n_estimators=150) 
                
                base_model.fit(X_train, Y_train)
        
                base_model_   = CalibratedClassifierCV(base_model, cv='prefit')
        
                base_model_.fit(X_test, Y_test)
    
                models_.append(base_model_)
    
            self.sampled_models.append(models_)
    
    
    def predict(self, X):
        
        preds     = []
        
        if len(X.shape)==1:
            
            X = X.reshape((1, -1))
               
        
        for v in range(X.shape[0]):
            
            surv_curv = [monotonize([self.sampled_models[u][k].predict_proba(X[v, :].reshape((1,-1)))[:, 1][0] for k in range(len(self.horizons))]) for u in range(len(self.sampled_models))]
        
            preds.append(np.mean(surv_curv, axis=0))
        
        return preds
        
def predict_batch(model, X):

  if len(X.shape) == 1:
    X = X.reshape((1, -1))

  prediction_list = []

  for u in range(len(model.sampled_models)):

    y_hat_list = []

    for k in range(len(model.horizons)):
      # N,
      y_hat = model.sampled_models[u][k].predict_proba(X)[:, 1]
      y_hat_list.append(y_hat)

    # N, horizon
    y_hat_mat = np.stack(y_hat_list, axis=1)
    curve_list = []
    for i in range(y_hat_mat.shape[0]):
      single_curve = y_hat_mat[i]
      single_curve = np.array(monotonize(single_curve))
      curve_list.append(single_curve)

    # N, horizon
    model_curve = np.stack(curve_list, axis=0)
    prediction_list.append(model_curve)

  prediction_mat = np.stack(prediction_list, axis=-1)
  predictions = np.mean(prediction_mat, axis=-1)
  return predictions

if __name__ == "__main__":
    
    feature_groups  = ["Personal info", "Ethnicity info", "Comorbidity info", "Lab test info", 
                       "Hospitalization info", "Complications info", "Interventions info"]
    
    death_model     = adjutorium_mortality_model(number_of_days=14, n_bootstraps=20, data_collection_date="14/4/2020")
    discharge_model = adjutorium_mortality_model(number_of_days=14, n_bootstraps=20, data_collection_date="14/4/2020", outcome="Discharged", surrogate_outcome="Dead")

    ICU_model       = adjutorium_mortality_model(number_of_days=14, n_bootstraps=20, outcome="ICU Admission", 
                                                 surrogate_outcome=None, outcome_time="Time to ICU", data_collection_date="14/4/2020", 
                                                 feature_groups=["Personal info", "Comorbidity info"], train_frac=.8)

    
    death_model.train()
    discharge_model.train()
    ICU_model.train()
    
    with open('adjutorium_mortality', 'wb') as handle:
        
        pickle.dump(death_model, handle)

    with open('adjutorium_discharge', 'wb') as handle:
        
        pickle.dump(discharge_model, handle)

    with open('adjutorium_icu', 'wb') as handle:
        
        pickle.dump(ICU_model, handle)


