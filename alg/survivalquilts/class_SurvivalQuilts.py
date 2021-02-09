import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import sys, os

import GPyOpt

from sklearn.model_selection import train_test_split, StratifiedKFold

#user defined
from class_UnderlyingModels import CoxPH, CoxPHRidge, Weibull, LogNormal, LogLogistic, RandomSurvForest
from utils_eval import calc_metrics



class SurvivalQuilts(): 
    
    def __init__(self, K=10, num_bo=50, num_outer=3, num_cv=10, step_ahead=5):
        self.K             = K          #number of time-horizons for temporal quilting
        self.step_ahead    = step_ahead #step_ahead calculation for robust selection
        
        self.num_bo        = num_bo     # BO iteration
        self.num_outer     = num_outer  # maximum number of BO
        
        self.num_cv        = num_cv    # number of cross-validation
        
        self.lmbda         = 0.
        self.rho           = 0.5
        
        self.SEED          = 1234
                
        # self.model_names   = ['CoxPH', 'Weibull', 'LogNormal']
        self.model_names   = ['CoxPH', 'CoxPHRidge', 'Weibull', 'LogNormal', 'LogLogistic', 'RandomSurvForest']
        self.M             = len(self.model_names)
        self.ens_domain    = [{'name': 'w_' + str(m), 'type': 'continuous', 'domain': (0,1),'dimensionality': 1} for m in range(self.M)] 
        
    
    def train(self, X, T, Y):
        t_start = int(T[Y.iloc[:,0] == 1].quantile(0.1))
        t_end   = int(np.max(T[Y.iloc[:,0] == 1]))
        
        self.time_horizons      = [t for t in np.linspace(t_start, t_end, self.K, dtype=int)]        
        self.all_time_horizons  = [t for t in range(int(np.min(T[Y.iloc[:,0] == 1])), int(np.max(T[Y.iloc[:,0] == 1])))]

        
        ### INITIAL TRAINING - UNDERLYING MODELS
        print('initial training of underlying models...')
        
        metric_CINDEX, metric_BRIER = np.zeros([self.num_cv, self.M, self.K]), np.zeros([self.num_cv, self.M, self.K])
        self.CV_pulled_models       = []
        
        for cv_idx in range(self.num_cv):
            print('CV.. {}/{}'.format(cv_idx+1, self.num_cv))
            pulled_models, tmp_CINDEX, tmp_BRIER = self._get_models_pulled_CV(X, T, Y, seed=cv_idx)
            
            metric_CINDEX[cv_idx,:,:] = tmp_CINDEX
            metric_BRIER[cv_idx,:,:]  = tmp_BRIER
            
            self.CV_pulled_models.append(pulled_models)
            
        X_inits = np.zeros([1,self.M])
        X_inits[0, np.argmax(np.mean(metric_CINDEX, axis=0)[:,0])] = 1  #put more weights on the "best" one (at the first step)
        X_inits = self._get_normalized_X_step(X_inits)

        W_prev = np.zeros([self.K, self.M])
        W_prev[:,:] = X_inits

        ### BAYESIAN OPTIMIZATION -- TEMPORAL QUILTING
        for k in range(self.K):
            lmbda_   = self.lmbda
            rho_     = self.rho

            print('TIME K = ' +str(k))
            print(W_prev)

            ### INITIALIZATION FOR TIME-STEP k
            X_inits = np.zeros([1,self.M])
            X_inits[0, np.argmax(np.mean(metric_CINDEX, axis=0)[:,k])]=1  #put more weights on the "best" one (at the first step)
            X_inits = self._get_normalized_X_step(X_inits)

            beta_ = np.median(np.mean(np.mean(metric_BRIER, axis=0)[:,k:(k+self.step_ahead+1)], axis=1))

            W = np.copy(W_prev)
            W[k:,:] = X_inits

            Yo_inits, Yc_inits = [], []

            tmp_o_prev, tmp_c_prev = self._get_Y_step_pulled(W_prev, X, T, Y, K_step=k)
            tmp_o, tmp_c           = self._get_Y_step_pulled(W, X, T, Y, K_step=k)     

            Yo_next = np.asarray(tmp_o[0])
            Yc_next = self._get_AL_constraint(tmp_c[0], beta_, lmbda_, rho_)

            Yo_inits.append(Yo_next)
            Yc_inits.append(Yc_next)

            Yo_inits = np.asarray(Yo_inits).reshape([-1,1])
            Yc_inits = np.asarray(Yc_inits).reshape([-1,1])
            for out_itr in range(self.num_outer):
                X_step_ens   = X_inits
                Y_step_ens   = Yo_inits + Yc_inits

                print(X_inits)
                print(Yo_inits + Yc_inits)

                for itr in range(self.num_bo):
                    gp = GPyOpt.methods.BayesianOptimization(f = None, domain = self.ens_domain, X = X_step_ens, 
                                                                        Y = Y_step_ens, acquisition_type='EI', 
                                                                        model_type='GP', exact_feval = True,
                                                                        cost_withGradients=None)

                    X_next = gp.suggest_next_locations()
                    X_next = self._get_normalized_X_step(X_next)

                    W[k:, :] = X_next

                    if itr < (self.num_bo-1):
                        tmp_o, tmp_c = self._get_Y_step_pulled(W, X, T, Y, K_step=k)
                        Yo_next = np.asarray(tmp_o[0]).reshape([-1,1])
                        Yc_next = self._get_AL_constraint(tmp_c[0], beta_, lmbda_, rho_)
                        Y_next  = Yo_next + Yc_next

                        X_step_ens = np.vstack([X_step_ens, X_next])
                        Y_step_ens = np.vstack([Y_step_ens, Y_next])

                print('=========== BO Finished ===========')

                GP_ens = gp.model.model

                if GP_ens is not None:
                    X_opt    = X_step_ens[np.argmin(Y_step_ens,axis=0)]                
                    W[k:, :] = X_opt

                    print('out_itr: ' + str(out_itr) + ' | BEST X: ' + str(X_opt) )
                    tmp_o, tmp_c = self._get_Y_step_pulled(W, X, T, Y, K_step=k)

                    print(tmp_o[0])

                    if max(0, tmp_c[0] - beta_) < 0.005*beta_: #1% off from the median
                        print('====================================')
                        print('THRESHOLD SATISFIED')
                        print('BEST: ' + str(X_opt))
                        print('Objective val.: ' + str(tmp_o[0]))
                        print('Constraint val.: ' + str(tmp_c[0]))
                        print('====================================')
                        break
                else:
                    raise ValueError('BO failed...')

                lmbda_ = max(0, lmbda_ + 1./rho_ * tmp_c[0])

                if tmp_c[0] <= 0.:
                    rho_ = rho_
                else:
                    rho_ = rho_/2.

                X_inits  = X_opt
                Yo_inits = np.asarray(tmp_o[0]).reshape([-1,1])
                Yc_inits = self._get_AL_constraint(tmp_c[0], beta_, lmbda_, rho_)

                print( 'out_itr: {} | Lambda: {} | Rho: {}'.format(out_itr, lmbda_, rho_) )


            thres_split = abs(tmp_o_prev[0] * 0.005) # 2% improvement -> update
            if -(tmp_o[0] - tmp_o_prev[0]) > thres_split: # since tmp_o is negative C-index
                W_prev = np.copy(W)   #only update if W is not significantly better

                
        ### FINAL MODEL:
        self.quilting_patterns = np.copy(W_prev)
        self.underlying_models = self._get_trained_models(X, T, Y)
        
    
    def predict(self, X, eval_time_horizons=None):
        '''
            The underlying models are trained and quilting patterns are given after training.
                - self.underlying_models
                - self.quilting_patterns
                
            eval_time_horizons is either a list of evaluation times or None
                - None gives all the possible prediction values.
                
            output: risk
        '''
        pred_all = self._get_ensemble_prediction(self.underlying_models, self.quilting_patterns, X, self.all_time_horizons)
        
        if eval_time_horizons:
            pred     = np.zeros([np.shape(pred_all)[0], len(eval_time_horizons)])
            for t, eval_time in enumerate(eval_time_horizons):
                pred[:, t] = pred_all[:, np.where(np.asarray(self.all_time_horizons) <= eval_time)[0][-1]]
        else:
            pred     = np.copy(pred_all)
            
        return pred
    
    
    def _make_ModelList(self):
        models = []
        for tmp_name in self.model_names:
            if tmp_name == 'CoxPH':
                models += [CoxPH()]
            elif tmp_name == 'CoxPHRidge':
                models += [CoxPHRidge()]
            elif tmp_name == 'Weibull':
                models += [Weibull()]
            elif tmp_name == 'LogNormal':
                models += [LogNormal()]
            elif tmp_name == 'LogLogistic':
                models += [LogLogistic()]
            elif tmp_name == 'RandomSurvForest':
                models += [RandomSurvForest()]
        return models

    
    def _get_ensemble_prediction(self, models, W_, X_, all_time_horizons_):
        for m in range(self.M):
            tmp_pred_ = models[m].predict(X_, all_time_horizons_)

            if m == 0:
                pred_ = np.zeros(np.shape(tmp_pred_))
            else:
                for tt in range(self.K):
                    if tt == 0:
                        tmp_time_idx1 = np.asarray(all_time_horizons_) <= self.time_horizons[tt]
                        tmp_time_idx2 = np.asarray(all_time_horizons_) > self.time_horizons[tt]

                        increment = tmp_pred_[:, tmp_time_idx1] - np.matmul(tmp_pred_[:, tmp_time_idx1][:,[0]], np.ones([1,np.sum(tmp_time_idx1)]))

                        pred_[:, tmp_time_idx1] =  pred_[:, tmp_time_idx1] + W_[tt,m] * increment
                        pred_[:, tmp_time_idx2] =  pred_[:, tmp_time_idx2] + W_[tt,m] * np.matmul(increment[:,[-1]], np.ones([1,np.sum(tmp_time_idx2)]))

                    elif tt == (self.K - 1): #the last index  
                        tmp_time_idx1 = np.asarray(all_time_horizons_) > self.time_horizons[tt-1]

                        increment = tmp_pred_[:, tmp_time_idx1] - np.matmul(tmp_pred_[:, tmp_time_idx1][:,[0]], np.ones([1,np.sum(tmp_time_idx1)]))

                        pred_[:, tmp_time_idx1] =  pred_[:, tmp_time_idx1] + W_[tt,m] * increment

                    else:
                        tmp_time_idx1 = (np.asarray(all_time_horizons_) > self.time_horizons[tt-1]) & (np.asarray(all_time_horizons_) <= self.time_horizons[tt])
                        tmp_time_idx2 = np.asarray(all_time_horizons_) > self.time_horizons[tt]

                        increment = tmp_pred_[:, tmp_time_idx1] - np.matmul(tmp_pred_[:, tmp_time_idx1][:,[0]], np.ones([1,np.sum(tmp_time_idx1)]))

                        pred_[:, tmp_time_idx1] =  pred_[:, tmp_time_idx1] + W_[tt,m] * increment
                        pred_[:, tmp_time_idx2] =  pred_[:, tmp_time_idx2] + W_[tt,m] * np.matmul(increment[:,[-1]], np.ones([1,np.sum(tmp_time_idx2)]))

        return pred_
    
    
    def _get_Y_step_pulled(self, W_, X_, T_, Y_, K_step):

        metric_CINDEX_ = np.zeros([self.num_cv])
        metric_BRIER_  = np.zeros([self.num_cv])

        for cv_idx in range(self.num_cv):
            _,X_va, T_tr,T_va, Y_tr,Y_va = train_test_split(X_, T_, Y_, test_size=0.20, random_state=cv_idx+self.SEED)

            pred = self._get_ensemble_prediction(self.CV_pulled_models[cv_idx], W_, X_va, self.time_horizons)

            new_K_step = min(K_step + 1 + self.step_ahead, self.K)
            for k in range(K_step, new_K_step):
                eval_time    = self.time_horizons[k]            
                tmp_C, tmp_B = calc_metrics(T_tr, Y_tr, T_va, Y_va, pred[:, k], eval_time)

                metric_CINDEX_[cv_idx] += 1./len(self.time_horizons) * tmp_C
                metric_BRIER_[cv_idx]  += 1./len(self.time_horizons) * tmp_B

                metric_CINDEX_[cv_idx] += 1./(new_K_step - K_step) * tmp_C
                metric_BRIER_[cv_idx]  += 1./(new_K_step - K_step) * tmp_B

        Output_CINDEX = (- metric_CINDEX_.mean(),1.96*np.std(metric_CINDEX_)/np.sqrt(self.num_cv))
        Output_BRIER  = (metric_BRIER_.mean(),1.96*np.std(metric_BRIER_)/np.sqrt(self.num_cv))

        return Output_CINDEX, Output_BRIER

    
    def _get_models_pulled_CV(self, X, T, Y, seed):
        X_tr, X_va, T_tr, T_va, Y_tr, Y_va = train_test_split(X, T, Y, test_size=0.20, random_state=seed+self.SEED)
        
        pulled_models = self._get_trained_models(X_tr, T_tr, Y_tr)

        metric_CINDEX, metric_BRIER = np.zeros([self.M, self.K]), np.zeros([self.M, self.K])
        
        for m, model in enumerate(pulled_models):
            pred = model.predict(X_va, self.time_horizons)

            for t, eval_time in enumerate(self.time_horizons):
                tmp_C, tmp_B        = calc_metrics(T_tr, Y_tr, T_va, Y_va, pred[:, t], eval_time)                
                metric_CINDEX[m, t] = tmp_C
                metric_BRIER[m, t]  = tmp_B

        return pulled_models, metric_CINDEX, metric_BRIER
    

    def _get_trained_models(self, X, T, Y):
        models = self._make_ModelList()
        for m in range(self.M):
            models[m].fit(X, T,  Y)
        return models
    
    def _get_AL_constraint(self, g, beta_, lmbda_, rho_):
        return np.asarray(lmbda_ * (g-beta_) + 0.5 / rho_ * max(0,(g-beta_))**2).reshape([-1,1])

    def _get_normalized_X_step(self, X_step_):
        for k in range(np.shape(X_step_)[0]):
            X_step_[k, :] = X_step_[k, :]/(np.sum(X_step_[k, :])+1e-8)
        return X_step_
