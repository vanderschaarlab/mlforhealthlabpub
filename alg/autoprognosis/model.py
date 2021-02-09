import numpy as np
from sklearn.model_selection import StratifiedKFold
from pivottablejs import pivot_ui
import warnings
import GPyOpt
import GPy
from IPython.display import Markdown, display, Image
import logging
import copy
import time
from tqdm import tnrange
from models.imputers import missForest, matrix_completion, mean, median
from models.imputers import most_frequent
from models.classifiers import RandomForest, RandomForestClassifier
from models.classifiers import GradientBoosting, GradientBoostingClassifier
from models.classifiers import XGboost, Adaboost, Bagging, BernNaiveBayes
from models.classifiers import BernoulliNB, GaussNaiveBayes
from models.classifiers import MultinomialNaiveBayes, MultinomialNB
from models.classifiers import LogisticReg
from models.classifiers import Percept, DecisionTrees, DecisionTreeClassifier
from models.classifiers import QDA_, LDA_, KNN, LinearSVM, NeuralNet
from models.classifiers import classifier_set_xtr_arg, classifier_get_xtr_arg
from models.preprocessors import Scaler, MinMaxScaler, UniformTransform
from models.preprocessors import GaussianTransform, FeatureNormalizer
from models.preprocessors import GaussProjection, PrincipalComponentAnalysis
from pipelines.basePipeline import basePipeline
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import initpath_ap
initpath_ap.init_sys_path()
import utilmlab


def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


class evaluate:
    ''' 
    wrapper class: returns the score as set by the user.
    '''
    def __init__(self):

        self.m_metric = None
        self.m_metric_allowed = ['aucprc', 'aucroc']

    def set_metric(self, metric):
        
        assert metric in self.m_metric_allowed
        self.m_metric = metric

    def get_metric(self):
        assert self.m_metric is not None
        return self.m_metric
    
    def score(self, y_test, y_pred, y_pred_proba):

        assert y_test is not None
        assert y_pred_proba is not None
        if self.m_metric == 'aucprc':
            score_val = self.average_precision_score(y_test, y_pred_proba)
        elif self.m_metric == 'aucroc':
            score_val = self.roc_auc_score(y_test, y_pred_proba)
        else:
            assert 0
        logger.debug('evaluate:{:0.5f}'.format(score_val))
        return score_val
    
    def score_proba(self, y_test, y_pred_proba):

        assert y_test is not None
        assert y_pred_proba is not None

        return self.score(y_test, None, y_pred_proba)

    def roc_auc_score(self, y_test, y_pred_proba):

        return utilmlab.evaluate_auc(y_test, y_pred_proba)[0]
    
    def average_precision_score(self, y_test, y_pred_proba):

        return utilmlab.evaluate_auc(y_test, y_pred_proba)[1]

    def accuracy_score(self, y_true, y_pred, y_pred_proba):
        y_hat = np.argmax(y_pred_proba, axis=1)
        return sum(y_true == y_hat) / float(len(y_true))


def set_xtr_arg(xtr):
    global model_xtr_arg
    model_xtr_arg = xtr
    classifier_set_xtr_arg(xtr)


def show_xtr_arg():
    print(model_xtr_arg)
    print(classifier_get_xtr_arg())
    

def get_xtr_arg():
    return model_xtr_arg

    
def printmd(string):
    global gis_ipython
    if gis_ipython:
        display(Markdown(string))
    logger.info('{}'.format(string))


class AutoPrognosis_Classifier:
    
    def __init__(
            self,
            CV=5,
            num_iter=100,
            kernel_freq=10,
            ensemble=True,
            ensemble_size=10,
            Gibbs_iter=100,
            burn_in=50,
            num_components=3,
            is_nan=True,
            metric='aucroc',
            acquisition_type='LCB',
            **kwargs):

        eva.set_metric(metric)
        self.is_pred_proba = True
        self.acquisition_type = acquisition_type
        self.is_nan = is_nan
        self.num_iter       = num_iter
        self.x_opt          = 0
        self.CV             = CV
        
        self.Gibbs_iter     = Gibbs_iter
        self.burn_in        = burn_in
        self.num_components = num_components
        self.kernel_freq    = kernel_freq 
        self.ensemble_size  = ensemble_size
        self.ensemble       = ensemble

        self.model_         = []
        self.models_        = []
        self.scores_        = []
        
        self.model_names    = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Adaboost', 'Bagging', 'Bernoulli Naive Bayes',
                               'Gauss Naive Bayes', 'Multinomial Naive Bayes', 'Logistic Regression','Perceptron',
                               'Decision Trees','QDA','LDA','KNN','Linear SVM', 'Neural Network']
        
        
        self.initialize_hyperparameter_map()
        self.name = 'autoprognosis_clf'
        self.exe_start_time = time.time()
        
    def initialize_hyperparameter_map(self):
        self.modind         = [0,1,2,3,4,5,7,8,9,10,13,14,15] # indexes of models with hyperparam
        self.noHyp          = [6,11,12]                       # indexes of models without hyperparam
        self.hypMAP         = [[1,2],[3,4,5],[6,7,8],[9,10],[11,12,13,14],[15],[],[16],
                               [17,18,19],[20,21],[22],[],[],[23,24,25,26],[27],[28,29,30,31]]
        
    def get_model(self,dom_,comp_no,comp_map,x_next):
        
        imputers_ = [
            [],
            [missForest()],
            [mean()],
            [median()],
            [most_frequent()]
        ]
        
        preprocessors_= [
            [],
            [MinMaxScaler()],
            [UniformTransform()],
            [PrincipalComponentAnalysis()]
        ]

        if get_xtr_arg() < 1:

            imputers_.append([matrix_completion()])
            
            # WA: bagging with base estimator cannot handle negative values
            preprocessors_.append([Scaler()])
            preprocessors_.append([GaussProjection()])
            preprocessors_.append([GaussianTransform()])
            preprocessors_.append([FeatureNormalizer()])

            
        select_imp = np.random.randint(
            1 if self.is_nan else 0,
            len(imputers_)-1)
        select_pre = np.random.randint(0, len(preprocessors_)-1)        
        
        # Create hyper-parameter dictionary
        #----------------------------------
        base_     = [DecisionTreeClassifier(), RandomForestClassifier(), BernoulliNB(), MultinomialNB(), 
                     GradientBoostingClassifier()]
        LR_solvs  = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        ppens     = ['l2','l1','elasticnet']
        dtcr      = ['gini','entropy']
        knnalgs   = ['auto','ball_tree','kd_tree','brute']
        knnwgs    = ['uniform','distance']
        svmpen    = ['l2','l1']
        nnsolvers = ['lbfgs','sgd','adam']
        nnactiv   = ['identity','logistic','tanh','relu']
        crit_     = ['gini','entropy']
        
        # Unpack model names
        #----------------------------------            
        domain_list = [dom_[k]['name'] for k in range(len(dom_))]
        pivot_ = domain_list.index('classifier')
        mdl_index = comp_map[comp_no][int(x_next[pivot_])]

        # Create a list of model instantiations
        #--------------------------------------
        if mdl_index==0:
    
            model = RandomForest(n_estimators=int(x_next[domain_list.index('RandomForest.ntrees')]),
                                 criterion=crit_[int(x_next[domain_list.index('RandomForest.criterion')])])
        
        elif mdl_index==1:
            
            model = GradientBoosting(n_estimators=int(x_next[domain_list.index('GradientBoosting.n_estimators')]),
                                     learning_rate=x_next[domain_list.index('GradientBoosting.learning_rate')],
                                     max_depth=int(x_next[domain_list.index('GradientBoosting.max_depth')]))
        
        elif mdl_index==2:
            
            model = XGboost(n_estimators=int(x_next[domain_list.index('XGboost.n_estimators')]),       
                            learning_rate=x_next[domain_list.index('XGboost.learning_rate')],
                            max_depth=int(x_next[domain_list.index('XGboost.max_depth')]))
            
        elif mdl_index==3:
            
            model = Adaboost(n_estimators=int(x_next[domain_list.index('Adaboost.n_estimators')]),
                             learning_rate=x_next[domain_list.index('Adaboost.learning_rate')])
            
        elif mdl_index==4:
            
            model = Bagging(n_estimators=int(x_next[domain_list.index('Bagging.n_estimators')]),
                            max_samples=x_next[domain_list.index('Bagging.max_samples')],
                            max_features=1.0,#x_next[domain_list.index('Bagging.max_features')],
                            base_estimator=base_[int(x_next[domain_list.index('Bagging.base_estimator')])])
            
        elif mdl_index==5:
            
            model = BernNaiveBayes(alpha=x_next[domain_list.index('BernoullinNaiveBayes.alpha')]) 
            
        elif mdl_index==6:
            
            model = GaussNaiveBayes()
            
        elif mdl_index==7:
            
            model = MultinomialNaiveBayes(alpha=x_next[domain_list.index('MultinomialNaiveBayes.alpha')])
            
        elif mdl_index==8:
            
            model = LogisticReg(C=x_next[domain_list.index('LogisticRegression.C')],
                                solver=LR_solvs[int(x_next[domain_list.index('LogisticRegression.solver')])],
                                max_iter=int(x_next[domain_list.index('LogisticRegression.max_iter')]))
            
        elif mdl_index==9:
            
            model = Percept(alpha=x_next[domain_list.index('Perceptron.alpha')],
                            penalty=ppens[int(x_next[domain_list.index('Perceptron.penalty')])])     
            
        elif mdl_index==10:
            
            model = DecisionTrees(criterion=dtcr[int(x_next[domain_list.index('DecisionTrees.criterion')])])
            
        elif mdl_index==11:
            
            model = QDA_() 

        elif mdl_index==12:
            
            model = LDA_()

        elif mdl_index==13:
            
            model = KNN(p=int(x_next[domain_list.index('KNN.p')]),
                        algorithm=knnalgs[int(x_next[domain_list.index('KNN.algorithm')])],
                        weights=knnwgs[int(x_next[domain_list.index('KNN.weights')])],
                        n_neighbors=int(x_next[domain_list.index('KNN.n_neighbors')]))
            
        elif mdl_index==14:
            
            model = LinearSVM(penalty=svmpen[int(x_next[domain_list.index('LinearSVM.penalty')])])

        elif mdl_index==15:
            
            model = NeuralNet(num_layers=int(x_next[domain_list.index('NeuralNet.num_layers')]),
                              num_units=int(x_next[domain_list.index('NeuralNet.num_units')]),
                              solver=nnsolvers[int(x_next[domain_list.index('NeuralNet.solver')])],
                              activation=nnactiv[int(x_next[domain_list.index('NeuralNet.activation')])])
        
        model_list_base = imputers_[select_imp] + preprocessors_[select_pre] + [model]

        if nmax_model:
            model_list = model_list_base[-nmax_model:]
        else:
            # auto
            model_list = list()
            if self.is_nan:
                model_list.append(model_list_base[0]) # add imputer
            model_list.append(model_list_base[-1]) # add clf
            
        model = basePipeline(model_list=model_list)
        return model
    
    
    def get_opt_domain(self):

        assert 0
        
        # Define domains for classifier sets
        base_models_ = [RandomForest(),GradientBoosting(),XGboost(),Adaboost(),Bagging(),BernNaiveBayes(),
                        GaussNaiveBayes(),MultinomialNaiveBayes(),LogisticReg(),Percept(),DecisionTrees(),
                        QDA_(),LDA_(),KNN(),LinearSVM(),NeuralNet()]
        
        domain_      = [{'name': 'classifier', 'type': 'categorical', 
                         'domain': (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),'dimensionality': 1}]
        
        for bmodel_ in base_models_:
            domain_  = domain_ + bmodel_.get_hyperparameter_space()

        dim_         = len(domain_)
    
        return domain_, dim_
    
    
    def evaluate_CV_objective(self, X_in, Y_in, modraw_):
        
        mod_back = modraw_

        #mod_back.fit(X_in, Y_in)

        rval_eva = evaluate_clf(X_in.copy(), Y_in.copy(), copy.deepcopy(modraw_), n_folds = self.CV)
        logger.info('CV_objective:{}'.format(rval_eva))
        f = -1*rval_eva[0][0]
        return f, mod_back, rval_eva[1]
  
    #----------------------
    
    def load_initial_model(self):
        # [RF 0 (1,2),GBM 1 (3,4,5),XGB 2 (6,7,8),Ada 3 (9,10),Bag 4 (11,12,13,14), BNB 5 (15), 
        # GNB 6 [], MNB 7 (16), LR 8 (17,18,19), Perc 9 (20,21), DT 10 (22), QDA_ 11 (),
        # LDA_ 12 (), KNN 13 (23,24,25,26), LSVM 14 (27), NN 15 (28,29,30,31)]
        
        init_assigns = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                        [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                        [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
                        [0,1,0],[0,1,0],[0,1,0],[0,1,0]]
        
        X_inits      = [np.array([[1,100,6,0.1,100,6,0.1,100,0,2,100,0,0]]),
                        np.array([[0,1,1,100,1,1,0,100,1]]),
                        np.array([[0,1,0,0,0,30,0,1,0,1,0,100]])]
        
        return X_inits, init_assigns  
    
    def BO_(self,domains_,X_step,Y_step):
        
        bo_steps = []
        x_next   = []
        GP_      = []
        
        for u in range(len(domains_)):
            
            # GPyOpt.methods.BayesianOptimization(
            #     None,
            #     domain=domains_[u],
            #     X=X_step[u],
            #     Y=Y_step[u],
            #     acquisition_type='LCB')

            bo_steps.append(GPyOpt.methods.BayesianOptimization(
                f=None, domain=domains_[u], X=X_step[u],
                Y=Y_step[u],
                acquisition_type=self.acquisition_type,
                model_type='GP', exact_feval=True,
                cost_withGradients=None
            ))

            x_next.append(bo_steps[-1].suggest_next_locations()[0])
            GP_.append(bo_steps[-1].model.model)

        return x_next, GP_
    
    
    def BO_ens(self,domains_,X_step,Y_step,constr_):
        
        bo_steps = []
        x_next   = []
        GP_      = []
        
        for u in range(len(domains_)):

            bo_steps.append(GPyOpt.methods.BayesianOptimization(
                f=None,
                domain=domains_[u],
                constraints=constr_, 
                X=X_step[u],
                Y=Y_step[u],
                acquisition_type=self.acquisition_type, 
                model_type='GP',
                exact_feval=True,
                cost_withGradients=None
            ))
            
            x_next.append(bo_steps[-1].suggest_next_locations()[0])
            GP_.append(bo_steps[-1].model.model)    
        
        return x_next, GP_
    
    def fit(self, X, Y):

        # -------------------------------------------
        # Initial decomposition and hyper-parameters
        # -------------------------------------------
        X_inits, self._assigns          = self.load_initial_model()
        domains_, dummy_, self.compons_ = get_clustered_domains(self.modind,self.hypMAP,self.noHyp,self._assigns)
        self.domains_                   = domains_

        # -------------------------------------------
        # Obtain initial BO objective
        # -------------------------------------------
        Y_inits      = [np.array([self.evaluate_CV_objective(X.copy(), Y.copy(), self.get_model(domains_[k],k,self.compons_,X_inits[k][0]))[0]]).reshape((1,1)) for k in range(len(X_inits))] 
        
        X_step       = X_inits
        Y_step       = Y_inits
        self.scores_ = list(Y_inits[0])

        eva_prop_ = list([{
            'name': 'initial',
            eva.get_metric(): -float(self.scores_[0])
        }])  # element 0

        assert len(self.scores_) == len(eva_prop_)
        
        # -------------------------------------------
        # Initialize BO iterations counter
        # -------------------------------------------        
        current_iter = 0

        time_start_fit = time.time()
        assert len(self.model_) == 0
        for i in tnrange(self.num_iter, desc='BO progress'):
            x_next, GP_ = self.BO_(self.domains_,X_step,Y_step)
            self.GP_    = GP_
            y_next      = []
            for u in range(self.num_components):

                self.models_.append(self.get_model(self.domains_[u],u,self.compons_,x_next[u]))
                rval_model = self.get_model(self.domains_[u],u,self.compons_,x_next[u])

                y_next_, modb_, eva_prp = self.evaluate_CV_objective(X.copy(), Y.copy(), rval_model)

                eva_prp['iter'] = current_iter
                eva_prp['component_idx'] = u
                eva_prp['hyperparameter_properties'] = rval_model.get_properties()
                eva_prp['model'] = str(rval_model)

                y_next.append(y_next_)
                
                self.models_.append(modb_)
                
                X_step[u] = np.vstack((X_step[u], x_next[u]))
                Y_step[u] = np.vstack((Y_step[u], y_next_))
                self.scores_.append(y_next_) 
                eva_prop_.append(eva_prp)
                assert len(self.scores_) == len(eva_prop_)
            current_iter += 1

            time_fit = time.time() - time_start_fit
            time_fold = (self.num_iter) * (time_fit/current_iter)
            time_iter_info = '{:0.0f}s ({:0.0f}s) ({:0.0f}s)'.format(
                time_fit,
                time_fit/current_iter,
                time_fold)
            
            ### Update Kernel

            if (np.mod(i,self.kernel_freq) == 0) and (i != 0):
                
                msg_  = "Iteration number: " + str(current_iter) + " " + time_iter_info + " ||--------------------------- ** Updating kernel ** ---------------------------||                                      "  

                logger.warning('{}'.format(msg_))
                
                self.merged_domain_, self.X_merged, self.Y_merged = merge_domains(self.domains_, X_step, Y_step, self.compons_)
                Decomposed_kern, Gposter_                         = self.Kernel_decomposition([self.X_merged], [self.Y_merged], [self.merged_domain_])            
                self.Gposter_                                     = Gposter_
                self.kernel_merged_domain_                        = self.merged_domain_
                self.domains_ , dummy_dims, self.compons_         = get_clustered_domains(self.modind, self.hypMAP, self.noHyp, Gposter_)
                X_step, Y_step                                    = split_domain(self.domains_, self.merged_domain_, self.compons_, self.X_merged, self.Y_merged)
                self.X_step  = X_step
                self.Y_step  = Y_step
                
            ##########
            
            else:

                display_msg = " "
            
                for u in range(self.num_components):
                
                    display_msg = display_msg + "[[" + self.get_model(self.domains_[u],u,self.compons_,x_next[u]).name + "]]" + ", " 
                
                best_ = str(np.min(np.array([np.min(self.GP_[u].Y) for u in range(self.num_components)]))) 
            
                msg_  = "Iteration number: " + str(current_iter) + " " + time_iter_info + ", Current pipelines: " + display_msg  + "BO objective: " + str(best_)

                logger.warning('{}'.format(msg_))

        self.X_step  = X_step
        self.Y_step  = Y_step

        #self.x_opts_ = [X_step[u][np.argmin(self.GP_[u].Y)] for u in range(self.num_components)]
        #self.y_opts_ = np.array([np.min(self.GP_[u].Y) for u in range(self.num_components)])
        #best_comp    = np.argmin(self.y_opts_)
        #self.x_opt   = self.x_opts_[best_comp]
        
        #self.model = self.get_model(self.domains_[best_comp],best_comp,self.compons_,self.x_opt)
        
        self.merged_domain_, self.X_merged, self.Y_merged = merge_domains(self.domains_, X_step, Y_step, self.compons_)
        dummy_x, final_GPmodel                            = self.BO_([self.merged_domain_],[self.X_merged], [self.Y_merged])
        self.GPmodel_                                     = final_GPmodel
        self.x_opt                                        = self.X_merged[np.argmin(final_GPmodel[0].Y)]
        self.model = self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], self.x_opt)
        
        
        printmd("\r{0}".format("**The best model is: **"+self.model.name))
        self.model.fit(X.copy(), Y.copy())
        
        if self.ensemble:
            self.ensemble_score = 0
            printmd("\r{0}".format(" |||| Now building the ensemble..."))
            self.ensemble_models, self.ensemble_weights = self.get_ensemble_(X.copy(), Y.copy(), CV_=5)
            #self.ensemble_models, self.ensemble_weights = self.build_ensemble_(X, Y, CV_=5)

            printmd("\r{0}".format("**Ensemble: **"+ str([self.ensemble_models[k].name for k in range(len(self.ensemble_models))])))
            printmd("\r{0}".format("**Ensemble weights: **"+ str(self.ensemble_weights))) 
            
            best_score      = float(np.min(np.array(self.scores_)))

            logger.info('score:{:0.3f} {} ensemble score:{} {}'.format(
                best_score,
                self.model.name,
                str(self.ensemble_score),
                str([self.ensemble_models[k].name for k in range(len(self.ensemble_models))])))

            if self.ensemble_score < best_score:
                
                printmd("\r{0}".format("**The ensemble helps!**"))
            
            else:  
                
                printmd("\r{0}".format("**The ensemble did not help.**"))
        return eva_prop_
    #----------------------    
        
    def predict(self,X):

        pred      = self.model.predict(X)

        #W         = self.get_weights()
        #bigpreds_ = [self.models_[k].predict(X)*W[k] for k in range(self.num_iter)] 
        pred_ens  = pred#np.sum(np.array(bigpreds_),axis=0) 

        return pred, pred_ens
    
    #----------------------
    
    def get_ensemble(self,X_, Y_label):
    
        ensemble_size  = self.ensemble_size 
        BO_num_iter    = 30
        x_opts_        = self.X_merged[np.argsort(np.array(self.GPmodel_[0].Y),axis=0)[0:ensemble_size]]
    
        pulled_models_ = [self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_].reshape(x_opts_[mm_].shape[1],)) for mm_ in range(len(x_opts_))]

        for k in range(len(pulled_models_)): 

            pulled_models_[k].fit(X_.copy(),Y_label)
    
    
        ens_domain     = [{'name': 'W'+str(k), 'type': 'continuous', 'domain': (0,1),'dimensionality': 1} for k in range(ensemble_size)]
    
    
        X_inits      = (1/ensemble_size)*np.ones((1,ensemble_size))
        assert 0
        Y_inits      = evaluate_ensemble(X_, Y_label, pulled_models_, X_inits, 'AUC')         
        X_step_ens   = X_inits
        Y_step_ens   = np.array([Y_inits]).reshape(1,1)
        y_next       = []

        for i in range(BO_num_iter):
     
            x_next, GP_ens = self.BO_([ens_domain], [X_step_ens], [Y_step_ens])
            x_next         = x_next/np.sum(x_next)
            y_next_        = evaluate_ensemble(X_, Y_label, pulled_models_, x_next, 'AUC')
            y_next.append(y_next_)                  
            X_step_ens = np.vstack((X_step_ens, x_next))
            Y_step_ens = np.vstack((Y_step_ens, y_next_))
            

        ensemble_weights = X_step_ens[np.argmin(GP_ens[0].Y)]       
    
        return pulled_models_, ensemble_weights 
    
    def get_ensemble_(self,X_, Y_label, CV_):
    
        ensemble_size  = self.ensemble_size 
        BO_num_iter    = 30
        x_opts_        = self.X_merged[np.argsort(np.array(self.GPmodel_[0].Y),axis=0)[0:ensemble_size]]
    
        ens_models_    = [self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_].reshape(x_opts_[mm_].shape[1],)) for mm_ in range(len(x_opts_))]
        pulled_models_ = [[self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_].reshape(x_opts_[mm_].shape[1],))]*CV_ for mm_ in range(len(x_opts_))]
        skf_           = StratifiedKFold(n_splits=CV_) 
        X_test_        = []
        Y_test_        = []
        X_train_       = []
        Y_train_       = []
                
        for train_index, test_index in skf_.split(X_, Y_label):
            
            X_train_.append(X_.loc[X_.index[train_index]])
            Y_train_.append(Y_label.loc[Y_label.index[train_index]])
            X_test_.append(X_.loc[X_.index[test_index]])
            Y_test_.append(Y_label.loc[Y_label.index[test_index]])
        
        for k in range(len(pulled_models_)):
            
            ens_models_[k].fit(X_.copy(), Y_label.copy())
            ee            = 0
            pulled_models_init = list()
            for idx in range(CV_):
                pulled_models_init.append(copy.deepcopy(pulled_models_[k][idx]))
            for uu in range(CV_):
                # ee            = 0
                pulled_models_[k][uu] = pulled_models_init[uu]
                pulled_models_[k][uu].fit(X_train_[uu].copy(),Y_train_[uu].copy())
                ee           += 1
        
        ens_domain     = [{'name': 'W'+str(k), 'type': 'continuous', 'domain': (0,1),'dimensionality': 1} for k in range(ensemble_size)]
    
    
        X_inits      = (1/ensemble_size)*np.ones((1,ensemble_size))
        Y_inits      = evaluate_ensemble_(X_test_, Y_test_, pulled_models_, X_inits, 'AUC')         
        X_step_ens   = X_inits
        Y_step_ens   = np.array([Y_inits]).reshape(1,1)
        y_next       = [Y_inits]

        for i in range(BO_num_iter):
            x_next, GP_ens = self.BO_([ens_domain], [X_step_ens], [Y_step_ens])
            x_next         = x_next/np.sum(x_next)
            y_next_        = evaluate_ensemble_(X_test_, Y_test_, pulled_models_, x_next, 'AUC')
            y_next.append(y_next_)                  
            X_step_ens = np.vstack((X_step_ens, x_next))
            Y_step_ens = np.vstack((Y_step_ens, y_next_))
        
        self.ensemble_score = np.min(np.array(y_next))
        ensemble_weights    = X_step_ens[np.argmin(GP_ens[0].Y)]

        return ens_models_, ensemble_weights 
    
    
    def build_ensemble_(self,X_, Y_label, CV_):
    
        ensemble_size  = self.ensemble_size 
        BO_num_iter    = 50
        #x_opts_        = self.X_merged[np.unique(np.argsort(np.array(self.GPmodel_[0].Y),axis=0),return_index=True)[1][0:ensemble_size]]    
        
        x_opts_        = self.X_merged[np.argsort(np.array(self.GPmodel_[0].Y),axis=0)[0:ensemble_size]]    
    
        ens_models_    = [self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_].reshape(x_opts_[mm_].shape[1],)) for mm_ in range(len(x_opts_))]
        pulled_models_ = [[self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_].reshape(x_opts_[mm_].shape[1],))]*CV_ for mm_ in range(len(x_opts_))]
        
        #ens_models_    = [self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_]) for mm_ in range(len(x_opts_))]
        #pulled_models_ = [[self.get_model(self.merged_domain_, 0, [list(range(len(self.modind)+len(self.noHyp)))], x_opts_[mm_])]*CV_ for mm_ in range(len(x_opts_))]
        skf_           = StratifiedKFold(n_splits=CV_) 
        X_test_        = []
        Y_test_        = []
        X_train_       = []
        Y_train_       = []
                
        for train_index, test_index in skf_.split(X_, Y_label):
            
            X_train_.append(X_.loc[X_.index[train_index]])
            Y_train_.append(Y_label.loc[Y_label.index[train_index]])
            X_test_.append(X_.loc[X_.index[test_index]])
            Y_test_.append(Y_label.loc[Y_label.index[test_index]])
        
        for k in range(len(pulled_models_)):
            
            ens_models_[k].fit(X_, Y_label)
            
            for uu in range(CV_):
                ee            = 0
                pulled_models_[k][ee].fit(X_train_[ee],Y_train_[ee])
                ee           += 1
        
        ens_domain     = [{'name': 'W'+str(k), 'type': 'continuous', 'domain': (0,1),'dimensionality': 1} for k in range(ensemble_size)]
    
    
        X_inits      = (1/ensemble_size)*np.ones((1,ensemble_size))
        Y_inits      = evaluate_ensemble_(X_test_, Y_test_, pulled_models_, X_inits, 'AUC')         
        X_step_ens   = X_inits
        Y_step_ens   = np.array([Y_inits]).reshape(1,1)
        y_next       = []

        for i in range(BO_num_iter):
            x_next, GP_ens = self.BO_ens([ens_domain], [X_step_ens], [Y_step_ens], get_ensemble_constraints(ensemble_size))
            x_next         = x_next/np.sum(x_next)
            y_next_        = evaluate_ensemble_(X_test_.copy(), Y_test_.copy(), pulled_models_, x_next, 'AUC')
            y_next.append(y_next_)                  
            X_step_ens = np.vstack((X_step_ens, x_next))
            Y_step_ens = np.vstack((Y_step_ens, y_next_))
             
        ensemble_weights    = X_step_ens[np.argmin(GP_ens[0].Y)]       
    
        return ens_models_, ensemble_weights 
    

    def get_weights(self):
    
        #mu_   = self.GP_.model.model.predict(self.GP_.model.model.X)[0]
        #vr_   = self.GP_.model.model.predict(self.GP_.model.model.X)[1]

        #M_    = -np.repeat(mu_, len(mu_), axis=1) + np.transpose(np.repeat(mu_, len(mu_), axis=1))
        #V_    = np.sqrt(np.repeat(vr_, len(vr_), axis=1) + np.transpose(np.repeat(vr_, len(vr_), axis=1)))
        #Args_ = norm.cdf(np.divide(M_,V_))
        #W_    = [np.prod(Args_[u, [x for x in range(len(Args_[u,:])) if x != u]]) for u in range(len(Args_))] 
        W     = np.array([-1*(self.scores_[k]<0)*self.scores_[k] for k in range(len(self.scores_))])
        W_    = (W-np.min(W))/np.sum((W-np.min(W)))

        return W_
    
    
    def Kernel_decomposition(self,dataX,dataY,domn_):

        num_hyperparams  = len(domn_[0])-1
        M                = self.num_components
        alphas_          = [10]*M
        init_assign      = [np.argmax(self._assigns[k])+1 for k in range(len(self._assigns))]
        Late_assign      = [init_assign]
        Gibbs_iters      = self.Gibbs_iter
        Burn_in          = self.burn_in 
        GibbsPoster      = []
    
        dummy_x, GPmodel = self.BO_(domn_, dataX, dataY)
        
        for g in range(Gibbs_iters):
            for h in range(num_hyperparams):
                Gumbel_dummy          = np.random.gumbel(size=M)
                Temp_assigns          = np.repeat(np.array(Late_assign[-1]), M, axis=0).reshape(num_hyperparams,M).T
                Temp_assigns[:,h]     = np.array(list(range(M)))+1
                Gram_matxs            = [get_GPy_logLikelhood(dataX[0],dataY[0],get_Gibbs_kernel_(GPmodel[0], M, list(Temp_assigns[m,:]))) for m in range(M)]
                A_m                   = [[np.sum((Temp_assigns[k,:]==m+1)*1) for m in range(M)] for k in range(M)]
                Gumbal_max_vec        = Gumbel_dummy + Gram_matxs + 0.5*(num_hyperparams+1)*np.log(2*np.pi) + np.log(np.diag(np.matrix(A_m))+alphas_)  
                z_assign              = np.argmax(Gumbal_max_vec) 
                Temp_Late_assign      = Late_assign[-1].copy()
                Temp_Late_assign[h]   = z_assign + 1
                Late_assign.append(Temp_Late_assign)
            
        Late_assign_      = Late_assign[Burn_in:][::num_hyperparams]        
        GibbsPoster       = [[len(np.where(np.array(Late_assign_)[:,k]==m+1)[0])/len(np.array(Late_assign_)[:,k]) for m in range(M)] for k in range(num_hyperparams)]
    
        final_assignments = [int(np.argmax(np.array(GibbsPoster[u]))+1) for u in range(len(GibbsPoster))]
        M_                = len(np.unique(final_assignments))
        Decomposed_kern   = get_Gibbs_kernel_(GPmodel[0],M_,final_assignments) 
    
        return Decomposed_kern, GibbsPoster
        
        # Visualization
        
    def visualize_data(self,X):
            
        return pivot_ui(X)
        
    def APReport(self):

        report_d = dict()
        
        best_score = float(np.min(np.array(self.scores_)))

        report_d['best_score_single_pipeline'] = -best_score
        report_d['model_names_single_pipeline'] = self.model.name
        report_d['ensemble_score'] = -float(self.ensemble_score)
        report_d['ensemble_pipelines'] = [self.ensemble_models[u].name for u in range(self.ensemble_size)]
        report_d['ensemble_pipelines_weight'] = [self.ensemble_weights[u] for u in range(self.ensemble_size)]
        report_d['optimisation_metric'] = eva.get_metric()
        report_d['hyperparameter_properties'] = self.model.get_properties()
        report_d['acquisition_type'] = self.acquisition_type

        logger.info('score:{:0.3f} {} ensemble score:{:0.3f} {}'.format(
            best_score,
            self.model.name,
            self.ensemble_score,
            str([self.ensemble_models[k].name for k in range(len(self.ensemble_models))])))

        if self.ensemble_score < best_score:
            logger.info("**The ensemble helps!**")
        else:  
            logger.info("**The ensemble did not help.**")

        printmd("\r{0}".format("***Ensemble Report***"))
        printmd("\r{0}".format("**----------------------**"))
    
        for u in range(self.ensemble_size):
        
            printmd("\r{0}".format("**Rank"+ str(u) +":   "+ self.ensemble_models[u].name+",   Ensemble weight: "+ str(self.ensemble_weights[u]) +"**")) # print picture and colored text
            printmd("\r{0}".format("**----------------------**"))
            print(self.ensemble_models[u].__dict__)
            logger.info('ensemble_models dict:{}'.format(self.ensemble_models[u].__dict__))
            printmd("\r{0}".format("**_____________________________________________**"))
            printmd("\r{0}".format(self.ensemble_models[u].explained))
            if self.ensemble_models[u].image_name is not None:
                i = Image(filename=self.ensemble_models[u].image_name,width=self.ensemble_models[u].image_size[0],height=self.ensemble_models[u].image_size[1])
                display(i,width=self.ensemble_models[u].image_size[0],height=self.ensemble_models[u].image_size[1])
            else:
                logger.debug('warning:no image_name')

        printmd("\r{0}".format("**----------------------**"))
        printmd("\r{0}".format("***Kernel Report***"))

        report_d['kernel_members'] = dict()
        for k in range(self.num_components):
            report_d['kernel_members'][k] = [self.model_names[self.compons_[k][m]] for m in range(len(self.compons_[k]))]
            printmd("\r{0}".format("**Component"+" "+str(k)+"**"))
            printmd("\r{0}".format("**Members:"+" "+str([self.model_names[self.compons_[k][m]] for m in range(len(self.compons_[k]))])+"**")) 
            print(self.GP_[k].kern)
            logger.info('GP kern:{}'.format(self.GP_[k].kern))
        return report_d


# In[141]:

def predict_ens(X, models_, W):
        
    bigpreds_ = [models_[k].predict(X)*W[0][k] for k in range(len(models_))] 
    pred_ens  = np.sum(np.array(bigpreds_),axis=0) 

    return pred_ens

def evaluate_ensemble(X, Y, models_, W, metric):

    assert 0
    
    score_ens = -1*roc_auc_score(Y,predict_ens(X, models_, W)) 
    
    return score_ens
    
    #----------------------
def predict_ens_(X, models_, W):

    # logger.info('predict_ens_{}'.format(models_))
    # for k in range(len(models_)):
    #     pred = models_[k].predict(X)
    #     logger.info('pred {}'.format(pred.shape))
    #     logger.info('W0 {}'.format(W[0]))
    
    bigpreds_ = [models_[k].predict(X)*W[0][k] for k in range(len(models_))] 
    pred_ens  = np.sum(np.array(bigpreds_),axis=0) 

    return pred_ens

def evaluate_ensemble_(X, Y, models_, W, metric):

    logger.info('evaluate_ensemble_ {} w:{}'.format(metric, W))
    
    pulmodels_ = [[models_[k][m] for k in range(len(models_))] for m in range(len(models_[0]))]
    score_ens_roc       = np.mean(np.array([-1*eva.roc_auc_score  (np.array(Y[m]).reshape((len(Y[m]),1)),predict_ens_(X[m], pulmodels_[m], W)) for m in range(len(pulmodels_))]))
    score_ens_metric    = np.mean(np.array([-1*eva.score_proba(np.array(Y[m]).reshape((len(Y[m]),1)),predict_ens_(X[m], pulmodels_[m], W)) for m in range(len(pulmodels_))])) 
    if False:
        logger.info('old score (roc)')
        score_ens  = score_ens_roc
    else:
        logger.info('new score (via cli)')
        score_ens  = score_ens_metric
    logger.info('-evaluate_ensemble_ {:0.4f} {:0.4f} {:0.4f}'.format(score_ens, score_ens_roc, score_ens_metric))
    return score_ens
    
    #----------------------   
    
def get_ensemble_constraints(ens_size):
    
    
    stems_ = 'x[:,0]'

    for uu_ in range(ens_size-1):
    
        stems_ = stems_ + ' + x[:,' + str(uu_+1) + ']'

    cnstrs_ = '(' + stems_ + ')' 
    

    ens_constraints = [{'name': 'constr_1','constraint': cnstrs_ + ' - 1 - 0.0001'},
                       {'name': 'constr_2','constraint': '1 - ' + cnstrs_ + ' - 0.0001'}]

    return ens_constraints
    

def evaluate_clf(X, Y, model_input, n_folds, visualize=False):

    metric_      = np.zeros(n_folds)
    indx         = 0
    skf          = StratifiedKFold(n_splits=n_folds)
    
    score_roc_lst = list()
    score_prc_lst = list()

    is_pred_proba = True
    
    if hasattr(model_input, 'get_is_pred_proba'):
        is_pred_proba = model_input.get_is_pred_proba()

    for train_index, test_index in skf.split(X, Y):

        X_train  = X.loc[X.index[train_index]].copy()
        Y_train  = Y.loc[Y.index[train_index]].copy()
        X_test   = X.loc[X.index[test_index]].copy()
        Y_test   = Y.loc[Y.index[test_index]].copy()

        assert set(Y_train) == set(Y_test)
        nnan = 0

        model = copy.deepcopy(model_input)

        if is_pred_proba:
            logger.info('+fit {} {}'.format(X_train.shape,list(set(np.ravel(Y_train)))))
            model.fit(X_train, Y_train)
            preds            = model.predict(X_test)
            nnan = sum(np.ravel(np.isnan(preds)))

        # metric_[indx]    = roc_auc_score(Y_test, preds)
        # metric_[indx]    = average_precision_score(Y_test, preds)

        if nnan:
            logger.info('nan preds:{} ratio:{} ratio:{}'.format(
                nnan,
                nnan/float(len(np.ravel(np.isnan(preds)))),
                nnan/len(preds)
            ))
            logger.info(
                'warning: nan in predictions, cannot calculate score'
                ': use low score instead, model:{}'.format(model.name))
            score_roc, score_prc = 0.5, 0.0
        elif not is_pred_proba:
            logger.info(
                'warning: clf has no probabilities'
                ': use low score instead, model:{}'.format(model.name))
            score_roc, score_prc = 0.5, 0.0
        else:
            # score_roc = metrics.roc_auc_score(Y_test, preds)
            # score_prc = metrics.average_precision_score(Y_test, preds)
            score_roc = eva.roc_auc_score(Y_test, preds)
            score_prc = eva.average_precision_score(Y_test, preds)

        score_roc_lst.append(score_roc)
        score_prc_lst.append(score_prc)

        metric_to_score = {
            'aucprc': score_prc,
            'aucroc': score_roc
        }
        metric_[indx] = eva.score_proba(Y_test, preds) if is_pred_proba and not nnan else metric_to_score[eva.get_metric()]
        logger.info('{} metric_{}'.format(indx, metric_))
        
        if visualize:
            printmd("\r{0}".format("**Cross-validation score: **"+ str(metric_[indx])))
            print("---------------------------------------------------------")
        
        indx          += 1
    
    Output = (metric_.mean(), 1.96*np.std(metric_)/np.sqrt(len(metric_))) 
    
    if visualize:
        printmd("\r{0}".format("**Final Cross-validation score: **"+ str(Output)))
        print("---------------------------------------------------------")

    roc = float(np.mean(score_roc_lst))
    prc = float(np.mean(score_prc_lst))
    logger.info(' evaluate_clf::aucroc {:0.4f} #({}) {}'.format(
        roc,
        len(score_roc_lst),
        model.name))
    logger.info(' evaluate_clf::aucprc {:0.4f} #({}) {}'.format(
        prc,
        len(score_prc_lst),
        model.name
    ))
    eva_prop = {
        'aucprc': prc,
        'aucroc': roc,
        'name': model.name,
        'cv': n_folds
    }
    logger.info('-evaluate_clf {} {}'.format(Output, eva_prop))
    return Output, eva_prop


def evaluate_ens(X, Y, model_input, n_folds, visualize=False):
    
    logger.info('+evaluate_ens shape x:{} y:{}'.format(X.shape, Y.shape))
    logger.info('nan x:{} {}'.format(
        sum(np.ravel(np.isnan(X))),
        sum(np.ravel(np.isnan(X)))/len(np.ravel(X))))

    metric_      = np.zeros(n_folds)
    metric_ens   = np.zeros(n_folds)
    indx         = 0
    skf          = StratifiedKFold(n_splits=n_folds)
    start_fold   = 0

    score_d = dict()
    score_d['clf']  = dict()
    score_d['clf_ens']  = dict()

    score_d['clf']['roc_lst'] = list()
    score_d['clf_ens']['roc_lst'] = list()
    score_d['clf']['prc_lst'] = list()
    score_d['clf_ens']['prc_lst'] = list()
    eva_prop_lst = []
    for train_index, test_index in skf.split(X, Y):

        X_train  = X.loc[X.index[train_index]]
        Y_train  = Y.loc[Y.index[train_index]]
        X_test   = X.loc[X.index[test_index]]
        Y_test   = Y.loc[Y.index[test_index]]
        
        if indx >= start_fold:

            model = copy.deepcopy(model_input)  # start with the initial model

            eva_prop = model.fit(X_train, Y_train)
            eva_prop_lst.append(eva_prop)

            preds             = model.predict(X_test)[0]
            #preds_ens        = model.predict(X_test)[1]

            preds_ens         = predict_ens(X_test, model.ensemble_models, [model.ensemble_weights])
        
            # metric_[indx]     = roc_auc_score(Y_test, preds)
            # metric_ens[indx]  = roc_auc_score(Y_test, preds_ens)
            
            # metric_[indx]     = average_precision_score(Y_test, preds)
            # metric_ens[indx]  = average_precision_score(Y_test, preds_ens)

            score_d['clf']['roc_cur'] = eva.roc_auc_score(Y_test, preds)
            score_d['clf_ens']['roc_cur'] = eva.roc_auc_score(Y_test, preds_ens)

            score_d['clf']['prc_cur'] = eva.average_precision_score(Y_test, preds)
            score_d['clf_ens']['prc_cur'] = eva.average_precision_score(Y_test, preds_ens)

            score_d['clf']['roc_lst'].append(score_d['clf']['roc_cur'])
            score_d['clf_ens']['roc_lst'].append(score_d['clf_ens']['roc_cur'])
            score_d['clf']['prc_lst'].append(score_d['clf']['prc_cur'])
            score_d['clf_ens']['prc_lst'].append(score_d['clf_ens']['prc_cur'])

            metric_[indx] = eva.score_proba(Y_test, preds)
            metric_ens[indx]  = eva.score_proba(Y_test, preds_ens)

            if visualize:
                printmd("\r{0}".format("**Cross-validation score: **"+ str(metric_[indx])))
                printmd("\r{0}".format("**Cross-validation score with ensembles: **"+ str(metric_ens[indx])))
                print("---------------------------------------------------------")
        
        indx          += 1
    
    Output     = (metric_.mean(),1.96*np.std(metric_)/np.sqrt(len(metric_))) 
    Output_ens = (metric_ens.mean(),1.96*np.std(metric_ens)/np.sqrt(len(metric_ens))) 
    
    if visualize:
        printmd("\r{0}".format("**Final Cross-validation score: **"+ str(Output)))
        printmd("\r{0}".format("**Final Cross-validation score with ensembles: **"+ str(Output_ens)))
        print("---------------------------------------------------------")

    return Output, Output_ens, score_d, model, eva_prop_lst


# In[143]:

def get_Gibbs_kernel_(APmodel,M,assignments):

    assignments = np.array(assignments)
    baseKernels = GPy.kern.Matern52(input_dim=1,active_dims=[0],variance = APmodel.kern.variance[0], lengthscale = APmodel.kern.lengthscale[0])
    newKernels  = [baseKernels] + [GPy.kern.Matern52(input_dim=len(list(np.where(assignments==m+1)[0])), active_dims=list(np.where(assignments==m+1)[0]),variance = APmodel.kern.variance[0], lengthscale = APmodel.kern.lengthscale[0]) for m in range(M)]
    FinalKernel = [] #
    
    for k in range(len(newKernels)):
        if newKernels[k].input_dim > 0 and type(FinalKernel)==list:
            FinalKernel = newKernels[k]
        elif newKernels[k].input_dim > 0 and type(FinalKernel)!=list:
            FinalKernel = FinalKernel + newKernels[k] 
        
    return FinalKernel

def get_GPy_logLikelhood(dataX,dataY,Kernel_in):
    
    return GPy.models.GPRegression(dataX, dataY, kernel=Kernel_in).log_likelihood()
    


# In[144]:

def get_clustered_domains(modind,hypMAP,noHyp,assigns):
        
    # Define domains for classifier sets
    #[RF 0 (1,2),GBM 1 (3,4,5),XGB 2 (6,7,8),Ada 3 (9,10),Bag 4 (11,12,13,14), BNB 5 (15), 
    # GNB 6 [], MNB 7 (16), LR 8 (17,18,19), Perc 9 (20,21), DT 10 (22), QDA_ 11 (),
    # LDA_ 12 (), KNN 13 (23,24,25,26), LSVM 14 (27), NN 15 (28,29,30,31)]
    
    GibbScores_  = [np.mean(np.array([assigns[hypMAP[modind[m]][k]-1] for k in range(len(hypMAP[modind[m]]))]),axis=0) for m in range(len(modind))]
    num_clusters = len(assigns[0])
    chunk_size   = int(np.floor(len(modind)/num_clusters))
    
    model_indxs  = [[]]*num_clusters
    dims_        = [[]]*num_clusters
    cumulat_     = []
    
    for nc_ in range(num_clusters):
        
        if nc_ == 0:
            model_indxs[nc_] = list(np.argsort(np.array(GibbScores_)[:,nc_])[::-1])[:chunk_size]
            cumulat_         = cumulat_ + model_indxs[nc_] 
        else:
            remaining_models = np.argsort(np.array(GibbScores_)[:,nc_])[::-1]
            remaining_indxs  = [remaining_models[k] for k in range(len(remaining_models)) if remaining_models[k] not in cumulat_]
            
            if nc_ < num_clusters-1:
                model_indxs[nc_] = remaining_indxs[:chunk_size]  
            else:
                model_indxs[nc_] = remaining_indxs 
            
            cumulat_         = cumulat_ + model_indxs[nc_]
            
    model_indexes     = [[modind[model_indxs[nn][kk]] for kk in range(len(model_indxs[nn]))] for nn in range(len(model_indxs))]       
    model_indexes[-1] = model_indexes[-1] + noHyp
    
    base_models_      = [RandomForest(),GradientBoosting(),XGboost(),Adaboost(),Bagging(),BernNaiveBayes(),
                         GaussNaiveBayes(),MultinomialNaiveBayes(),LogisticReg(),Percept(),DecisionTrees(),
                         QDA_(),LDA_(),KNN(),LinearSVM(),NeuralNet()]
    
    component_maps     = [model_indexes[k] for k in range(num_clusters)]
    domains_           = [[{'name': 'classifier', 'type': 'categorical', 'domain': tuple(range(len(model_indexes[k]))),'dimensionality': 1}] for k in range(num_clusters)]
            
    for nums_cl in range(num_clusters): 
        for mm_ in range(len(model_indexes[nums_cl])):
            logger.info("base_hyper_parameter_space {}".format(base_models_[model_indexes[nums_cl][mm_]].get_hyperparameter_space()))
            domains_[nums_cl]  = domains_[nums_cl] + base_models_[model_indexes[nums_cl][mm_]].get_hyperparameter_space()
        
        dims_[nums_cl]         = len(domains_[nums_cl])
    
    return domains_ , dims_, component_maps        


def merge_domains(doms_,X_s,Y_s,modelcompons_):
    
    merged_doms_ = [{'name': 'classifier', 'type': 'categorical',
                     'domain': tuple(range(np.sum([len(modelcompons_[k]) for k in range(len(modelcompons_))]))),
                     'dimensionality': 1}]
    
    for k in range(len(doms_)):
        
        merged_doms_ = merged_doms_ + doms_[k][1:] 
        
    T            = len(X_s[0])
    NumHyp       = len(merged_doms_)
    X_merged     = [] 
    Y_merged     = [] 
    
    for tt in range(T):
        
        X_temp = [] 
        
        for mm in range(len(doms_)):
 
            X_temp = X_temp + list(X_s[mm][tt][1:])
                
        
        Xbig_temp      = np.repeat(np.array(X_temp).reshape(len(X_temp),1),len(X_s),axis=1).T
        Xbig_temp_     = np.array([modelcompons_[k][int(X_s[k][tt][0])] for k in range(len(X_s))])
        Xbig_temp      = np.hstack((Xbig_temp_.reshape(len(Xbig_temp_),1),Xbig_temp))
        
        X_merged.append(Xbig_temp)        
        Y_merged.append(np.array([Y_s[k][tt] for k in range(len(X_s))]))
        
    
    X_merged_final = X_merged[0]
    Y_merged_final = Y_merged[0]

    for u in range(1,len(Y_merged)):

        X_merged_final = np.vstack((X_merged_final,X_merged[u]))
        Y_merged_final = np.vstack((Y_merged_final,Y_merged[u]))
        

    return merged_doms_,X_merged_final,Y_merged_final


# In[146]:

def split_domain(doms_, dom_merged, comp_map, X_m, Y_m):
    
    X_s = [[]]*len(doms_)
    Y_s = [[]]*len(doms_)
    
    # map of models to classifier index
    dom_lists = [dom_merged[k]['name'] for k in range(len(dom_merged))]
    hyperlist = [[dom_lists.index(doms_[uu_][u]['name']) for u in range(len(doms_[uu_]))] for uu_ in range(len(doms_))]
    
    for u in range(len(X_m)):
        relevant_comp = np.where(np.array([(X_m[u][0] in comp_map[k])*1 for k in range(len(comp_map))])==1)[0][0] 
        X_coded       = [comp_map[relevant_comp].index(X_m[u][0])] + list(X_m[u][hyperlist[relevant_comp][1:]])
        
        if len(X_s[relevant_comp]) == 0:
            
            X_s[relevant_comp] = [X_coded]
            Y_s[relevant_comp] = list(Y_m[u])
        
        else:    
            
            X_s[relevant_comp].append(X_coded)
            Y_s[relevant_comp] = Y_s[relevant_comp] + list(Y_m[u])
            
    Y_s_    = [[Y_s[u][k] for k in range(len(Y_s[u]))] for u in range(len(Y_s))]   

    
    max_dim = np.max(np.array([len(X_s[k]) for k in range(len(X_s))]))  
    XX__s   = [[]]*len(doms_)
    YY__s   = [[]]*len(doms_)
    
    for u in range(len(X_s)):
        
        XX_s     = np.repeat(np.array(X_s[u]),np.floor(max_dim/np.array(X_s[u]).shape[0]),axis=0)
        YY_s     = np.repeat(np.array(Y_s_[u]),np.floor(max_dim/np.array(X_s[u]).shape[0]),axis=0)
        
        if XX_s.shape[0] < max_dim:

            X_remain = np.repeat(XX_s[XX_s.shape[0]-1,:].reshape(1,len(XX_s[XX_s.shape[0]-1,:])),max_dim-XX_s.shape[0],axis=0)
            Y_remain = YY_s[XX_s.shape[0]-1]*np.ones(max_dim-XX_s.shape[0])
            
            XX_s     = np.vstack((XX_s,X_remain))
            YY_s     = np.vstack((YY_s.reshape(len(YY_s),1),Y_remain.reshape(len(Y_remain),1)))
            XX__s[u] = XX_s
            YY__s[u] = YY_s.reshape((max_dim,1))
            
        else:
            
            XX__s[u] = XX_s[0:max_dim]
            YY__s[u] = YY_s[0:max_dim].reshape((max_dim,1))
            
            
    return XX__s, YY__s


eva = evaluate()

logger = logging.getLogger()
logging.basicConfig(level=logging.WARNING, format='%(message)s')

nmax_model = 0
model_xtr_arg = 0
gis_ipython = is_ipython()
