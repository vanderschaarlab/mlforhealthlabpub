
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger()

classifier_xtr_arg = 0


def classifier_set_xtr_arg(xtr):
    global classifier_xtr_arg
    classifier_xtr_arg = xtr


def classifier_get_xtr_arg():
    return classifier_xtr_arg


class baseClassifier:
    """ A parent class for all classification estimators. Particular classif. models will inherit from this class."""
    # methods
    def __init__(self):

        self.model_type = 'classifier'
        
    def fit(self, X, Y):

        logging.info(' fit::clf {} {}'.format(X.shape, list(set(np.ravel(Y)))))

        if self.name in [
                'MultinomialNaiveBayes',
                'GaussianNaiveBayes',
                'BernoullinNaiveBayes']:

            self.model.fit(np.abs(X), Y)

        else:

            self.model.fit(X, Y)    

    def predict(self,X): 

        if self.is_pred_proba:
            # preds_ = self.model.predict_proba(X)[:,1]
            preds_ = self.model.predict_proba(X)
        else:
            preds_ = self.model.predict(X)
        
        return preds_    
    

class RandomForest(baseClassifier): 
    """ Random forest classifier model."""
    
    def __init__(self,
                 n_estimators=100,
                 criterion='gini'):
        
        super().__init__()
   
        self.n_estimators  = n_estimators
        self.criterion     = criterion
        self.name          = 'Random Forest'
        self.is_pred_proba = True
        self.model         = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion)
        
        self.explained     = "*Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.*"
        self.image_name    = "RanForest.png"
        self.image_size    = (750,750)

        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ((n_estimators)) : integer, positive, default: 100
        # ((criterion))    : “gini” | “entropy” 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_   = [{'name': 'RandomForest.ntrees', 'type': 'discrete', 'domain': tuple(range(10,1000)),'dimensionality': 1},
                  {'name': 'RandomForest.criterion', 'type': 'categorical', 'domain': (0,1),'dimensionality': 1}]

        return hyp_   
        
    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }



class GradientBoosting(baseClassifier): 
    """ Gradient Boosting classification model."""
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=6):
        
        super().__init__()
   
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.is_pred_proba = True
        self.name          = 'Gradient Boosting'
        self.model         = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                        learning_rate=self.learning_rate,max_depth=self.max_depth)
        
        
        self.explained     = "*Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function..*"
        self.image_name    = "GBM.png"
        self.image_size    = (1000,1000)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ((n_estimators)) : integer, positive, default: 100
        # Number of decision tree estimators
        # ((max_depth)), ((learning_rate)) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'GradientBoosting.n_estimators', 'type': 'discrete', 'domain': tuple(range(10,500)),'dimensionality': 1},
                {'name': 'GradientBoosting.max_depth', 'type': 'discrete', 'domain': tuple(range(1,10)),'dimensionality': 1},
                {'name': 'GradientBoosting.learning_rate', 'type': 'continuous', 'domain': (0.005,0.5),'dimensionality': 1}]
        return hyp_   
        
    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }



class XGboost(baseClassifier): 
    """ XGBoost classification model."""
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=6):
        
        super().__init__()
   
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.name          = 'XGBoost'
        self.is_pred_proba = True
        self.model         = XGBClassifier(n_estimators=self.n_estimators,
                                           learning_rate=self.learning_rate,max_depth=self.max_depth)
        
        self.explained     = "*GBoost is an open-source software library which provides the gradient boosting framework for C++, Java, Python, R, and Julia.*"
        self.image_name    = "XGBoost.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ((n_estimators)) : integer, positive, default: 100
        # ((max_depth)), ((learning_rate)) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'XGboost.n_estimators', 'type': 'discrete', 'domain': tuple(range(10,500)),'dimensionality': 1},
                {'name': 'XGboost.max_depth', 'type': 'discrete', 'domain': tuple(range(1,10)),'dimensionality': 1},
                {'name': 'XGboost.learning_rate', 'type': 'continuous', 'domain': (0.005,0.5),'dimensionality': 1}]
        return hyp_   
        
    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class Adaboost(baseClassifier): 
    """ AdaBoost classification model."""
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=1):
        
        super().__init__()
   
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.name          = 'AdaBoost'
        self.is_pred_proba = True
        self.model         = AdaBoostClassifier(n_estimators=self.n_estimators,
                                                learning_rate=self.learning_rate)
        
        self.explained     = "*AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers.*"
        self.image_name    = "AdaBoost.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ((n_estimators)) : integer, positive, default: 100
        # ((learning_rate)): default = 1.0 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'Adaboost.n_estimators', 'type': 'discrete', 'domain': tuple(range(10,5000)),'dimensionality': 1},
                {'name': 'Adaboost.learning_rate', 'type': 'continuous', 'domain': (0.005,5),'dimensionality': 1}]
        return hyp_
    
    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }



class Bagging(baseClassifier): 
    """ Bagging classification model."""
    
    def __init__(self,
                 base_estimator=None,
                 max_samples=1.0,
                 max_features=1.0,
                 n_estimators=100):
        
        super().__init__()
        logger.info('Bagging::init be:{} ms:{} mf:{} nest:{}'.format(base_estimator, max_samples, max_features, n_estimators))
        self.n_estimators   = n_estimators
        self.max_samples    = max_samples
        self.name           = 'Bagging'
        self.max_features   = max_features
        self.base_estimator = None if classifier_get_xtr_arg() < 1 else base_estimator # LogisticRegression()#base_estimator
        self.is_pred_proba  = True
        self.model          = BaggingClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_samples=self.max_samples,
            base_estimator=self.base_estimator
        )
        
        self.explained     = "*Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.*"
        self.image_name    = "Bagging.png"
        self.image_size    = (500, 500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ((n_estimators)) : integer, positive, default: 100
        # ((max_samples)): maximum number of samples for each estimator, default 1.0
        # Default base estimator is decision trees
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'Bagging.n_estimators', 'type': 'discrete', 'domain': tuple(range(10,5000)),'dimensionality': 1},
                # {'name': 'Bagging.max_samples', 'type': 'continuous', 'domain': (0.005,1),'dimensionality': 1},
                {'name': 'Bagging.max_samples', 'type': 'continuous', 'domain': (0.01,1),'dimensionality': 1},
                {'name': 'Bagging.max_features', 'type': 'continuous', 'domain': (0.005,1),'dimensionality': 1},
                {'name': 'Bagging.base_estimator', 'type': 'categorical', 'domain': (0,1,2,3,4),'dimensionality': 1}]
        return hyp_

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class BernNaiveBayes(baseClassifier): 
    """ Bernoulli Naive Bayes classification model."""
    
    def __init__(self,alpha=1.0): 
        super().__init__()
         
        self.alpha          = alpha    
        self.is_pred_proba  = True
        self.model          = BernoulliNB(alpha=self.alpha)
        self.name           = 'BernoullinNaiveBayes'
        
        self.explained     = "*A Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features.*"
        self.image_name    = "BernoullinNaiveBayes.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # alpha = Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'BernoullinNaiveBayes.alpha', 'type': 'continuous', 'domain': (0.005,5),'dimensionality': 1}]
        
        return hyp_   

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class GaussNaiveBayes(baseClassifier):  
    """ Gaussian Naive Bayes classification model."""
    
    def __init__(self): 
        super().__init__()
   
        self.is_pred_proba = True
        self.model      = GaussianNB()
        self.name       = 'GaussianNaiveBayes' 
        
        self.explained     = "*A Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features.*"
        self.image_name    = "BernoullinNaiveBayes.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = []
        
        return hyp_   

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class MultinomialNaiveBayes(baseClassifier): 
    """ Multinomial Naive Bayes classification model."""
    
    def __init__(self,alpha=1.0): 
        super().__init__()
   
        self.alpha      = alpha 
        self.is_pred_proba = True
        self.model      = MultinomialNB(alpha=self.alpha)
        self.name       = 'MultinomialNaiveBayes' 
        
        self.explained     = "*A Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features.*"
        self.image_name    = "BernoullinNaiveBayes.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # alpha = Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'MultinomialNaiveBayes.alpha', 'type': 'continuous', 'domain': (0.005,5),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class LogisticReg(baseClassifier): 
    """ Logistic Regression classification model."""
    
    def __init__(self,
                 C=1.0,
                 solver='lbfgs',
                 max_iter=100):
        
        super().__init__()
   
        self.C          = C 
        self.max_iter   = max_iter 
        self.solver     = solver
        self.is_pred_proba = True
        self.model      = LogisticRegression(C=self.C,solver=self.solver,max_iter=self.max_iter)
        self.name       = 'LogisticRegression'
        
        self.explained     = "*A statistical model that is usually taken to apply to a binary dependent variable. In regression analysis, logistic regression or logit regression is estimating the parameters of a logistic model. More formally, a logistic model is one where the log-odds of the probability of an event is a linear combination of independent or predictor variables.*"
        self.image_name    = "LogisticRegression.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # C        = Inverse of regularization strength;
        #            must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
        # solver   = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
        # max_iter = 100 default
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'LogisticRegression.C', 'type': 'continuous', 'domain': (0.005,10),'dimensionality': 1},
                {'name': 'LogisticRegression.solver', 'type': 'categorical', 'domain': (0,1,2,3,4),'dimensionality': 1},
                {'name': 'LogisticRegression.max_iter', 'type': 'discrete', 'domain': (0,500),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }



class Percept(baseClassifier): 
    """ Perceptron classification model (1 layer neural networks)."""
    
    def __init__(self,
                 penalty='l2',
                 alpha=0.0001
                 ):
        
        super().__init__()
   
        self.penalty    = penalty 
        self.alpha      = alpha 
        self.is_pred_proba = False
        self.model      = Perceptron(penalty=self.penalty ,alpha=self.alpha)
        self.name       = 'Perceptron'
        
        self.explained     = "*The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not).*"
        self.image_name    = "Perceptron.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # alpha    = regularizer factor
        # penalty  = {‘l2’ or ‘l1’ or ‘elasticnet’}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'Perceptron.alpha', 'type': 'continuous', 'domain': (0.00005,0.001),'dimensionality': 1},
                {'name': 'Perceptron.penalty', 'type': 'categorical', 'domain': (0,1,2),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class DecisionTrees(baseClassifier): 
    """ Decision Tree classification model."""
    
    def __init__(self,criterion='gini'):
        
        super().__init__()
   
        self.criterion  = criterion 
        self.is_pred_proba = False
        self.model      = DecisionTreeClassifier(criterion=self.criterion)
        self.name       = 'DecisionTrees'
        
        self.explained     = "*A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.*"
        self.image_name    = "DecisionTrees.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # criterion = 'gini' or 'entropy'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'DecisionTrees.criterion', 'type': 'categorical', 'domain': (0,1),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class QDA_(baseClassifier): 
    """ Quadratic Discriminant Analysis classification model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.is_pred_proba = True
        self.model      = QuadraticDiscriminantAnalysis()
        self.name       = 'QDA'
        
        self.explained     = "*The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not).*"
        self.image_name    = "Perceptron.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = []
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class LDA_(baseClassifier): 
    """ Linear Discriminant Analysis classification model."""
    
    def __init__(self):
        
        super().__init__()
   
        self.is_pred_proba = True
        self.model      = LinearDiscriminantAnalysis()
        self.name       = 'LDA'
        
        self.explained     = "*The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not).*"
        self.image_name    = "Perceptron.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = []
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class KNN(baseClassifier): 
    """ K nearest neighbor classification model."""
    
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2
                 ):
        
        super().__init__()
   
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.algorithm   = algorithm
        self.leaf_size   = leaf_size
        self.p           = p
        self.is_pred_proba= True
        self.model       = KNeighborsClassifier(n_neighbors=self.n_neighbors,weights=self.weights,
                                                algorithm=self.algorithm,leaf_size=self.leaf_size,p=self.p)
        self.name        = 'KNN'
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # n_neighbors : Number of neighbors , int, optional (default = 5)
        # weights : str or callable, optional (default = ‘uniform’ also 'distance')
        # algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
        # leaf_size : int, optional (default = 30)
        # p : integer 1 or 2, optional (default = 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'KNN.p', 'type': 'discrete', 'domain': (1,2),'dimensionality': 1},
                {'name': 'KNN.algorithm', 'type': 'categorical', 'domain': (0,1,2,3),'dimensionality': 1},
                {'name': 'KNN.weights', 'type': 'categorical', 'domain': (0,1),'dimensionality': 1},
                {'name': 'KNN.n_neighbors', 'type': 'discrete', 'domain': (1,50),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class LinearSVM(baseClassifier): 
    """ Linear SVM classification model."""
    
    def __init__(self,penalty='l2'): 
        super().__init__()
   
        self.penalty    = penalty 
        self.is_pred_proba = False
        self.model      = LinearSVC(penalty=self.penalty,dual=False)
        self.name       = 'LinearSVM'
        
        self.explained     = "*The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not).*"
        self.image_name    = "Perceptron.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # penalty  = {‘l2’ or ‘l1’}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'LinearSVM.penalty', 'type': 'categorical', 'domain': (0,1),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }


class NeuralNet(baseClassifier): 
    """ Neural network classification model."""
    
    def __init__(self,
                 num_layers=2,
                 num_units=100,
                 activation='relu',
                 solver='adam'
                 ):
        
        super().__init__()
   
        self.hidden_layer_sizes = tuple([num_units for k in range(num_layers)])
        self.activation         = activation
        self.solver             = solver
        self.is_pred_proba      = True
        self.model              = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,activation=self.activation,
                                                solver=self.solver)
        self.name               = 'NeuralNet'
        
        self.explained     = "*Artificial neural networks (ANNs) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems learn to perform tasks by considering examples, generally without being programmed with any task-specific rules.*"
        self.image_name    = "NeuralNet.png"
        self.image_size    = (500,500)
        
        # ****Model hyper-parameters****
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        # activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
        # solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_hyperparameter_space(self):
        
        hyp_ = [{'name': 'NeuralNet.num_layers', 'type': 'discrete', 'domain': (1,2),'dimensionality': 1},
                {'name': 'NeuralNet.num_units', 'type': 'discrete', 'domain': (50,200),'dimensionality': 1},
                {'name': 'NeuralNet.solver', 'type': 'categorical', 'domain': (0,1,2),'dimensionality': 1},
                {'name': 'NeuralNet.activation', 'type': 'categorical', 'domain': (0,1,2,3),'dimensionality': 1}]
        
        return hyp_ 

    def get_properties(self):
        return {
            'name': self.name,
            'hyperparameters': {
                'model': str(self.model)
            }
        }

