
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pandas as pd
import numpy as np
import GPy

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")


from GPy import Model, Param
import scipy
from GPy.core import GP
from GPy import likelihoods
from GPy import kern
from GPy import util
from GPy.inference.latent_function_inference.posterior import PosteriorExact as Posterior
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class RiskEmpiricalBayes(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict          = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        
        return input_dict


    def inference(self, 
                  kern, 
                  X, 
                  W, 
                  likelihood, 
                  Y, 
                  mean_function=None, 
                  Y_metadata=None, 
                  K=None, 
                  variance=None, 
                  Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        if K is None:
            K = kern.K(X)
    
        Ky = K.copy()
        
        diag.add(Ky, variance+1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal = 0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
        
        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)
        
        posterior_ = Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K)

        return posterior_, log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}, W_logdet



class risk_based_empirical_bayes_GP(GP):
    """
    Gaussian Process model for Causal Inference

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, 
                 X_list, 
                 Y_list, 
                 W, 
                 kernel=None, 
                 likelihoods_list=None, 
                 name='GPCR', 
                 W_rank=1, 
                 kernel_name='coreg'):

        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny                    = len(Y_list)
        self.opt_trajectory   = []
        self.PEHE_trajectory  = []
        self.MSE_trajectory   = []
        self.treatment_assign = W
        
        self.logdetK          = 0
        
        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1]-1)
            
            kernel = util.multioutput.ICM(input_dim=X.shape[1]-1, num_outputs=Ny, kernel=kernel, W_rank=1,name=kernel_name)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)

        super(CMGP, self).__init__(X, Y, kernel, likelihood, 
                                   inference_method=RiskEmpiricalBayes(), Y_metadata={'output_index':self.output_index})
        
        
        self.X     = Param("input", X)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict, self.logdetK = self.inference_method.inference(self.kern, self.X, self.treatment_assign, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.opt_trajectory.append(self._log_marginal_likelihood)
        
        # Invoke predict and compute our loss function
        W_0_locs     = [k for k in range(len(self.treatment_assign)) if (self.treatment_assign[k]== 0)]
        W_1_locs     = [k for k in range(len(self.treatment_assign)) if (self.treatment_assign[k]== 1)]
        
        X_0_locs     = [self.X[k] for k in W_0_locs]
        X_1_locs     = [self.X[k] for k in W_1_locs]
        
        X_0          = np.array(np.hstack([self.X, np.zeros_like(self.X[:,1].reshape((len(self.X[:,1]),1)))]))
        X_1          = np.array(np.hstack([self.X, np.ones_like(self.X[:,1].reshape((len(self.X[:,1]),1)))]))
        
        X_0_shape    = X_0.shape
        X_1_shape    = X_1.shape
        
        noise_dict_0 = {'output_index': X_0[:,X_0_shape[1]-1].reshape((X_0_shape[0],1)).astype(int)}
        noise_dict_1 = {'output_index': X_1[:,X_1_shape[1]-1].reshape((X_1_shape[0],1)).astype(int)}
        
        mu_0         = np.array(list(self.predict(X_0, Y_metadata = noise_dict_0)[0]))
        mu_1         = np.array(list(self.predict(X_1, Y_metadata = noise_dict_1)[0]))
                
        var_0        = self.predict(X_0, Y_metadata = noise_dict_0)[1]
        var_1        = self.predict(X_1, Y_metadata = noise_dict_1)[1]
        
        Y_est             = np.zeros((len(self.Y),1))
        Var_est           = np.zeros((len(self.Y),1))
        
        Y_est[W_0_locs]   = mu_0[W_0_locs] 
        Y_est[W_1_locs]   = mu_1[W_1_locs] 
        
        Var_est[W_0_locs] = var_1[W_0_locs]
        Var_est[W_1_locs] = var_0[W_1_locs]
        
        regularizer_term  = 0#0.0001
        
        self.PEHE         = np.sqrt(np.mean((Y_est-self.Y)**2 + Var_est)) + regularizer_term*self.logdetK
        self.MSE          = np.sqrt(np.mean((Y_est-self.Y)**2)) + regularizer_term*self.logdetK
        
        self.PEHE_trajectory.append(np.sqrt(np.mean((Y_est-self.Y)**2 + Var_est)))
        self.MSE_trajectory.append(np.sqrt(np.mean((Y_est-self.Y)**2)))
        
        #-----------------------------------------------------------------
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

        
    def log_likelihood_adjusted(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """

        return self.PEHE   
    
    
    def objective_function_adjusted(self): 
        """ 
        The objective function for the given algorithm. 
   
        This function is the true objective, which wants to be minimized.  
        Note that all parameters are already set and in place, so you just need  
        to return the objective function here. 
   
        For probabilistic models this is the negative log_likelihood 
        (including the MAP prior), so we return it here. If your model is not  
        probabilistic, just return your objective here! 
        """
        
        return -float(self.PEHE)


class CMGP:
    """
    An implementation of various Gaussian models for Causal inference building on GPy.
    
    """
#----------------------------------------------------------------
#----------------------------------------------------------------
# This method implements the class constructor, automatically
# invoked for every class instance
#----------------------------------------------------------------
    def __init__(self, mode="CMGP", **kwargs):
        """
        Class constructor. 
        Initialize a GP object for causal inference. 
    
        :mod: ['VirtualTwin'], ['Counterfactual'] or ['Multitask'], default is ['VirtualTwin']
        :dim: the dimension of the input. Default is 1
        :kern: ['Matern'] or ['RBF'], Default is the Radial Basis Kernel
        :mkern: For multitask models, can select from IMC and LMC models, default is IMC  
        """ 
        # %%%%%%%%%%%%%%%%%
        # **Set defaults**
        # %%%%%%%%%%%%%%%%%
        self.mod_list   = ['VirtualTwin','Counterfactual','Multitask']
        self.kern_list  = ['RBF','Matern']
        self.mkern_list = ['ICM','LCM']
        self.mod        = self.mod_list[2]
        self.dim        = 1
        self.kern       = self.kern_list[0]
        self.mkern      = self.mkern_list[0]
        self.mode       = mode   
        self.Bayesian   = True
        self.Confidence = True
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** Read input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        if kwargs.__contains__('mod'):
            self.mod   = kwargs['mod']
        if kwargs.__contains__('dim'):
            self.dim   = kwargs['dim']
        if kwargs.__contains__('kern'):
            self.kern  = kwargs['kern']
        if (kwargs.__contains__('mkern')) and (self.mod == self.mod_list[2]):
            self.mkern = kwargs['mkern']
        # ++++++++++++++++++++++++++++++++++++++++++    
        # ** catch exceptions ** handle wrong inputs 
        # ++++++++++++++++++++++++++++++++++++++++++        
        try:
            if (self.dim < 1) or (type(self.dim) != int):
                raise ValueError('Invalid value for the input dimension! Input dimension has to be a positive integer.')
            if (self.mod not in self.mod_list) or (self.kern not in self.kern_list) or (self.mkern not in self.mkern_list):
                raise ValueError('Invalid input!')
            if (kwargs.__contains__('mkern')) and (self.mod != 'Multitask'):
                raise ValueError('Invalid input! Multitask kernels are valid only for the Multitask mode')
                
        except ValueError:
            if (self.mod not in self.mod_list):
                raise ValueError('Invalid input: The provided mode is undefined for class GaussianProcess_Model.')
            elif (self.kern not in self.kern_list):
                raise ValueError('Invalid input: The provided kernel is undefined for class GaussianProcess_Model.')
            elif (self.mkern not in self.mkern_list):
                raise ValueError('Invalid input: The provided Multitask kernel is undefined for class GaussianProcess_Model.')     
            else:
                raise ValueError('Invalid input for GaussianProcess_Model!')     
        else:
            #*************************************************************************
            # Initialize the kernels and likelihoods depending on the specified model
            #*************************************************************************
            if (self.mod == self.mod_list[0]):
                
                del self.mkern
                
                if(self.kern == self.kern_list[0]): 
                    self.ker0   = GPy.kern.RBF(input_dim = self.dim,ARD=True)
                    self.ker1   = GPy.kern.RBF(input_dim = self.dim,ARD=True)
                else:    
                    self.ker0   = GPy.kern.Matern32(input_dim = self.dim,ARD=True)
                    self.ker1   = GPy.kern.Matern32(input_dim = self.dim,ARD=True)
                    
                self.lik0       = GPy.likelihoods.Gaussian()
                self.lik1       = GPy.likelihoods.Gaussian()
                
            elif (self.mod == self.mod_list[1]):
                del self.mkern
                
                if(self.kern == self.kern_list[0]):
                    self.ker    = GPy.kern.RBF(input_dim = self.dim + 1,ARD=True)
                else:    
                    self.ker    = GPy.kern.Matern32(input_dim = self.dim + 1,ARD=True)
                    
                self.lik        = GPy.likelihoods.Gaussian() 
                
            elif (self.mod == self.mod_list[2]): # edit this later   
                if(self.kern == self.kern_list[0]):
                    base_kernel = GPy.kern.RBF(input_dim = self.dim,ARD=True)
                    self.ker    = GPy.util.multioutput.ICM(self.dim,2,base_kernel,W_rank=1,W=None,kappa=None,name='ICM')
                else:    
                    self.ker    = GPy.kern.Matern32(input_dim = self.dim)
                    
                self.lik        = GPy.likelihoods.Gaussian()
#-----------------------------------------------------------------------------------------------------------
# This method optimizes the model hyperparameters using the factual samples for the treated and control arms
#------------------------------------------------------------------------------------------------------------ 
# ** Note ** all inputs to this method are positional arguments
#---------------------------------------------------------------
    def fit(self, X, Y, W):
        """
        Optimizes the model hyperparameters using the factual samples for the treated and control arms.
        X has to be an N x dim matrix. 
        
        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """  
        # -----------------------------------------------------------------
        # Inputs: X (the features), Y (outcomes), W (treatment assignments)
        # X has to be an N x dim matrix. 
        # -----------------------------------------------------------------
        # Situate the data in a pandas data frame
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Dataset       = pd.DataFrame(X)
        Dataset['Y']  = Y
        Dataset['W']  = W
        Xshape        = np.array(X).shape
        Yshape        = np.array(Y).shape
        W_comp        = Dataset.loc[Dataset['W'] != 1, 'W']
        
        self.X_train  = np.array(X)
        
        if (self.dim > 1):
            Feature_names = list(range(self.dim))
        else:
            Feature_names = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Catch exceptions: handle errors in the input sizes, size mismatches, or undefined
        # treatment assignments
        #----------------------
        #try:
        #    if (Xshape[1] != self.dim) or (Yshape[1] != 1) or (Xshape[0] != Yshape[0]) or (len(W_comp)>0):
        #        raise ValueError('Invalid Inputs!')
        #except ValueError:
        #    if (Xshape[1] != self.dim):
        #        raise ValueError('Invalid input: Dimension of input covariates do not match the model dimensions')
        #    elif (Yshape[1] != 1):
        #        raise ValueError('Invalid input: Outcomes must be formatted in a 1D vector.')
        #    elif (Xshape[0] != Yshape[0]):
        #        raise ValueError('Invalid input: Outcomes and covariates do not have the same number of samples.')     
        #    elif (len(W_comp)>0):
        #        raise ValueError('Invalid input: Treatment assignment vector has non-binary values.')     
        #else:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.mod == self.mod_list[0]):
            Dataset0   = Dataset[Dataset['W']==0].copy()
            Dataset1   = Dataset[Dataset['W']==1].copy()
            # Extract data for the first model
            # `````````````````````````````````````````````````````````````````
            X0         = np.reshape(np.array(Dataset0[Feature_names].copy()),(len(Dataset0),self.dim))
            y0         = np.reshape(np.array(Dataset0['Y'].copy()),(len(Dataset0),1))
            # Extract data for the second model
            # `````````````````````````````````````````````````````````````````
            X1         = np.reshape(np.array(Dataset1[Feature_names].copy()),(len(Dataset1),self.dim))
            y1         = np.reshape(np.array(Dataset1['Y'].copy()),(len(Dataset1),1))
            
            self.model = [GPy.core.GP(X0, y0, kernel = self.ker0, likelihood = self.lik0),
                          GPy.core.GP(X1, y1, kernel = self.ker1, likelihood = self.lik1)]
            self.model[0].optimize(messages=False,max_f_eval = 1000)
            self.model[1].optimize(messages=False,max_f_eval = 1000)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        elif(self.mod == self.mod_list[1]):    
            #X          = np.array(Dataset[[Feature_names,'W']])
            X          = np.array(Dataset[Feature_names+['W']])
            y          = np.reshape(np.array(Dataset['Y']),(len(np.array(Dataset['Y'])),1))
            self.model = GPy.core.GP(X, y, kernel = self.ker, likelihood = self.lik)
            self.model.optimize(messages=False,max_f_eval = 1000)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        elif(self.mod == self.mod_list[2]):
            Dataset0   = Dataset[Dataset['W']==0].copy()
            Dataset1   = Dataset[Dataset['W']==1].copy()
            # Extract data for the first learning task (control population)
            # `````````````````````````````````````````````````````````````````
            X0         = np.reshape(Dataset0[Feature_names].copy(),(len(Dataset0),self.dim))
            y0         = np.reshape(np.array(Dataset0['Y'].copy()),(len(Dataset0),1))
            # Extract data for the second learning task (treated population)
            # `````````````````````````````````````````````````````````````````
            X1         = np.reshape(Dataset1[Feature_names].copy(),(len(Dataset1),self.dim))
            y1         = np.reshape(np.array(Dataset1['Y'].copy()),(len(Dataset1),1))
            # Create an instance of a GPy Coregionalization model
            # `````````````````````````````````````````````````````````````````
            K0         = GPy.kern.Matern32(self.dim, ARD=True) #GPy.kern.RBF(self.dim, ARD=True) 
            K1         = GPy.kern.Matern32(self.dim)#, ARD=True) #GPy.kern.RBF(self.dim, ARD=True) 
            
            K0         = GPy.kern.RBF(self.dim, ARD=True) 
            K1         = GPy.kern.RBF(self.dim, ARD=True) 
            
            #K0         = GPy.kern.MLP(self.dim, ARD=True) 
            #K1         = GPy.kern.MLP(self.dim, ARD=True) 
            
            #K0         = GPy.kern.Spline(input_dim=self.dim) 
            #K1         = GPy.kern.Spline(input_dim=self.dim) 
            
            
            kernel_dict = {"CMGP": GPy.util.multioutput.LCM(input_dim=self.dim,num_outputs=2,kernels_list=[K0, K1]),
                           "NSGP": GPy.util.multioutput.ICM(input_dim=self.dim,num_outputs=2,kernel=K0)}
            
            #self.model = risk_based_empirical_bayes_GP(X_list = [X0,X1], Y_list = [y0,y1], W=W, 
            #                                           kernel = kernel_dict[self.mode])
            
            self.model = GPy.models.GPCoregionalizedRegression(X_list = [X0, X1], Y_list = [y0, y1], 
                                                               kernel = kernel_dict[self.mode])
            
            #self.initialize_hyperparameters(X, Y, W)

            try:

                self.model.optimize('bfgs', max_iters=500)
    
            except np.linalg.LinAlgError as err:

                print("Covariance matrix not invertible.")

            
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
#-----------------------------------------------------------------------------------------------------------
# This method Infers the treatment effect for a certain set of input covariates
#------------------------------------------------------------------------------------------------------------ 
# ** Note ** all inputs to this method are positional arguments
# This method returns the predicted ITE and posterior variance
# but does not store them in self
#---------------------------------------------------------------
    def predict(self, X):
        """
        Infers the treatment effect for a certain set of input covariates. 
        Returns the predicted ITE and posterior variance.
        
        :X: The input covariates at which the outcomes need to be predicted
        """
        # ```````````````````*\ Get input size /*``````````````````````````
        # ---------------------------------------------------------------- 
        Xshape             = np.array(X).shape
        # `````````````````````````````````````````````````````````````````
        if (self.mod == self.mod_list[0]): 

            Y_est_0, var_0 = self.model[0].predict(X)
            Y_est_1, var_1 = self.model[1].predict(X)
            TE_est         = Y_est_1 - Y_est_0
            var_all        = var_0 + var_1
        # ----------------------------------------------------------------    
        elif (self.mod == self.mod_list[1]):
            
            W0             = np.reshape([0]*Xshape[0],(Xshape[0],1))
            W1             = np.reshape([1]*Xshape[0],(Xshape[0],1))
            X_new_0        = np.hstack((np.reshape(np.array(X),(Xshape[0],self.dim)),W0))
            X_new_1        = np.hstack((np.reshape(np.array(X),(Xshape[0],self.dim)),W1))
            Y_est_0, var_0 = self.model.predict(X_new_0)
            Y_est_1, var_1 = self.model.predict(X_new_1)
            TE_est         = Y_est_1 - Y_est_0
            var_all        = var_0 + var_1 # correct this
        # ----------------------------------------------------------------     
        elif (self.mod == self.mod_list[2]):
            
            if self.dim == 1:
                X_           = X[:,None]
                X_0          = np.hstack([X_, np.reshape(np.array([0]*len(X)),(len(X),1))])
                X_1          = np.hstack([X_, np.reshape(np.array([1]*len(X)),(len(X),1))])
                noise_dict_0 = {'output_index': X_0[:,1:].astype(int)}
                noise_dict_1 = {'output_index': X_1[:,1:].astype(int)}
                Y_est_0      = self.model.predict(X_0, Y_metadata = noise_dict_0)[0]
                Y_est_1      = self.model.predict(X_1, Y_metadata = noise_dict_1)[0]
                
            else:
                
                X_0          = np.array(np.hstack([X, np.zeros_like(X[:,1].reshape((len(X[:,1]),1)))]))
                X_1          = np.array(np.hstack([X, np.ones_like(X[:,1].reshape((len(X[:,1]),1)))]))
                X_0_shape    = X_0.shape
                X_1_shape    = X_1.shape
                noise_dict_0 = {'output_index': X_0[:,X_0_shape[1]-1].reshape((X_0_shape[0],1)).astype(int)}
                noise_dict_1 = {'output_index': X_1[:,X_1_shape[1]-1].reshape((X_1_shape[0],1)).astype(int)}
                Y_est_0      = np.array(list(self.model.predict(X_0, Y_metadata = noise_dict_0)[0]))
                Y_est_1      = np.array(list(self.model.predict(X_1, Y_metadata = noise_dict_1)[0]))
                
                var_0        = self.model.predict(X_0, Y_metadata = noise_dict_0)
                var_1        = self.model.predict(X_1, Y_metadata = noise_dict_1)
                
            TE_est       = Y_est_1 - Y_est_0

        return TE_est, Y_est_0, Y_est_1
#-----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------- 
# This method initializes the model's hyper-parameters before passing to the optimizer
# Now working only for the multi-task model
#------------------------------------------------------------------------------------------------------------ 
    def initialize_hyperparameters(self, X, Y, W): 
        """
        Initializes the multi-tasking model's hyper-parameters before passing to the optimizer
        
        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """
        # -----------------------------------------------------------------------------------
        # Output Parameters:
        # -----------------
        # :Ls0, Ls1: length scale vectors for treated and control, dimensions match self.dim
        # :s0, s1: noise variances for the two kernels
        # :a0, a1: diagonal elements of correlation matrix 0 
        # :b0, b1: off-diagonal elements of correlation matrix 1 
        # -----------------------------------------------------------------------------------
        Dataset       = pd.DataFrame(X)
        Dataset['Y']  = Y
        Dataset['W']  = W
        Xshape        = np.array(X).shape
        Yshape        = np.array(Y).shape

        if (self.dim > 1):
            Feature_names = list(range(self.dim))
        else:
            Feature_names = 0
        
        if (self.mod == self.mod_list[2]):
            Dataset0        = Dataset[Dataset['W']==0]
            Dataset1        = Dataset[Dataset['W']==1]
            neigh0          = KNeighborsRegressor(n_neighbors=10)
            neigh1          = KNeighborsRegressor(n_neighbors=10)
            neigh0.fit(Dataset0[Feature_names], Dataset0['Y'])
            neigh1.fit(Dataset1[Feature_names], Dataset1['Y'])
            Dataset['Yk0']  = neigh0.predict(Dataset[Feature_names])
            Dataset['Yk1']  = neigh1.predict(Dataset[Feature_names])
            Dataset0['Yk0'] = Dataset.loc[Dataset['W']==0,'Yk0']
            Dataset0['Yk1'] = Dataset.loc[Dataset['W']==0,'Yk1']
            Dataset1['Yk0'] = Dataset.loc[Dataset['W']==1,'Yk0']
            Dataset1['Yk1'] = Dataset.loc[Dataset['W']==1,'Yk1']
            #`````````````````````````````````````````````````````
            a0       = np.sqrt(np.mean((Dataset0['Y']-np.mean(Dataset0['Y']))**2))
            a1       = np.sqrt(np.mean((Dataset1['Y']-np.mean(Dataset1['Y']))**2))
            b0       = np.mean((Dataset['Yk0']-np.mean(Dataset['Yk0']))*(Dataset['Yk1']-np.mean(Dataset['Yk1'])))/(a0*a1)
            b1       = b0
            s0       = np.sqrt(np.mean((Dataset0['Y']-Dataset0['Yk0'])**2))/a0
            s1       = np.sqrt(np.mean((Dataset1['Y']-Dataset1['Yk1'])**2))/a1
            #`````````````````````````````````````````````````````
            self.model.sum.ICM0.rbf.lengthscale = 10*np.ones(self.dim)
            self.model.sum.ICM1.rbf.lengthscale = 10*np.ones(self.dim)
            
            self.model.sum.ICM0.rbf.variance    = 1
            self.model.sum.ICM1.rbf.variance    = 1
            self.model.sum.ICM0.B.W[0]          = b0 
            self.model.sum.ICM0.B.W[1]          = b0
            
            self.model.sum.ICM1.B.W[0]          = b1 
            self.model.sum.ICM1.B.W[1]          = b1
            
            self.model.sum.ICM0.B.kappa[0]      = a0**2
            self.model.sum.ICM0.B.kappa[1]      = 1e-4
            self.model.sum.ICM1.B.kappa[0]      = 1e-4 
            self.model.sum.ICM1.B.kappa[1]      = a1**2
            
            self.model.mixed_noise.Gaussian_noise_0.variance = s0**2
            self.model.mixed_noise.Gaussian_noise_1.variance = s1**2

