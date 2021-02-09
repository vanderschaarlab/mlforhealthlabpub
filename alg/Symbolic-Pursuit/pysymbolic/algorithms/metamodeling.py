
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from pysymbolic.algorithms.instancewise_feature_selection import *
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import PolynomialFeatures
from pysymbolic.algorithms.keras_predictive_models import *
from pysymbolic.benchmarks.synthetic_datasets import *
from pysymbolic.utilities.instancewise_metrics import *
from pysymbolic.models.special_functions import MeijerG
from mpmath import *
from sympy import *
from sympy.functions import re
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression

# TO DO: re-fit linear coeff for every new func


def eval_one_dimension(Gmodel, X, midpoint=0.5):
        
    """
    Returns a polynomial approximate expression for the Meijer G-function using a Taylor series approximation 
    """ 
    try:
    
        Gmodel.Taylor_poly_ = taylor(Gmodel.math_expr, midpoint, Gmodel.approximation_order)
        coeffp              = Gmodel.Taylor_poly_[::-1]
    
        approx_expr         = [coeffp[k] * ((X - midpoint)**(Gmodel.approximation_order - k)) for k in range(Gmodel.approximation_order)]
        approx_expr         = np.sum(approx_expr, axis=0) + coeffp[-1]
        approx_expr         = [approx_expr[k].real for k in range(len(approx_expr))]  
    
    except:
        
        approx_expr         = [0 for k in range(X.shape[0])] 
    
    return np.float64(approx_expr)


def eval_one_dimension_(Meijer_G_func, dim_name, vals):

    evaluated = [simplify(sympify(str(re(Meijer_G_func.approx_expression()))).subs(dim_name, vals[k])).evalf() for k in range(vals.shape[0])]
    evaluated = np.array(evaluated)
    
    return evaluated 


def Kolmogorov_expression(n_dim, Thetas, Orders, Thetas_out, Orders_out, r=1):
    
    symbols_ = 'X0 '

    for m in range(n_dim - 1):
    
        if m < n_dim - 2:
            
            symbols_ += 'X'+str(m+1)+' '  
        else:
            
            symbols_ += 'X'+str(m+1)
    
    
    dims_       = symbols(symbols_)  
    
    inner_funcs = [MeijerG(theta=Thetas[k], order=Orders[k]) for k in range(n_dim)]
    outer_funcs = MeijerG(theta=Thetas_out[0], order=Orders_out[0])
    
    out_expr_   = 0
    x           = symbols('x')
    
    for v in range(n_dim):

        out_expr_  += sympify(str(re(inner_funcs[v].approx_expression()))).subs(x, dims_[v]) 

    
    out_expr_ = simplify(sympify(str(re(outer_funcs.expression()))).subs(x, out_expr_))

    return out_expr_, dims_


def eval_Kolmogorov_gradient_only(const_, X, theta, Orders, Orders_out, h):
    
    n_dim             = X.shape[1]
    theta             = theta.reshape((-1,1))
    theta_shifts      = np.tile(theta, [1, theta.shape[0]]).T + np.eye(theta.shape[0]) * h
    
    num_inner_params  = Orders[0][2] + Orders[0][3] + 1
    num_outer_params  = Orders_out[0][2] + Orders_out[0][3] + 1
    tot_num_params    = num_inner_params * n_dim + num_outer_params
    
    f_vals_h          = []
    
    f_val_0, f_inners = eval_Kolmogorov_expression_only(const_, X, theta, Orders, Orders_out)
     
    for k in range(tot_num_params):
        
        f_inners_h                   = deepcopy(f_inners)
        inner_funcs_h, outer_funcs_h = get_inner_outer_functions(n_dim, theta_shifts[k, :], Orders, Orders_out)
        idx                          = int(np.floor(k/num_inner_params))
        
        if k < num_inner_params * n_dim:

            f_inners_h[idx, :]       = eval_inner_functions_only(X[:, idx].reshape((-1,1)), [inner_funcs_h[idx]])

        f_vals_h.append(eval_Kolmogorov_expression_(const_, X, f_inners_h, outer_funcs_h))
                
    f_grad = (np.concatenate(f_vals_h, axis=1) - np.tile(f_val_0 , [1, tot_num_params]))/h    
        

    return f_val_0, f_grad  



def eval_Kolmogorov_expression_only(const_, X, theta, Orders, Orders_out):

    x     = symbols('x')
    n_dim = X.shape[1]
    
    inner_funcs, outer_funcs = get_inner_outer_functions(n_dim, theta, Orders, Orders_out) 
    f_inners                 = eval_inner_functions_only(X, inner_funcs) 
    f_val                    = eval_Kolmogorov_expression_(const_, X, f_inners, outer_funcs)
       
    return f_val, f_inners



def evaluate_Kolmogorov(out_expr, dims_, x_in):
    
    n_dim    = len(dims_)
    val_expr = sympify(str(re(out_expr))).subs(dims_[0], x_in[0])

    for k in range(n_dim-1):
    
        val_expr = sympify(str(re(val_expr))).subs(dims_[k+1], x_in[k+1])

    return val_expr.evalf()


def get_theta_parameters(Thetas, Orders_in, Orders_out, n_dim):
    
    Theta_inner      = []
    
    theta_sizes_in   = [Orders_in[k][2] + Orders_in[k][3] for k in range(len(Orders_in))]
    pivot_           = 0
    
    for k in range(len(Orders_in)):
        
        Theta_inner.append(Thetas[pivot_ : pivot_ + theta_sizes_in[k] + 1])
        pivot_      += theta_sizes_in[k] + 1
    
    Theta_outer      = Thetas[pivot_ : ] 

    return Theta_inner, [Theta_outer]



def Optimize(Loss, theta_0):
    
    opt       = minimize(Loss, theta_0, method='CG',  
                         options={'xtol': 1e-2, 'maxiter':1, 'eps': 0.5, 'disp': True})
    Loss_     = opt.fun
    theta_opt = opt.x
    
    return theta_opt, Loss_ 


def combine_inner_outer(const_, _outer_funcs, x, _inner_funcs):
    
    #return const_/(const_ + eval_one_dimension(_outer_funcs, _inner_funcs))
    return const_/(const_ + np.exp(-1*_inner_funcs))

def stretch_G_params(G):
    
    stretched_params_ = np.concatenate([np.concatenate(G.a_p).reshape((-1,1)), 
                                        np.concatenate(G.b_q).reshape((-1,1)), 
                                        np.array(G._const).reshape((-1,1))]).reshape(-1,)
    
    return stretched_params_


def assign_G_params(G, new_params):
    
    G_new     = deepcopy(G)
    
    a_p_      = new_params[: len(G.a_p) + 1] 
    b_q_      = new_params[len(G.a_p) + 1: len(G.a_p) + len(G.b_q)]
    
    G_new.a_p = [a_p_[:G_new.order[1]], a_p_[G_new.order[1]:]]
    G_new.b_q = [b_q_[:G_new.order[0]], b_q_[G_new.order[0]:]]
    
    G_new._const  = new_params[-1]
    
    return G_new
    
def MeijerG_finite_difference(G, h):
    
    num_params         = G.order[2] + G.order[3] + 1
    G_h                = []
    
    stretched_params_  = stretch_G_params(G)
    finite_diff_params = np.tile(stretched_params_, [num_params, 1]) + np.eye(num_params) * h
    
    for k in range(num_params):
        
        G_h.append(assign_G_params(G, finite_diff_params[k, :]))
    
    return G_h    

def get_inner_outer_functions(n_dim, theta, Orders, Orders_out):
    
    theta         = theta.reshape((-1,))
    
    Thetas, Thetas_out = get_theta_parameters(theta, Orders, Orders_out, n_dim=n_dim)
    
    inner_funcs   = [MeijerG(theta=Thetas[k], order=Orders[k]) for k in range(n_dim)]
    outer_funcs   = MeijerG(theta=Thetas_out[0], order=Orders_out[0])
    
    return inner_funcs, outer_funcs


def eval_inner_functions_only(X, inner_funcs):
    
    x             = symbols('x')
    n_dim         = X.shape[1]
    f_inners_     = np.real(np.array([eval_one_dimension(inner_funcs[u], X[:, u]) for u in range(len(inner_funcs))]))
    
    return f_inners_

def eval_Kolmogorov_expression_(const_, X, f_inners_, outer_funcs):
    
    x             = symbols('x')
    n_dim         = X.shape[1]
    
    f_inners      = np.sum(f_inners_, axis=0)
    f_val         = combine_inner_outer(const_, outer_funcs, x, f_inners).reshape((-1,1))
    
    return f_val


def exact_Kolmogorov_expression(n_dim, theta, Orders, Orders_out, const_, r=1):
    
    Thetas, Thetas_out = get_theta_parameters(theta.reshape((-1,)), Orders, Orders_out, n_dim)
    
    symbols_ = 'X0 '

    for m in range(n_dim - 1):
    
        if m < n_dim - 2:
            symbols_ += 'X'+str(m+1)+' '  
        else:
            symbols_ += 'X'+str(m+1)
    
    dims_       = symbols(symbols_)  
    
    inner_funcs = [MeijerG(theta=Thetas[k], order=Orders[k]) for k in range(n_dim)]
    outer_funcs = MeijerG(theta=Thetas_out[0], order=Orders_out[0])
    
    out_expr_   = 0
    x           = symbols('x')
    
    for v in range(n_dim):
        
        out_expr_  += sympify(str(re(inner_funcs[v].expression()))).subs(x, dims_[v]) 

    
    out_expr_  = simplify(sympify(str(re(outer_funcs.expression()))).subs(x, out_expr_))
    
    final_expr = simplify(sympify(str(const_/(const_ + out_expr_))))

    return final_expr, dims_


class Symbolic_Metamodel:
    
    def __init__(self, 
                 n_dim=10, 
                 batch_size=100, 
                 num_iter=30,
                 learning_rate= 1e-3,
                 feature_types=None,
                 **kwargs):
        
        self.n_dim         = n_dim
        self.batch_size    = batch_size
        self.num_iter      = num_iter
        self.learning_rate = learning_rate
        self.exact_grad    = False
        self.epsilon       = 1e-6
        self.feature_types = feature_types

        
    def fit(self, pred_model, x_train):
        
        self.pred_model = pred_model
        self.x_train    = x_train
        self.n_dim      = x_train.shape[1]
        
        self.initialize_thetas()
        
        thetas_sgd, Losses_   = self.SGD_optimizer()
        self.thetas_opt       = thetas_sgd[-1]
        
        self.metamodel, dims_ = self.get_exact_Kolmogorov_expression(self.thetas_opt)
        self.exact_pred_expr  = pred_model 
        
    
    def get_exact_Kolmogorov_expression(self, theta):
    
        Thetas, Thetas_out        = get_theta_parameters(theta.reshape((-1,)), self.Orders_in, self.Orders_out, self.n_dim)
        Thetas_in_0, Thetas_out_0 = get_theta_parameters(self.theta_0.reshape((-1,)), self.Orders_in, self.Orders_out, self.n_dim)  
    
        symbols_ = 'X0 '

        for m in range(self.single_dim - 1):
    
            if m < self.single_dim - 2:
            
                symbols_ += 'X' + str(m + 1) + ' '  
            
            else:
            
                symbols_ += 'X' + str(m + 1)
    
        dims_       = symbols(symbols_)  
    
        inner_funcs = [MeijerG(theta=Thetas[k], order=self.Orders_in[k]) for k in range(self.n_dim)]
        outer_funcs = MeijerG(theta=Thetas_out[0], order=self.Orders_out[0])

        inner_fun_0 = [MeijerG(theta=Thetas_in_0[k], order=self.Orders_in[k]) for k in range(self.n_dim)]
    
        out_expr_   = 0
        x           = symbols('x')
    
        for v in range(self.n_dim):
            
            if v < self.single_dim:

                if v not in self.zero_locs:

                    out_expr_  += sympify(str(re(inner_funcs[v].expression()))).subs(x, dims_[v])

            else:

                if v not in self.zero_locs:
                
                    dim_0       = self.dim_combins[self.single_dim + (v-self.single_dim)][0]
                    dim_1       = self.dim_combins[self.single_dim + (v-self.single_dim)][1]

                    out_expr_  += sympify(str(re(inner_funcs[v].expression()))).subs(x, dims_[dim_0] * dims_[dim_1])

        out_expr_  = simplify(sympify(str(re(outer_funcs.expression()))).subs(x, out_expr_))           
    
        final_expr = simplify(sympify(str(self.init_scale/(self.init_scale + out_expr_))))

        return final_expr, dims_    
        
        
    def initialize_thetas(self):
        
        self.poly = PolynomialFeatures(interaction_only=True, include_bias = False)
        self.poly.fit(self.x_train)

        self.origin_x_train  = self.x_train
        self.x_train         = self.poly.transform(self.x_train)
        self.single_dim      = self.n_dim
        self.n_dim           = self.x_train.shape[1]
        self.dim_combins     = list(self.poly._combinations(self.single_dim, degree=2, interaction_only=True, include_bias = False))
        
        initializer_model = LogisticRegression()
        binarized_thresh  = 0.5
        
        if hasattr(self.pred_model, "predict_proba"):
            
            initializer_model.fit(self.x_train, (self.pred_model.predict_proba(self.origin_x_train)[:,1] > binarized_thresh) * 1)
            
        else:
            
            initializer_model.fit(self.x_train, (self.pred_model.predict(self.origin_x_train) > binarized_thresh) * 1)
         
        
        self.init_coeff = initializer_model.coef_[0] + self.epsilon
        self.init_scale = np.exp(initializer_model.intercept_)
        self.zero_locs  = list(np.where(np.abs(self.init_coeff) <= self.epsilon)[0])
    
        Thetas_in_0, Thetas_out_0 = self.initialize_hyperparameters()
        self.theta_0              = np.hstack((Thetas_in_0, Thetas_out_0)).reshape((-1,))
    
    
    def initialize_hyperparameters(self):
        
        self.Orders_in  = [[0,1,3,1]] * self.n_dim 
        self.Orders_out = [[1,0,0,1]]
    
        Thetas       = [np.array([2.0, 2.0, 2.0, 1.0, self.init_coeff[k]]) for k in range(self.n_dim)] 
        Thetas_out   = [np.array([0.0,1.0])]
    
        Thetas_in_0  = np.array(Thetas).reshape((1,-1))
        Thetas_out_0 = np.array(Thetas_out).reshape((1,-1))
    
        return Thetas_in_0, Thetas_out_0

    
    def SGD_optimizer(self):
        
        theta_     = self.theta_0.reshape((-1,1)) 
        losses_    = []
        thetas_opt = []
        
        beta_1     = 0.9 
        beta_2     = 0.999 

        eps_stable = 1e-8 
        m          = 0
        v          = 0
        step_size  = 0.001
        
        for _ in range(self.num_iter):
            
            start_time       = time.time()
            loss_, loss_grad = self.Loss_grad(theta_)
            
            #
            loss_grad[self.single_dim : loss_grad.shape[0] - 2, 0] = 0  
            #

            if _ <= self.num_iter - 1:
                
                theta_  = theta_ - self.learning_rate * loss_grad
            
            m     = beta_1 * m + (1 - beta_1) * loss_grad
            v     = beta_2 * v + (1 - beta_2) * np.power(loss_grad, 2)
    
            m_hat = m / (1 - np.power(beta_1, _))
            v_hat = v / (1 - np.power(beta_2, _))

            #theta_ = theta_ - (step_size * m_hat / (np.sqrt(v_hat) + eps_stable)) 
           
            print("-- Search epoch: %s --- Loss: %0.5f --- Run time: %0.2f seconds ---" % (_, loss_, time.time() - start_time)) 
            
            losses_.append(loss_)
            thetas_opt.append(theta_)
        
        return thetas_opt, losses_    
        
    
    def Loss_grad(self, theta):

        subsamples_    = np.random.choice(list(range(self.x_train.shape[0])), 
                                          size=self.batch_size, 
                                          replace=False)
        
        x_             = self.x_train[subsamples_, :]
        x_original     = self.origin_x_train[subsamples_, :]
        
        f_est, f_grad  = eval_Kolmogorov_gradient_only(self.init_scale,
                                                       x_, 
                                                       theta.reshape((-1,)),
                                                       self.Orders_in,
                                                       self.Orders_out,
                                                       h=0.01)
        
        f_est          = f_est.reshape((-1,1))
        
        if hasattr(self.pred_model, "predict_proba"):
            
            #y_true = ((self.pred_model.predict_proba(x_original)[:,1] > 0.5)*1).reshape((-1,1))
            f_true = self.pred_model.predict_proba(x_original)[:,1].reshape((-1,1))
            
        else:
            
            #y_true = ((self.pred_model.predict(x_original)[:,1] > 0.5)*1).reshape((-1,1))
            f_true = self.pred_model.predict(x_original)[:,1].reshape((-1,1))
        
        
        loss_type = 'mean_square_error'
        
        if loss_type == 'mean_square_error':
            
            loss_per_param = np.tile(-1 * 2 * (f_true - f_est), [1, f_grad.shape[1]])
            loss_grad      = np.mean(loss_per_param * f_grad, axis=0).reshape((-1,1))
            loss_          = np.mean((f_true - f_est)**2)
            
        elif loss_type == 'cross_entropy':
            
            loss_per_param = np.tile(- (f_true - f_est)/((1 - f_est) * f_est), [1, f_grad.shape[1]])
            loss_grad      = np.mean(loss_per_param * f_grad, axis=0).reshape((-1,1))
            loss_          = np.mean(-1 * (np.multiply(f_true, np.log(f_est)) + np.multiply((1 - f_true), np.log(1 - f_est)))) 
            
        #loss_c     = roc_auc_score(y_true, f_est)
        #print("AUC", loss_c)
        
        return loss_, loss_grad    

    
    def evaluate(self, x_in):

        y_est, _  = eval_Kolmogorov_gradient_only(self.init_scale,
                                                  self.poly.transform(x_in), 
                                                  self.thetas_opt,
                                                  self.Orders_in,
                                                  self.Orders_out,
                                                  h=0.01)
        return y_est
    
    def get_gradient(self, x_in):
        
        grad_ = []
        h     = 0.01
        x_in  = x_in.reshape((1,-1))
        
        if self.exact_grad:
            
            gradients_ = []
            
            for var in vars_:
                
                gradients_.append(diff(sym_meta_mod, var))

                for k in range(x_test.shape[0]):
                    
                    grad_.append([gradients_[v].subs(vars_[0], x_test[k,0]).subs(vars_[1], x_test[k,1]).subs(vars_[2], x_test[k,2]).subs(vars_[3], x_test[k,3]).subs(vars_[4], x_test[k,4]).subs(vars_[5], x_test[k,5]).subs(vars_[6], x_test[k,6]).subs(vars_[7], x_test[k,7]).subs(vars_[8], x_test[k,8]).subs(vars_[9], x_test[k,9]) for v in range(len(gradients_))])
            
        else:
            
            for u in range(x_in.shape[1]):
        
                x_in_h       = deepcopy(x_in)

                if self.feature_types is not None:
                
                    x_in_h[0, u] = ((self.feature_types[u]=='c')*1) * (x_in_h[0, u] + h) + ((self.feature_types[u]=='b')*1) * ((x_in_h[0, u]==0)*1)     

                else:

                    x_in_h[0, u] = x_in_h[0, u] + h
                    
        
                grad_.append(np.abs(self.exact_pred_expr.predict_proba(x_in_h)[:,1] - self.exact_pred_expr.predict_proba(x_in)[:,1])/h)
        
    
        return np.array(grad_)
        
    def get_instancewise_scores(self, x_in):
        
        scores = []
        
        for v in range(x_in.shape[0]): 
            
            scores.append(self.get_gradient(x_in[v, :]))
        
        
        return scores

