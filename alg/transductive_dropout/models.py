import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats as ss
from autograd.misc import flatten
from autograd.misc.optimizers import adam
from tqdm import trange, tqdm

import matplotlib.pyplot as plt
from autograd import grad

def sigmoid(x):
  '''
  Standard sigmoid function, should be a little more numerically 
  stable than completely naive version.
  '''
  return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                            np.exp(x) / (1 + np.exp(x)))

def concrete_s(p=0.5,n=1,t=0.1):
    '''
    Sample from the concrete distribution
    p - Probability of ''success''
    n - Number of samples
    t - ''Temperature'' of the distribution, controls the level of 
        smoothing basically
    '''
    if p >= 1:
        return np.ones(n)
    u = npr.uniform(size = n)
    u2 = np.log(p) - np.log(1-p) + np.log(u) - np.log(1-u)
    sample = sigmoid(u2/t)
  
    return sample

class mc_dropout():
    '''
    Implement MC Dropout MLP with tanh activation.
    Initial arguments:
    layers - list of number of neurons per layer
    d_rate - list of dropout rate per layer, first layer fixed rate 1.0
             so len(d_rate) = len(layers) - 1. 
             NOTE: d_rate = 1.0 means NO dropout as opposed to all 
             neurons dropped.
    
    Trained under squared error loss using Adam.
    
    Use:
    ex = mc_dropout(layers,d_rate)
    ex.train(train_x,train_y,1000)
    ex.forward_pass(test_x)
    '''
    def __init__(self, layers, d_rate):
        
        self.layers = layers
        self.d_rate = d_rate
        
        self.params = self.init_random_params(layer_sizes = layers)
    
    @staticmethod
    def init_random_params(layer_sizes, scale=0.1, rs=npr.RandomState(0)):

        return [(rs.randn(insize, outsize) * scale,   # weight matrix
                 rs.randn(outsize) * scale)           # bias vector
                for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    def forward_pass(self,params,inputs):
        # Simple MLP, repeated matrix multiplication plus bias
        # Applies tanh activation followed by bernoulli dropout mask
        
        for (W, b), d_rate in zip(params,self.d_rate):
            
            mask = npr.binomial(1, d_rate, size = len(b))
            
            outputs = (np.dot(inputs, W) + b)
            inputs = np.tanh(outputs) * mask
            
        return outputs
    
    def objective_train(self,params,t):
        # Squared error loss
        
        return np.sum( (self.forward_pass(params,self.train_x) - self.train_y)**2 )\
                #+ np.sum(flatten(params)[0]**2)

    def callback(self,params, t, g):        
        # Record training loss and increment progress bar
        
        self.train_loss.append(self.objective_train(params,t))
        self.pbar.update(1)
        
    def train(self, train_x, train_y,iters):
        
        self.train_x = train_x
        self.train_y = train_y 
        
        self.train_loss = []
        self.pbar = tqdm(total=iters, desc = 'Optimising parameters')
        
        init_params = self.params
        # Optimisation via Autograd's implementation of Adam
        optimised_params = adam(grad(self.objective_train), init_params,
                            step_size=0.01, num_iters=iters,callback=self.callback)
            
        self.params = optimised_params
        self.pbar.close()

        # Plot evolution of training loss
        means = []
        for i in range(iters):
            if i == 0:
                means.append(self.train_loss[i])
            else:
                mean = ( (means[(i-1)] * i) + self.train_loss[i] ) / (i+1)
                means.append(mean)
        
        plt.plot(self.train_loss, label='SE Loss')
        plt.plot(means,c='r', linewidth=3, label='Averge SE Loss')
        plt.title("Training Error")
        plt.legend()
        plt.show()

        return

class mc_dropout_concrete(mc_dropout):

    def forward_pass(self,params,inputs):
      # Simple MLP, repeated matrix multiplication plus bias
      # Applies tanh activation followed by bernoulli dropout mask
        
        for (W, b), d_rate in zip(params,self.d_rate):
            
            mask = concrete_s(p = d_rate, n = len(b))
            
            outputs = (np.dot(inputs, W) + b)
            inputs = np.tanh(outputs) * mask
            
        return outputs
    

class dropout_regularised():

    def __init__(self, layers):
        
        self.layers = layers
        self.d_rate = [0.0 for _ in range(len(layers)-1)]
        
        self.params = (self.init_random_params(layer_sizes = layers),
                       self.d_rate)
        self.rate_change = []
        self.params_history = []
    
    @staticmethod
    def init_random_params(layer_sizes, scale=0.1, rs=npr.RandomState(0)):

        return [(rs.randn(insize, outsize) * scale,   # weight matrix
                 rs.randn(outsize) * scale)           # bias vector
                for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


    
    def forward_pass(self,params,inputs):
        # Simple MLP, repeated matrix multiplication plus bias
        # Applies tanh activation followed by bernoulli dropout mask
        pars,d_rates = params
        for (W, b), d_rate in zip(pars,d_rates):
            
            mask = concrete_s(p = sigmoid(d_rate), n = len(b))
            
            outputs = (np.dot(inputs, W) + b)
            inputs = np.tanh(outputs) * mask
            
        return outputs
    
    def objective_train(self,params,t):
        # Squared error loss
        pars, d_rates = params 

        drop_reg = 0
        for i in range(len(d_rates)):
            p = sigmoid(d_rates[i])
            ent = (-p * np.log(p)) - ((1-p)*np.log(1-p))
            drop_reg += self.layers[i+1]*ent 

        y_hats = [self.forward_pass(params,self.train_x) for _ in range(10)]
        y_hats = np.array(y_hats)
        y_hats = np.reshape(y_hats,(10,len(self.train_x)))

        mean = y_hats.mean(axis=0)
        std = y_hats.std(axis=0)  
        sqerr = np.sum((mean.reshape((len(mean),1)) - self.train_y)**2)

        y_hat_q = [self.forward_pass(params,self.test_x) for _ in range(10)]
        y_hat_q = np.array(y_hat_q)
        y_hat_q = np.reshape(y_hat_q,(10,len(self.test_x)))
        std_q = y_hat_q.std(axis=0)

        sqn = 1 - (1/(1+std_q))
        spn = 1 - (1/(1+std))
        ce_sqn = np.sum(np.log(1-sqn))
        ce_spn = np.sum(np.log(spn))
        ce_loss = 0.25*ce_sqn + 0.11*ce_spn

        return 3.0*sqerr + ce_loss - 0.02*drop_reg
                #+ np.sum(flatten(params)[0]**2)

    def callback(self,params, t, g):        
        # Record training loss and increment progress bar
        
        _, d = params
        self.rate_change.append(d)
        self.train_loss.append(self.objective_train(params,t))
        self.pbar.update(1)
        
        if t%10 == 0:
            self.params_history.append(params)
        
    def train(self, train_x, train_y, test_x, iters):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x 
        
        self.train_loss = []
        self.pbar = tqdm(total=iters, desc = 'Optimising parameters')
        
        init_params = self.params

        optimised_params = adam(grad(self.objective_train), init_params,
                            step_size=0.01, num_iters=iters,callback=self.callback)
            
        self.params = optimised_params
        _, self.d_rate = optimised_params
        self.pbar.close()

        # Plot evolution of training loss
        means = []
        for i in range(iters):
            if i == 0:
                means.append(self.train_loss[i])
            else:
                mean = ( (means[(i-1)] * i) + self.train_loss[i] ) / (i+1)
                means.append(mean)
        
        plt.plot(self.train_loss, label='SE Loss')
        plt.plot(means,c='r', linewidth=3, label='Averge SE Loss')
        plt.title("Training Error")
        plt.legend()
        plt.show()

        return
    
    
    

class transductive(dropout_regularised):

    def __init__(self, layers, d_units = 16):
        
        self.layers = layers
        self.d_units = d_units
        self.d_rate = [0.0 for _ in range(len(layers)-1)]

        self.params = self.init_random_params(layer_sizes = layers, 
                                              d_size = d_units)
        self.rate_change = []
        self.params_history = []
        

    @staticmethod
    def init_random_params(layer_sizes, d_size, 
                           scale=0.1, rs=npr.RandomState(0)):

        pars = [(rs.randn(insize, outsize) * scale,   # weight matrix
                 rs.randn(outsize) * scale)           # bias vector
                for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]
        d_pars = [[rs.randn(1,d_size)*scale,
                   rs.randn(d_size)*scale],
                  [rs.randn(d_size,(len(layer_sizes)-1))*scale, 
                   rs.randn(len(layer_sizes)-1)*scale]]
        
        return [pars,d_pars]

    def objective_train(self,params,t):
        
        pars, d_pars = params 
  
        y_hats = [self.forward_pass2(params,self.train_x) for _ in range(10)]
        y_hats = np.array(y_hats)
        y_hats = np.reshape(y_hats,(10,len(self.train_x)))

        mean = y_hats.mean(axis=0)
        std = y_hats.std(axis=0)  
        sqerr = np.sum((mean.reshape((len(mean),1)) - self.train_y)**2)

        y_hat_q = [self.forward_pass2(params,self.test_x) for _ in range(10)]
        y_hat_q = np.array(y_hat_q)
        y_hat_q = np.reshape(y_hat_q,(10,len(self.test_x)))
        std_q = y_hat_q.std(axis=0)

        sqn = 1 - (1/(1+std_q))
        spn = 1 - (1/(1+std))
        ce_sqn = np.sum(np.log(1-sqn))
        ce_spn = np.sum(np.log(spn))
        ce_loss = 1.0*ce_sqn + 0.1*ce_spn

        return 5.0*sqerr + 0.05*ce_loss + 0.001*np.sum(flatten(params)[0]**2)

    def forward_pass(self,params,x):
        # Simple MLP, repeated matrix multiplication plus bias
        # Applies tanh activation followed by concrete dropout mask
        pars,d_pars = params

        d_rates = self.get_rates(d_pars,x)

        for (W, b), d_rate in zip(pars,d_rates):
            
            mask = concrete_s(p = d_rate, n = len(b))
            
            output = (np.dot(x, W) + b)
            x = np.tanh(output) * mask

        
        return output

    def forward_pass2(self,params,inputs):
        '''
        Given the dropout rate dependence on x, vectorising this model is non-trivial.
        This implementation is not optimised for speed (at all) but gives a clearer 
        view of what's going on.
        '''
        return np.array([self.forward_pass(params,x) for x in inputs])

    def get_rates(self,d_pars,x):
        
        for (W,b) in d_pars:
            output = (np.dot(x,W) + b)
            x = sigmoid(output)
        return sigmoid(output)