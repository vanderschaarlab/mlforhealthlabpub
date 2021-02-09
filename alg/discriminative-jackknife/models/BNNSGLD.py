
# coding: utf-8

# In[11]:
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer


# In[12]:

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


# In[13]:

class Langevin_SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(Langevin_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Langevin_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if len(p.shape) == 1 and p.shape[0] == 1:
                    p.data.add_(-group['lr'], d_p)
                    
                else:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    unit_noise = Variable(p.data.new(p.size()).normal_())

                    p.data.add_(-group['lr'], 0.5*d_p + unit_noise/group['lr']**0.5)

        return loss


# In[14]:

def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    
    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()


# In[15]:

class Langevin_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Langevin_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
    def forward(self, x):
        
        return torch.mm(x, self.weights) + self.biases
    


# In[19]:

class Langevin_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(Langevin_Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # network with two hidden and one output layer
        self.layer1 = Langevin_Layer(input_dim, no_units)
        self.layer2 = Langevin_Layer(no_units, no_units)
        self.layer3 = Langevin_Layer(no_units, output_dim)
        
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.tensor([init_log_noise]).float())

    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        
        x = self.layer1(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        
        return x


# In[22]:

class Langevin_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, init_log_noise, weight_decay):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        
        self.network = Langevin_Model(input_dim = input_dim, output_dim = output_dim,
                                      no_units = no_units, init_log_noise = init_log_noise)

        
        self.optimizer = Langevin_SGD(self.network.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        
        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)
        
        loss.backward()
        self.optimizer.step()

        return loss


# In[33]:

def BNN_sgld(x_train, y_train, x_test, input_dim=1):
    
    best_net, best_loss        = None, float('inf')
    num_nets, nets, losses     = 50, [], []
    mix_epochs, burnin_epochs  = 100, 3000
    num_epochs                 = mix_epochs*num_nets + burnin_epochs
    
    net = Langevin_Wrapper(input_dim=input_dim, output_dim=1, no_units=100, learn_rate=1e-4,
                       batch_size=len(x_train), no_batches=1, init_log_noise=0, weight_decay=1)
    
    
    for i in range(num_epochs):
    
        loss = net.fit(x_train, y_train)
    
        if i % mix_epochs == 0 and i > burnin_epochs: nets.append(copy.deepcopy(net.network))
        
        
    samples = []
    noises = []

    for network in nets:
        preds = network.forward(x_test.float()).float().data.numpy()
        samples.append(preds)
        noises.append(torch.exp(network.log_noise).detach().numpy())
    
    samples = np.array(samples)
    noises = np.array(noises).reshape(-1)
    means = (samples.mean(axis = 0)).reshape(-1)

    aleatoric = noises.mean()
    epistemic = (samples.var(axis = 0)**0.5).reshape(-1)
    total_unc = (aleatoric**2 + epistemic**2)**0.5    
    
    return means, total_unc 
    




