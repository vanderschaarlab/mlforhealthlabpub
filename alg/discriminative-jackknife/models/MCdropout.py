
# Code from MC dropout
# This code is from
# ---------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import KFold


def to_variable(var=(), cuda=True, volatile=False):
    
    out = []
    
    for v in var:
        
        if isinstance(v, np.ndarray):
            
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not isinstance(v, Variable):
            
            v = Variable(v, volatile=volatile)

        out.append(v)
        
    return out

def log_gaussian_loss(output, target, sigma, no_dim):
    
    exponent  = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    
    prior_loglik   = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik    = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    
    def __init__(self, mu, sigma):
        
        self.mu    = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        
        exponent  = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()


class MC_Dropout_Model(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        
        super(MC_Dropout_Model, self).__init__()
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.drop_prob  = drop_prob
        
        # network with two hidden and one output layer
        self.layer1     = nn.Linear(input_dim, num_units)
        self.layer2     = nn.Linear(num_units, 2*output_dim)
        
        self.activation = nn.ReLU(inplace = True)

    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        
        x = self.layer1(x)
        x = self.activation(x)
        
        x = F.dropout(x, p=self.drop_prob, training=True)
        
        x = self.layer2(x)
        
        return x


class MC_Dropout_Wrapper:
    def __init__(self, network, learn_rate, batch_size, weight_decay):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        
        self.network = network
        #self.network.cuda()
        
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        
        output = self.network(x)
        loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1)
        
        loss.backward()
        self.optimizer.step()

        return loss
    
    def get_loss_and_rmse(self, x, y, num_samples):
        x, y = to_variable(var=(x, y), cuda=True)
        
        means, stds = [], []
        for i in range(num_samples):
            output = self.network(x)
            means.append(output[:, :1])
            stds.append(output[:, 1:].exp())
        
        means, stds = torch.cat(means, dim=1), torch.cat(stds, dim=1)
        mean = means.mean(dim=-1)[:, None]
        std = ((means.var(dim=-1) + stds.mean(dim=-1)**2)**0.5)[:, None]
        loss = self.loss_func(mean, y, std, 1)
        
        rmse = ((mean - y)**2).mean()**0.5

        return loss.detach().cpu(), rmse.detach().cpu()


def get_MC_dropout(X, Y, X1, input_dim=1, num_epochs=500, learn_rate=1e-5, verbose=False):

    batch_size = len(X)

    net = MC_Dropout_Wrapper(network=MC_Dropout_Model(input_dim=input_dim, output_dim=1, num_units=100, drop_prob=0.5),
                         learn_rate=learn_rate, batch_size=batch_size, weight_decay=1e-2)

    fit_loss_train = np.zeros(num_epochs)
    best_net, best_loss = None, float('inf')
    nets, losses = [], []

    for i in range(num_epochs):
    
        loss = net.fit(X, Y)
    
        if (i % 200 == 0) and verbose:
            print('Epoch: %4d, Train loss = %7.3f' % (i, loss.cpu().data.numpy()/batch_size))        


    samples = []
    noises  = []

    for i in range(1000):
    
        preds = net.network.forward(torch.tensor(X1).float()).cpu().data.numpy()
    
        samples.append(preds[:, 0])
        noises.append(np.exp(preds[:, 1]))
    
    samples   = np.array(samples)
    noises    = np.array(noises)
    means     = (samples.mean(axis = 0)).reshape(-1)

    aleatoric = (noises**2).mean(axis = 0)**0.5
    epistemic = (samples.var(axis = 0)**0.5).reshape(-1)
    total_unc = (aleatoric**2 + epistemic**2)**0.5
    
    return means, total_unc  



class MC_Dropout_Model_UCI(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(MC_Dropout_Model_UCI, self).__init__()
        

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        
        # network with two hidden and one output layer
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer2 = nn.Linear(num_units, num_units)
        self.layer3 = nn.Linear(num_units, 2*output_dim)
        
        self.activation = nn.ReLU(inplace = True)

    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        
        x = self.layer1(x)
        x = self.activation(x)
        
        x = F.dropout(x, p=self.drop_prob, training=True)
        
        x = self.layer2(x)
        x = self.activation(x)
        
        x = F.dropout(x, p=self.drop_prob, training=True)
        
        x = self.layer3(x)
        
        return x


def train_mc_dropout(data, drop_prob, n_splits, num_epochs, num_units, learn_rate, weight_decay, log_every, num_samples):

    batch_size = data.shape[0]
    kf = KFold(n_splits=n_splits)
    in_dim = data.shape[1] - 1
    train_logliks, test_logliks = [], []
    train_rmses, test_rmses = [], []

    for j, idx in enumerate(kf.split(data)):
        #print('FOLD %d:' % j)
        train_index, test_index = idx

        x_train, y_train = data[train_index, :in_dim], data[train_index, in_dim:]
        x_test, y_test = data[test_index, :in_dim], data[test_index, in_dim:]

        x_means, x_stds = x_train.mean(axis = 0), x_train.var(axis = 0)**0.5
        y_means, y_stds = y_train.mean(axis = 0), y_train.var(axis = 0)**0.5

        x_train = (x_train - x_means)/x_stds
        y_train = (y_train - y_means)/y_stds

        x_test = (x_test - x_means)/x_stds
        y_test = (y_test - y_means)/y_stds

        net = MC_Dropout_Wrapper(network=MC_Dropout_Model_UCI(input_dim=in_dim, output_dim=1, num_units=num_units, drop_prob=drop_prob),
                                 learn_rate=learn_rate, batch_size=batch_size, weight_decay=weight_decay)

        losses = []
        fit_loss_train = np.zeros(num_epochs)

        for i in range(num_epochs):
            loss = net.fit(x_train, y_train)
                
            if i % log_every == 0 or i == num_epochs - 1:
                test_loss, rmse = net.get_loss_and_rmse(x_test, y_test, num_samples=num_samples)
                test_loss, rmse = test_loss.cpu().data.numpy(), rmse.cpu().data.numpy()

                #print('Epoch: %4d, Train loss: %6.3f Test loss: %6.3f RMSE: %.3f Num. networks: %2d' %
                #      (i, loss.cpu().data.numpy()/len(x_train), test_loss/len(x_test), rmse*y_stds[0], len(nets)))


        train_loss, train_rmse = net.get_loss_and_rmse(x_train, y_train, num_samples=num_samples)
        test_loss, test_rmse = net.get_loss_and_rmse(x_test, y_test, num_samples=num_samples)
        
        train_logliks.append((train_loss.cpu().data.numpy()/len(x_train) + np.log(y_stds)[0]))
        test_logliks.append((test_loss.cpu().data.numpy()/len(x_test) + np.log(y_stds)[0]))

        train_rmses.append(y_stds[0]*train_rmse.cpu().data.numpy())
        test_rmses.append(y_stds[0]*test_rmse.cpu().data.numpy())


    #print('Train log. lik. = %6.3f +/- %6.3f' % (-np.array(train_logliks).mean(), np.array(train_logliks).var()**0.5))
    #print('Test  log. lik. = %6.3f +/- %6.3f' % (-np.array(test_logliks).mean(), np.array(test_logliks).var()**0.5))
    #print('Train RMSE      = %6.3f +/- %6.3f' % (np.array(train_rmses).mean(), np.array(train_rmses).var()**0.5))
    #print('Test  RMSE      = %6.3f +/- %6.3f' % (np.array(test_rmses).mean(), np.array(test_rmses).var()**0.5))
    
    return net



def MCDP_UCI(X, Y, X1):
    
    data = np.hstack((X, Y))
    net  = train_mc_dropout(data=data, drop_prob=0.1, num_epochs=100, n_splits=10, num_units=100, learn_rate=1e-4,
                       weight_decay=1e-1/len(data)**0.5, num_samples=20, log_every=50)

    samples = []
    noises  = []

    for i in range(1000):
    
        preds = net.network.forward(torch.tensor(X1).float()).cpu().data.numpy()
    
        samples.append(preds[:, 0])
        noises.append(np.exp(preds[:, 1]))
    
    samples = np.array(samples)
    noises  = np.array(noises)
    means   = (samples.mean(axis = 0)).reshape(-1)

    aleatoric = (noises**2).mean(axis = 0)**0.5
    epistemic = (samples.var(axis = 0)**0.5).reshape(-1)
    total_unc = (aleatoric**2 + epistemic**2)**0.5
    
    return means, total_unc







