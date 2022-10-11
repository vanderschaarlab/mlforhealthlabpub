import autograd.numpy.random as npr
import autograd.numpy as np

from models import transductive
from utils import nice_plot



npr.seed(1234)

class feature_dist():

    def __init__(self,mean=0,sd=1):
        self.mean = mean 
        self.sd = sd
    
    def pdf(self,x):
        return(ss.norm.pdf(x,loc=self.mean,scale=self.sd))
    
    def sample(self,num):
        return npr.normal(loc=self.mean,scale=self.sd,size = (num,1))

P_x = feature_dist(7,2)
Q_x = feature_dist(11,2)

x = np.linspace(0, 18,1000)

num_samples = 50

samples_p = P_x.sample(num_samples)
samples_q = Q_x.sample(num_samples)

def y_target(x):
    return np.sin(x)/2 + x/4 - x**2/100

def y_noise(n,sd):
    return npr.normal(loc=0,scale=sd,size=(n,1))


y_samples_p = y_target(samples_p) + y_noise(num_samples, 0.1)
y_samples_q = y_target(samples_q) + y_noise(num_samples, 0.1)
true_func = y_target(x)


model = transductive([1,32,64,1],d_units = 8)
model.train(samples_p, y_samples_p, samples_q, 1000)

nice_plot(model,samples_p,y_samples_p,samples_q, \
	y_samples_q,true_func,save='transductive_dropout.pdf')
