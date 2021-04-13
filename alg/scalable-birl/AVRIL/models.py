import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers
import jax
import haiku as hk

from tqdm import tqdm
import numpy as onp
import gym

from utils import load_data

def hidden_layers(layers=1,units=64):
    hidden = []
    for i in range(layers):
        hidden += [hk.Linear(units), jax.nn.elu]
    return hidden

def encoder_model(inputs,layers=2,units=64,state_only=True,a_dim=None):
    out_dim = 2
    if not state_only:
        out_dim = a_dim * 2
    mlp = hk.Sequential(hidden_layers(layers) + [hk.Linear(out_dim)])
    return mlp(inputs)

def q_network_model(inputs,a_dim,layers=2,units=64):
    
    mlp = hk.Sequential(hidden_layers(layers) + [hk.Linear(a_dim)])
    return mlp(inputs)

def kl_gaussian(mean, var):
    return 0.5 * (-np.log(var) - 1.0 + var + mean**2)


class avril():
    def __init__(self,
                inputs,
                targets,
                state_dim,
                action_dim,
                state_only     = True,
                encoder_layers = 2,
                encoder_units  = 64,
                decoder_layers = 2,
                decoder_units  = 64,
                seed           = 41310):
        
        self.key = random.PRNGKey(seed)

        self.encoder = hk.transform(encoder_model)
        self.q_network = hk.transform(q_network_model)

        self.inputs = inputs
        self.targets = targets 
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_layers = encoder_layers
        self.encoder_units = encoder_units
        self.decoder_layers = decoder_layers
        self.decoder_units = decoder_units

        self.e_params = self.encoder.init(self.key,inputs,encoder_layers,encoder_units,self.state_only,action_dim)
        self.q_params = self.q_network.init(self.key,inputs,action_dim,decoder_layers,decoder_units)

        self.params = (self.e_params,self.q_params)
        return

    def predict(self,state):

        logit = self.q_network.apply(self.q_params,self.key,state,self.a_dim,self.decoder_layers,self.decoder_units)
        return logit

    def reward(self,state):

        r_par = self.encoder.apply(self.e_params,key,state,self.encoder_layers,self.encoder_units,self.state_only,self.a_dim)
        return r_par

    def elbo(self,params,key,inputs,targets):
    
        e_params,q_params = params
    
        q_values = self.q_network.apply(q_params,key,inputs[:,0,:],self.a_dim,self.decoder_layers,self.decoder_units)
        q_values_a = np.take_along_axis(q_values,targets[:,0,:].astype(np.int32),axis=1).reshape(len(inputs))
    
        q_values_next = self.q_network.apply(q_params,key,inputs[:,1,:],self.a_dim,self.decoder_layers,self.decoder_units)
        q_values_next_a = np.take_along_axis(q_values_next,targets[:,1,:].astype(np.int32),axis=1).reshape(len(inputs))
    
        td = q_values_a - q_values_next_a
    
        r_par = self.encoder.apply(e_params,key,inputs[:,0,:],self.encoder_layers,self.encoder_units,self.state_only,self.a_dim)

        if self.state_only:
            means = r_par[:,0].reshape(len(inputs))
            log_sds = r_par[:,1].reshape(len(inputs))
        else:
            means = np.take_along_axis(r_par,(targets[:,0,:]).astype(int),axis=1).reshape((len(inputs),))
            log_sds = np.take_along_axis(r_par,(self.a_dim+targets[:,0,:]).astype(int),axis=1).reshape((len(inputs),))
    
        irl_loss = -jax.scipy.stats.norm.logpdf(td,means,np.exp(log_sds)).mean()
    
        kl = kl_gaussian(means,np.exp(log_sds)**2).mean()
    
        pred = jax.nn.log_softmax(q_values)
    
        neg_log_lik = -np.take_along_axis(pred,targets[:,0,:].astype(np.int32),axis=1).mean()
    
        return neg_log_lik + kl + irl_loss

    def train(self,iters=1000,batch_size=64,l_rate=1e-4):

        init_fun, update_fun, get_params = optimizers.adam(l_rate)
        update_fun = jit(update_fun)
        get_params = jit(get_params)

        params = self.params

        param_state = init_fun(params)

        loss_grad = jit(value_and_grad(self.elbo))

        len_x = len(self.inputs[:,0,:])
        num_batches = np.ceil(len_x / batch_size)

        indx_list = np.array(range(len_x))

        key = self.key

        for itr in tqdm(range(iters)):
    
            if itr%num_batches == 0:
                indx_list_shuffle = jax.random.permutation(key,indx_list)
    
            indx = int((itr%num_batches)*batch_size)
            indxes = indx_list_shuffle[indx:(batch_size+indx)]
            
            key,subkey = random.split(key)

            lik,g_params = loss_grad(params,key,inputs[indxes],targets[indxes])
    
            param_state = update_fun(itr,g_params,param_state)
    
            params = get_params(param_state)

        self.e_params = params[0]
        self.q_params = params[1]

    def gym_test(self,env_test,test_evals=10):
        results = []
        env = gym.make(env_test)
        for t in tqdm(range(test_evals),desc='Testing'):
            observation = env.reset()
            done = False
            rewards=[]
            while not done:
                logit = self.q_network.apply(self.q_params,self.key,observation,self.a_dim,self.decoder_layers,self.decoder_units)
                action = jax.nn.softmax(logit).argmax()
                observation,reward,done,info = env.step(int(action))
                rewards.append(reward)

            results.append(sum(rewards))
        env.close()
        mean_res = sum(results) / test_evals
        print(f'Mean Reward: {mean_res}') 



if __name__ == '__main__':

    inputs,targets,a_dim,s_dim = load_data('CartPole-v1',num_trajs=15)

    print(inputs.shape)
    print(targets.shape)

    model = avril(inputs,targets,s_dim,a_dim,state_only=True)
    model.train(iters=5000)
    model.gym_test('CartPole-v1')


