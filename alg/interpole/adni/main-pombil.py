import argparse
import copy
import dill
import numpy as np1
import jax
import jax.numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('-i', type=int, default=0)
parser.add_argument('-n', type=int, default=5)
args = parser.parse_args()

key = jax.random.PRNGKey(0)

###
S = 3
A = 2
Z = 12
with open('data/adni.obj', 'rb') as f:
    data = dill.load(f)
    data = data[:len(data)*args.i//args.n] + data[len(data)*(args.i+1)//args.n:]

n = len(data)
tau = max([d['tau'] for d in data])
data_a = -1 * np.ones((n,tau), 'int')
data_z = -1 * np.ones((n,tau), 'int')
for i, traj in zip(range(n), data):
    data_a = data_a.at[i,:traj['tau']].set(np.array(traj['a']))
    data_z = data_z.at[i,:traj['tau']].set(np.array(traj['z']))


def log_pi(mu, eta, b):
    res = -eta * np.sum((mu - b[None,...])**2, axis=-1)
    return res - np.log(np.sum(np.exp(res)))
log_pi = jax.jit(log_pi)

def _messages_alp(alp_T_O, a_z):
    alp, T, O = alp_T_O
    a, z = a_z
    alp1 = jax.lax.select(a >= 0, O[a,:,z] * (T[:,a,:].T @ alp), alp)
    alp1 = alp1 / alp1.sum()
    return (alp1, T, O), alp1


def messages_n(b0, T, O, traj_a, traj_z):
    _, alps = jax.lax.scan(_messages_alp, (b0, T, O), (traj_a, traj_z))
    alps = np.concatenate((b0[None,...], alps))
    return alps[:-1]
messages_n = jax.vmap(messages_n, in_axes=(None,None,None,0,0), out_axes=0)
messages_n = jax.jit(messages_n)

def likelihood_n(mu, eta, a, z, alp):
    return (a >= 0) * log_pi(mu, eta, alp)[a]
likelihood_n = jax.vmap(likelihood_n, in_axes=(None,None,0,0,0), out_axes=0)
likelihood_n = jax.vmap(likelihood_n, in_axes=(None,None,0,0,0), out_axes=0)
likelihood_n = jax.jit(likelihood_n)

###
def unpack_n(params):
    mu = params['mu']
    mu = mu / mu.sum(axis=-1, keepdims=True)
    eta = 1
    return mu, eta
unpack_n = jax.jit(unpack_n)

def objective_n(params, ALPS):
    mu, eta = unpack_n(params)
    return likelihood_n(mu, eta, data_a, data_z, ALPS).sum()
objective_n = jax.jit(objective_n)

grad_objective_n = jax.grad(objective_n)
grad_objective_n = jax.jit(grad_objective_n)


key, *subkey = jax.random.split(key, 4)
b0 = np1.array(jax.random.dirichlet(subkey[0], np.ones(S)))
T = np1.array(jax.random.dirichlet(subkey[1], np.ones((S,A,S)), shape=(S,A)))
O = np1.array(jax.random.dirichlet(subkey[2], np.ones((A,S,Z)), shape=(A,S)))

###
O[0,:,(1,2,3,5,6,7,9,10,11)] = np1.zeros((S,9)).T
O[1,:,(0,4,8)] = np1.zeros((S,3)).T
O /= O.sum(axis=-1, keepdims=True)

for i in range(1000):
    _b0 = np1.zeros(b0.shape)
    _T = np1.zeros(T.shape)
    _O = np1.zeros(O.shape)

    for traj in data:

        alp = [None] * (traj['tau']+1)
        alp[0] = b0
        for t in range(traj['tau']):
            alp[t+1] = O[traj['a'][t],:,traj['z'][t]] * (T[:,traj['a'][t],:].T @ alp[t])
            alp[t+1] /= alp[t+1].sum()

        bet = [None] * (traj['tau']+1)
        bet[-1] = np1.ones(S)
        for t in reversed(range(traj['tau'])):
            bet[t] = T[:,traj['a'][t],:] @ (O[traj['a'][t],:,traj['z'][t]] * bet[t+1])
            bet[t] /= bet[t].sum()

        gmm = [None] * (traj['tau']+1)
        for t in range(traj['tau']+1):
            gmm[t] = alp[t] * bet[t]
            gmm[t] /= gmm[t].sum()

        xi = [None] * traj['tau']
        for t in range(traj['tau']):
            xi[t] = T[:,traj['a'][t],:] * O[None,traj['a'][t],:,traj['z'][t]] * (alp[t][:,None] @ bet[t+1][None,:])
            xi[t] /= xi[t].sum()

        _b0 += gmm[0]
        for t in range(traj['tau']):
            _O[traj['a'][t],:,traj['z'][t]] += gmm[t+1]
        for t in range(traj['tau']):
            _T[:,traj['a'][t],:] += xi[t]

    _b0 /= _b0.sum()
    _T /= _T.sum(axis=-1, keepdims=True)
    _O /= _O.sum(axis=-1, keepdims=True)

    diff = max(np1.abs(_b0-b0).max(), np1.abs(_T-T).max(), np1.abs(_O-O).max())
    b0, T, O = _b0, _T, _O

    if not args.silent:
        print('i = {}, diff = {}'.format(i, diff))
        if diff < 1e-6:
            break

key, subkey = jax.random.split(key)
mu = np.ones((A,S))
mu += .001 * jax.random.normal(subkey, shape=(A,S))
mu /= mu.sum(axis=-1, keepdims=True)
eta = 1

params = dict()
params['mu'] = mu

adam_m = dict()
adam_v = dict()
for key in params:
    adam_m[key] = np.zeros(params[key].shape)
    adam_v[key] = np.zeros(params[key].shape)

max_objective = None
max_params = None

ALPS = messages_n(b0, T, O, data_a, data_z)

objectives = [None] * 100
objectives[0] = objective_n(params, ALPS)

for i in range(10000):

    grad = grad_objective_n(params, ALPS)
    for key in params:
        adam_m[key] = .1 * grad[key] + .9 * adam_m[key]
        adam_v[key] = .001 * grad[key]**2 + .999 * adam_v[key]
        adam_mhat = adam_m[key] / (1-.9**(i+1))
        adam_vhat = adam_v[key] / (1-.999**(i+1))
        params[key] += .001 * adam_mhat / (np.sqrt(adam_vhat) + 1e-8)

    objectives[1:] = objectives[:-1]
    objectives[0] = objective_n(params, ALPS)
    if max_objective is None or objectives[0] > max_objective:
        max_objective = objectives[0]
        max_params = copy.copy(params)

    if not args.silent:
        print('i = {}, objective = {}'.format(i, objectives[0]))

    if objectives[-1] is not None and objectives[0] - objectives[-1] < 1e-6:
        break

mu, eta = unpack_n(max_params)
if not args.silent:
    print('objective = {}'.format(max_objective))

res = dict()
res['b0'] = b0
res['T'] = T
res['O'] = O
res['mu'] = mu
res['eta'] = eta
with open('res/adni/res{}-pombil.obj'.format(args.i), 'wb') as f:
    dill.dump(res, f)
