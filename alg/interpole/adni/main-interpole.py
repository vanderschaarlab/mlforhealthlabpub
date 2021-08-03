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

def _messages_alp(alp_T_O, a_z):
    alp, T, O = alp_T_O
    a, z = a_z
    alp1 = jax.lax.select(a >= 0, O[a,:,z] * (T[:,a,:].T @ alp), alp)
    alp1 = alp1 / alp1.sum()
    return (alp1, T, O), alp1

def _messages_bet(bet_T_O, a_z):
    bet, T, O = bet_T_O
    a, z = a_z
    bet1 = jax.lax.select(a >= 0, T[:,a,:] @ (O[a,:,z] * bet), bet)
    bet1 = bet1 / bet1.sum()
    return (bet1, T, O), bet1

def _messages_xi(T, O, a, z, alp, bet1):
    xi = jax.lax.select(a >= 0, T[:,a,:] * O[None,a,:,z] * (alp[:,None] @ bet1[None,:]), np.eye(S))
    xi = xi / xi.sum()
    return xi
_messages_xi = jax.vmap(_messages_xi, in_axes=(None,None,0,0,0,0), out_axes=0)

def messages(b0, T, O, traj_a, traj_z):
    _, alps = jax.lax.scan(_messages_alp, (b0, T, O), (traj_a, traj_z))
    alps = np.concatenate((b0[None,...], alps))
    _, bets = jax.lax.scan(_messages_bet, (np.ones(S), T, O), (traj_a, traj_z), reverse=True)
    bets = np.concatenate((bets, np.ones(S)[None,...]))
    gmms = alps * bets
    gmms = gmms / gmms.sum(axis=-1, keepdims=True)
    xis = _messages_xi(T, O, traj_a, traj_z, alps[:-1], bets[1:])
    return gmms, xis
messages = jax.vmap(messages, in_axes=(None,None,None,0,0), out_axes=0)
messages = jax.jit(messages)

def _likelihood0(T, O, mu, eta, a, z, gmm1, xi, b):
    res = (a >= 0) * np.sum(gmm1 * np.log(O[a,:,z]+1e-6))
    res += (a >= 0) * np.sum(xi * np.log(T[:,a,:]+1e-6))
    res += (a >= 0) * log_pi(mu, eta, b)[a]
    return res
_likelihood0 = jax.vmap(_likelihood0, in_axes=(None,None,None,None,0,0,0,0,0), out_axes=0)

def _likelihood1(b_T_O, a_z):
    b, T, O = b_T_O
    a, z = a_z
    b1 = jax.lax.select(a >= 0, O[a,:,z] * (T[:,a,:].T @ b), b)
    b1 = b1 / b1.sum()
    return (b1, T, O), b1

def likelihood(b0, T, O, mu, eta, traj_a, traj_z, gmms, xis):
    _, bs = jax.lax.scan(_likelihood1, (b0, T, O), (traj_a, traj_z))
    bs = np.concatenate((b0[None,...], bs))
    res = np.sum(gmms[0] * np.log(b0))
    res = res + _likelihood0(T, O, mu, eta, traj_a, traj_z, gmms[1:], xis, bs[:-1]).sum()
    return res
likelihood = jax.vmap(likelihood, in_axes=(None,None,None,None,None,0,0,0,0), out_axes=0)
likelihood = jax.jit(likelihood)

###
def unpack(params):
    b0 = np.exp(params['b0'])
    b0 = b0 / b0.sum()
    T = np.exp(params['T'])
    T = T / T.sum()
    O1 = np.exp(params['O1'])
    O1 = O1 / O1.sum(axis=-1, keepdims=True)
    O2 = np.exp(params['O2'])
    O2 = O2 / O2.sum(axis=-1, keepdims=True)
    O = np.zeros((A,S,Z))
    O = O.at[0,:,(0,4,8)].set(O1)
    O = O.at[1,:,(1,2,3,5,6,7,9,10,11)].set(O2)
    mu = params['mu']
    mu = mu / mu.sum(axis=-1, keepdims=True)
    eta = 1
    return b0, T, O, mu, eta
unpack = jax.jit(unpack)

def objective(params, GMMS, XIS):
    b0, T, O, mu, eta = unpack(params)
    return likelihood(b0, T, O, mu, eta, data_a, data_z, GMMS, XIS).sum()
objective = jax.jit(objective)

grad_objective = jax.grad(objective)
grad_objective = jax.jit(grad_objective)


key, *subkey = jax.random.split(key, 4)
b0 = np1.array(jax.random.dirichlet(subkey[0], np.ones(S)))
T = np1.array(jax.random.dirichlet(subkey[1], np.ones((S,A,S)), shape=(S,A)))
O = np1.array(jax.random.dirichlet(subkey[2], np.ones((A,S,Z)), shape=(A,S)))

###
O[0,:,(1,2,3,5,6,7,9,10,11)] = np1.zeros((S,9)).T
O[1,:,(0,4,8)] = np1.zeros((S,3)).T
O /= O.sum(axis=-1, keepdims=True)

key, subkey = jax.random.split(key)
mu = np.ones((A,S))
mu += .001 * jax.random.normal(subkey, shape=(A,S))
mu /= mu.sum(axis=-1, keepdims=True)
eta = 1

###
params = dict()
params['mu'] = mu
params['b0'] = np.log(np.e * b0)
params['T'] = np.log(np.e * T)
params['O1'] = np.log(np.e * O[0,:,(0,4,8)].T)
params['O2'] = np.log(np.e * O[1,:,(1,2,3,5,6,7,9,10,11)].T)

adam_m = dict()
adam_v = dict()
for key in params:
    adam_m[key] = np.zeros(params[key].shape)
    adam_v[key] = np.zeros(params[key].shape)

max_objective = None
max_params = None

GMMS, XIS = messages(b0, T, O, data_a, data_z)

objectives = [None] * 100
objectives[0] = objective(params, GMMS, XIS)

for i in range(10000):

    grad = grad_objective(params, GMMS, XIS)
    for key in params:
        adam_m[key] = .1 * grad[key] + .9 * adam_m[key]
        adam_v[key] = .001 * grad[key]**2 + .999 * adam_v[key]
        adam_mhat = adam_m[key] / (1-.9**(i+1))
        adam_vhat = adam_v[key] / (1-.999**(i+1))
        params[key] += .001 * adam_mhat / (np.sqrt(adam_vhat) + 1e-8)

    b0, T, O, _, _ = unpack(params)
    GMMS, XIS = messages(b0, T, O, data_a, data_z)

    objectives[1:] = objectives[:-1]
    objectives[0] = objective(params, GMMS, XIS)
    if max_objective is None or objectives[0] > max_objective:
        max_objective = objectives[0]
        max_params = copy.copy(params)

    if not args.silent:
        print('i = {}, objective = {}'.format(i, objectives[0]))

    if objectives[-1] is not None and objectives[0] - objectives[-1] < 1e-6:
        break

b0, T, O, mu, eta = unpack(max_params)
if not args.silent:
    print('objective = {}'.format(max_objective))

res = dict()
res['b0'] = b0
res['T'] = T
res['O'] = O
res['mu'] = mu
res['eta'] = eta
with open('res/adni/res{}-interpole.obj'.format(args.i), 'wb') as f:
    dill.dump(res, f)
