import argparse
import dill
import jax
import jax.numpy as np1
import numpy as np

import pomdp
pomdp.horizon = 25

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('-i', type=int, default=0)
parser.add_argument('-n', type=int, default=5)
parser.add_argument('--cont', action='store_true')
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
data_a = -1 * np1.ones((n,tau), 'int')
data_z = -1 * np1.ones((n,tau), 'int')
for i, traj in zip(range(n), data):
    data_a = data_a.at[i,:traj['tau']].set(np1.array(traj['a']))
    data_z = data_z.at[i,:traj['tau']].set(np1.array(traj['z']))


def log_pi(alp, bet, b):
    res = np.zeros(A)
    for a in range(A):
        if alp[a].size == 0:
            res[a] = -1e6
        else:
            res[a] = bet * (alp[a] @ b).max()
    return res - np.log(np.sum(np.exp(res)))

def likelihood(b0, T, O, alp):
    res = 0
    for traj in data:
        b = b0
        for a, z in zip(traj['a'], traj['z']):
            res += log_pi(alp, 1, b)[a]
            b = O[a,:,z] * (T[:,a,:].T @ b)
            b /= b.sum()
    return res

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

def messages(b0, T, O, traj_a, traj_z):
    _, alps = jax.lax.scan(_messages_alp, (b0, T, O), (traj_a, traj_z))
    alps = np1.concatenate((b0[None,...], alps))
    _, bets = jax.lax.scan(_messages_bet, (np1.ones(S), T, O), (traj_a, traj_z), reverse=True)
    bets = np1.concatenate((bets, np1.ones(S)[None,...]))
    gmms = alps * bets
    gmms = gmms / gmms.sum(axis=-1, keepdims=True)
    return gmms
# messages = jax.vmap(messages, in_axes=(None,None,None,0,0), out_axes=0)
# messages = jax.jit(messages)

def _posterior(s_alpT_alpO, a_z_gmm_key):
    s, alpT, alpO = s_alpT_alpO
    a, z, gmm, key = a_z_gmm_key
    s1 = jax.random.choice(key, range(S), p=gmm)
    alpT = jax.lax.select(a >= 0, alpT.at[s,a,s1].add(1), alpT)
    alpO = jax.lax.select(a >= 0, alpO.at[a,s1,z].add(1), alpO)
    return (s1, alpT, alpO), None

def posterior(b0, T, O, traj_a, traj_z, key):
    gmms = messages(b0, T, O, traj_a, traj_z)
    key, subkey = jax.random.split(key)
    s0 = jax.random.choice(subkey, range(S), p=gmms[0])
    alpb0 = np1.zeros(S)
    alpb0 = alpb0.at[s0].add(1)
    alpT = np1.zeros((S,A,S))
    alpO = np1.zeros((A,S,Z))
    subkeys = jax.random.split(key, tau)
    (_, alpT, alpO), _ = jax.lax.scan(_posterior, (s0, alpT, alpO), (traj_a, traj_z, gmms[1:], subkeys))
    return alpb0, alpT, alpO
posterior = jax.vmap(posterior, in_axes=(None,None,None,0,0,0), out_axes=(0,0,0))
posterior = jax.jit(posterior)


if args.cont:

    with open('res/adni/res{}-offpoirl.obj'.format(args.i), 'rb') as f:
        res = dill.load(f)
        key = res['key']
        b0, T, O, R = res['out'][-1]

else:

    res = dict()
    res['out'] = list()

    key, *subkey = jax.random.split(key, 4)
    b0 = np.array(jax.random.dirichlet(subkey[0], np.ones(S)))
    T = np.array(jax.random.dirichlet(subkey[1], np.ones((S,A,S)), shape=(S,A)))
    O = np.array(jax.random.dirichlet(subkey[1], np.ones((A,S,Z)), shape=(A,S)))

    ###
    O[0,:,(1,2,3,5,6,7,9,10,11)] = np.zeros((S,9)).T
    O[1,:,(0,4,8)] = np.zeros((S,3)).T
    O /= O.sum(axis=-1, keepdims=True)

    key, subkey = jax.random.split(key)
    R = np.zeros((S,A))
    R += .001 * np.array(jax.random.normal(subkey, shape=(S,A)))

alp = pomdp.solve(S, A, Z, b0, T, O, R)
like = likelihood(b0, T, O, alp)

rtio = 0
rtio_n = 0
for i in range(len(res['out']), 1000):

    _b0, _T, _O, _R = b0, T, O, R

    key, subkey = jax.random.split(key)
    if jax.random.choice(subkey, [True, False]):

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, n)
        alpb0, alpT, alpO = posterior(b0, T, O, data_a, data_z, subkeys)

        alpb0 = 1 + alpb0.sum(axis=0)
        alpT = 1 + alpT.sum(axis=0)
        alpO = 1 + alpO.sum(axis=0)

        key, subkey = jax.random.split(key)
        _b0 = np.array(jax.random.dirichlet(subkey, alpb0))
        _T = np.zeros((S,A,S))
        for s in range(S):
            for a in range(A):
                key, subkey = jax.random.split(key)
                _T[s,a,:] = np.array(jax.random.dirichlet(subkey, alpT[s,a,:]))
        _O = np.zeros((A,S,Z))
        for a in range(A):
            for s in range(S):
                key, subkey = jax.random.split(key)
                _O[a,s,:] = np.array(jax.random.dirichlet(subkey, alpO[a,s,:]))

    else:

        key, subkey = jax.random.split(key)
        _R = R + .001 * np.array(jax.random.normal(subkey, shape=(S,A)))

    _alp = pomdp.solve(S, A, Z, _b0, _T, _O, _R)
    _like = likelihood(_b0, _T, _O, _alp)

    key, subkey = jax.random.split(key)
    unif = jax.random.uniform(subkey)
    if np.log(unif) < _like - like:
        b0, T, O, R = _b0, _T, _O, _R
        like = _like

    rtio += 1 if like == _like else 0
    rtio_n += 1
    if not args.silent:
        print('i = {}, like = {}, {} ({})'.format(i, like, '*' if like == _like else '-', rtio / rtio_n))

    res['key'] = key
    res['out'].append((b0, T, O, R))
    if (i+1) % 100 == 0:
        with open('res/adni/res{}-offpoirl.obj'.format(args.i), 'wb') as f:
            dill.dump(res, f)

with open('res/adni/res{}-offpoirl.obj'.format(args.i), 'wb') as f:
    dill.dump(res, f)
