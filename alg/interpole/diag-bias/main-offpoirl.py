import argparse
import dill
import jax
import numpy as np

import pomdp
pomdp.horizon = 25

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('--cont', action='store_true')
parser.add_argument('--bias', action='store_true')
args = parser.parse_args()

key = jax.random.PRNGKey(0)

with open('data/data{}-meta.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    data_meta = dill.load(f)
    S = data_meta['S']
    A = data_meta['A']
    Z = data_meta['Z']

with open('data/data{}.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    data = dill.load(f)


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
            res += log_pi(alp, 10, b)[a]
            b = O[a,:,z] * (T[:,a,:].T @ b)
            b /= b.sum()
    return res

if args.cont:

    with open(args.output, 'rb') as f:
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
    T = np.array([[[1,0],[1,0],[1,0]],[[0,1],[0,1],[0,1]]])
    O[:2,...] = np.array([[[1,0],[1,0]],[[0,1],[0,1]]])

    ###
    key, subkey = jax.random.split(key)
    R = np.array([[1,-1.5,0], [-1.5,1,0]]) * .25
    R += .001 * np.array(jax.random.normal(subkey, shape=(S,A)))

alp = pomdp.solve(S, A, Z, b0, T, O, R)
like = likelihood(b0, T, O, alp)

rtio = 0
rtio_n = 0
for i in range(len(res['out']), 1000):

    _b0, _T, _O, _R = b0, T, O, R

    key, subkey = jax.random.split(key)
    if jax.random.choice(subkey, [True, False]):

        for traj in data:

            alp = [None] * (traj['tau']+1)
            alp[0] = b0
            for t in range(traj['tau']):
                alp[t+1] = O[traj['a'][t],:,traj['z'][t]] * (T[:,traj['a'][t],:].T @ alp[t])
                alp[t+1] /= alp[t+1].sum()

            bet = [None] * (traj['tau']+1)
            bet[-1] = np.ones(S)
            for t in reversed(range(traj['tau'])):
                bet[t] = T[:,traj['a'][t],:] @ (O[traj['a'][t],:,traj['z'][t]] * bet[t+1])
                bet[t] /= bet[t].sum()

            gmm = [None] * (traj['tau']+1)
            for t in range(traj['tau']+1):
                gmm[t] = alp[t] * bet[t]
                gmm[t] /= gmm[t].sum()

            traj['s'] = [None] * (traj['tau']+1)
            for t in range(traj['tau']+1):
                key, subkey = jax.random.split(key)
                traj['s'][t] = jax.random.choice(subkey, range(S), p=gmm[t])

        dir_b0 = np.ones(b0.shape)
        dir_T = np.ones(T.shape)
        dir_O = np.ones(O.shape)

        for traj in data:

            dir_b0[traj['s'][0]] += 1
            for t in range(traj['tau']):
                dir_T[traj['s'][t],traj['a'][t],traj['s'][t+1]] += 1
            for t in range(traj['tau']):
                dir_O[traj['a'][t],traj['s'][t+1],traj['z'][t]] += 1

        ###
        key, subkey = jax.random.split(key)
        _b0 = np.array(jax.random.dirichlet(subkey, dir_b0))
        if args.bias:
            _b0 = np.array([.5,.5])
        _T = np.array([[[1,0],[1,0],[1,0]],[[0,1],[0,1],[0,1]]])
        _O = np.array([[[1,0],[1,0]],[[0,1],[0,1]],[[.5,.5],[.5,.5]]])
        for s in range(S):
            key, subkey = jax.random.split(key)
            _O[2,s,:] = np.array(jax.random.dirichlet(subkey, dir_O[2,s,:]))

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
        with open('res/res{}-offpoirl.obj'.format('-bias' if args.bias else ''), 'wb') as f:
            dill.dump(res, f)

with open('res/res{}-offpoirl.obj'.format('-bias' if args.bias else ''), 'wb') as f:
    dill.dump(res, f)
