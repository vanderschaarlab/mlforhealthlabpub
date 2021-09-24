import dill
import jax
import jax.numpy as np

from constants import *
from functions import *

inf = 1e3
eps = 1e-3

tags = ['alp-lowlow', 'alp-low', 'alp-med', 'alp-hgh', 'bet', 'eta']
hypers = [
    (gamma, 1e-5, inf, eps),
    (gamma, eps, inf, eps),
    (gamma, .5, inf, eps),
    (gamma, 10, inf, eps),
    (gamma, .5, 1.25, eps),
    (gamma, .5, inf, 75)]

for tag, hyper in zip(tags, hypers):
    print('alpha = {}, beta = {}, eta = {}'.format(*hyper[1:]))

    key = jax.random.PRNGKey(0)
    *_, pi, xi = solve(tilde_pi, tilde_xi, upsilon, hyper, 100)

    data = dict()
    data['u'] = list()
    data['x1'] = list()

    for i in range(1000):
        if i % 100 == 0:
            print('{} / 1000'.format(i))

        data['u'].append(list())
        data['x1'].append(list())

        key, subkey = jax.random.split(key)
        s = jax.random.choice(subkey, np.arange(dim_s, dtype='int'), p=ext_z0)
        z = ext_z0.copy()

        u = 2
        while u == 2:

            key, *subkeys = jax.random.split(key, 4)
            u = jax.random.choice(subkeys[0], np.arange(dim_u, dtype='int'), p=interp_dist(z, pi))
            s = jax.random.choice(subkeys[1], np.arange(dim_s, dtype='int'), p=ext_tau[s,u,:])
            x1 = jax.random.choice(subkeys[2], np.arange(dim_x, dtype='int'), p=ext_omega[s,:])
            z = rho(z, u, x1, xi)

            data['u'][-1].append(u)
            data['x1'][-1].append(x1)

    N = len(data['u'])
    T = max([len(us) for us in data['u']])
    print('N = {}, T = {}'.format(N, T))

    data1 = dict()
    data1['u'] = -1 * np.ones((N,T), dtype='int')
    data1['x1'] = -1 * np.ones((N,T), dtype='int')
    for i in range(N):
        data1['u'] = data1['u'].at[i,:len(data['u'][i])].set(np.array(data['u'][i]))
        data1['x1'] = data1['x1'].at[i,:len(data['x1'][i])].set(np.array(data['x1'][i]))

    with open('diag/data/{}.obj'.format(tag), 'wb') as f:
        dill.dump(data1, f)
