import dill
import numpy as np
import jax.numpy as jnp

np.random.seed(0)

dim_s = 3
dim_x = 12
dim_u = 2

with open('adni/data/adni.obj', 'rb') as f:
    data = dill.load(f)
with open('adni/data/adni-meta.obj', 'rb') as f:
    data_meta = dill.load(f)

###

tags = ['', 'age', 'apoe', 'female']
filters = [
    lambda i: True,
    lambda i: data_meta['age'][i] > 75,
    lambda i: data_meta['apoe'][i] != 0,
    lambda i: data_meta['gender'][i] == 'Female']

for tag, filter in zip(tags, filters):

    ii = [i for i in range(len(data)) if filter(i)]

    N = len(ii)
    T = max([data[i]['tau'] for i in ii])
    print('N = {}, T = {}'.format(N, T))

    data1 = dict()
    data1['u'] = -1 * jnp.ones((N,T), dtype='int32')
    data1['x1'] = -1 * jnp.ones((N,T), dtype='int32')
    for i in ii:
        data1['u'] = data1['u'].at[i,:data[i]['tau']].set(jnp.array(data[i]['a']))
        data1['x1'] = data1['x1'].at[i,:data[i]['tau']].set(jnp.array(data[i]['z']))

    with open('adni/data/data{}{}.obj'.format('-' if tag else '', tag), 'wb') as f:
        dill.dump(data1, f)

###

ext_z0 = np.random.dirichlet([1] * dim_s)
ext_tau = np.random.dirichlet([1] * dim_s, size=(dim_s, dim_u))
ext_omega = np.zeros((dim_u, dim_s, dim_x)) / dim_x
ext_omega[0,:,(0,4,8)] = np.random.dirichlet([1] * 3, size=dim_s).T
ext_omega[1,:,(1,2,3,5,6,7,9,10,11)] = np.random.dirichlet([1] * 9, size=dim_s).T

for i in range(1000):

    z0 = np.zeros(dim_s)
    tau = np.zeros((dim_s, dim_u, dim_s))
    omega = np.zeros((dim_u, dim_s, dim_x))

    for traj in data:

        alp = [np.zeros(ext_z0.shape) for _ in range(traj['tau']+1)]
        alp[0] = ext_z0.copy()
        for t in range(traj['tau']):
            alp[t+1] = np.sum(ext_omega[None,traj['a'][t],:,traj['z'][t]] * ext_tau[:,traj['a'][t],:] * alp[t][:,None], axis=0)
        bet = [np.ones(ext_z0.shape) for _ in range(traj['tau']+1)]
        for t in reversed(range(traj['tau'])):
            bet[t] = np.sum(ext_omega[None,traj['a'][t],:,traj['z'][t]] * ext_tau[:,traj['a'][t],:] * bet[t+1][None,:], axis=0)
        gmm = [a * b for a, b in zip(alp, bet)]
        gmm = [g / g.sum() for g in gmm]
        xi = [None] * traj['tau']
        for t in range(traj['tau']):
            xi[t] = ext_omega[None,traj['a'][t],:,traj['z'][t]] * ext_tau[:,traj['a'][t],:] * alp[t][:,None] * bet[t+1][None,:]
            xi[t] /= xi[t].sum()

        z0 += gmm[0]
        for t in range(traj['tau']):
            tau[:,traj['a'][t],:] += xi[t]
            omega[traj['a'][t],:,traj['z'][t]] += gmm[t+1]

    z0 /= z0.sum()
    tau /= tau.sum(axis=-1, keepdims=True)
    omega /= omega.sum(axis=-1, keepdims=True)

    if np.abs(z0-ext_z0).max() < 1e-3 \
        and np.abs(tau-ext_tau).max() < 1e-3 \
        and np.abs(omega-ext_omega).max() < 1e-3:
        break

    ext_z0 = z0
    ext_tau = tau
    ext_omega = omega

uni_tau = np.ones((dim_s, dim_u, dim_s)) / dim_s
uni_omega = np.zeros((dim_u, dim_s, dim_x)) / dim_x
uni_omega[0,:,(0,4,8)] = np.ones((dim_s, 3)).T / 3
uni_omega[1,:,(1,2,3,5,6,7,9,10,11)] = np.ones((dim_s, 9)).T / 9

pes_tau = .5 * uni_tau + .5 * ext_tau
pes_tau /= pes_tau.sum(axis=-1, keepdims=True)
pes_omega = .5 * uni_omega + .5 * ext_omega
pes_omega /= pes_omega.sum(axis=-1, keepdims=True)

val_tau = np.stack((ext_tau, pes_tau))
val_omega = np.stack((ext_omega, pes_omega))

with open('adni/data/data-meta.obj', 'wb') as f:
    dill.dump((ext_z0, val_tau, val_omega), f)
