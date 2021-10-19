import dill
import jax
import jax.numpy as np

dim_s = 3
dim_z0 = 20
dim_z = dim_z0 * (dim_z0+1) // 2
dim_x = 12
dim_u = 2
dim_xi = 2

val_s = np.arange(dim_s, dtype='int32')

val_z = np.array([(z0/(dim_z0-1), z1/(dim_z0-1)) for z0 in range(dim_z0) for z1 in range(dim_z0-z0)])
val_z = np.concatenate((val_z, 1-val_z.sum(axis=-1, keepdims=True)), axis=-1)
val_ind_z = np.arange(dim_z, dtype='int32')
val_x = np.arange(dim_x, dtype='int32')
val_u = np.arange(dim_u, dtype='int32')
val_ind_xi = np.arange(dim_xi, dtype='int32')
with open('adni/data/data-meta.obj', 'rb') as f:
    ext_z0, val_tau, val_omega = dill.load(f)

    ###
    val_tau = val_tau + 1e-3
    val_tau = val_tau / val_tau.sum(axis=-1, keepdims=True)
    val_omega = val_omega + 1e-3
    val_omega = val_omega / val_omega.sum(axis=-1, keepdims=True)

assert val_z.shape == (dim_z, dim_s)
assert val_tau.shape == (dim_xi, dim_s, dim_u, dim_s)
assert val_omega.shape == (dim_xi, dim_u, dim_s, dim_x)

tilde_pi = np.ones(dim_u) / dim_u
tilde_xi = np.ones(dim_xi) / dim_xi
tilde_rho_prime = 1 ###

gamma = .95
upsilon = (val_z.max(axis=-1) > .9).astype('float32')
upsilon = np.stack((upsilon, upsilon - 1626/3964), axis=-1)

assert upsilon.shape == (dim_z, dim_u)

@jax.jit
def interp(z, vals):
    z0 = (z[0] * (dim_z0-1)).astype(int)
    z1 = (z[1] * (dim_z0-1)).astype(int)

    i0 = z1 + dim_z0 * (dim_z0+1) // 2 - (dim_z0-z0) * (dim_z0-z0+1) // 2
    i1 = z1 + dim_z0 * (dim_z0+1) // 2 - (dim_z0-z0-1) * (dim_z0-z0) // 2
    i2 = z1 + 1 + dim_z0 * (dim_z0+1) // 2 - (dim_z0-z0) * (dim_z0-z0+1) // 2
    x0, y0, v0 = val_z[i0,0], val_z[i0,1], vals[i0]
    x1, y1, v1 = val_z[i1,0], val_z[i1,1], vals[i1]
    x2, y2, v2 = val_z[i2,0], val_z[i2,1], vals[i2]

    T = np.array([[x0-x2,x1-x2],[y0-y2,y1-y2]])
    w = np.linalg.inv(T) @ np.array([z[0]-x2, z[1]-y2])
    w0, w1, w2 = w[0], w[1], 1-w[0]-w[1]

    v = w0 * v0 + w1 * v1 + w2 * v2
    return v

_interp_dist = jax.vmap(interp, in_axes=(None,-1))
interp_dist = lambda z, vals: _interp_dist(z, vals)

interp_dist = jax.vmap(interp, in_axes=(None,-1))
interp_dist = jax.jit(interp_dist)

interp_batch = jax.vmap(interp, in_axes=(0,None))
interp_batch_batch = jax.vmap(interp_batch, in_axes=(0,None))
interp_batch_batch = jax.jit(interp_batch_batch)
