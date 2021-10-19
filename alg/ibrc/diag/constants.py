import jax
import jax.numpy as np

dim_s = 2
dim_z = 100
dim_x = 2
dim_u = 3
dim_xi = 3

val_s = np.arange(dim_s, dtype='int32')
val_z = np.linspace(0, 1, dim_z)
val_z = np.stack((val_z, 1-val_z), axis=-1)
val_ind_z = np.arange(dim_z, dtype='int32')
val_x = np.arange(dim_x, dtype='int32')
val_u = np.arange(dim_u, dtype='int32')
val_ind_xi = np.arange(dim_xi, dtype='int32')
val_tau = np.array([[[[.5,.5],[.5,.5],[1.,0.]],[[.5,.5],[.5,.5],[0.,1.]]]] * dim_xi)
val_omega = np.array([[[.6,.4],[.4,.6]],[[.7,.3],[.3,.7]],[[.8,.2],[.2,.8]]])

###
# dim_xi = 1
# val_ind_xi = np.arange(dim_xi, dtype='int32')
# val_tau = np.array([[[[.5,.5],[.5,.5],[1.,0.]],[[.5,.5],[.5,.5],[0.,1.]]]])
# val_omega = np.array([[[.7,.3],[.3,.7]]])

assert val_z.shape == (dim_z, dim_s)
assert val_tau.shape == (dim_xi, dim_s, dim_u, dim_s)
assert val_omega.shape == (dim_xi, dim_s, dim_x)

tilde_pi = np.ones(dim_u) / dim_u
tilde_xi = np.ones(dim_xi) / dim_xi
tilde_rho_prime = 1 ###

gamma = .95
# alpha, beta, eta = 1, 1, 1
# hyper = (gamma, alpha, beta, eta)

upsilon = np.array([[10,-36,-1],[-36,10,-1]])
assert upsilon.shape == (dim_s, dim_u)

ext_z0 = np.array([.5,.5])
ext_tau = np.array([[[.5,.5],[.5,.5],[1.,0.]],[[.5,.5],[.5,.5],[0.,1.]]])
ext_omega = np.array([[.7,.3],[.3,.7]])

assert ext_z0.shape == (dim_s,)
assert ext_tau.shape == (dim_s, dim_u, dim_s)
assert ext_omega.shape == (dim_s, dim_x)

# interp() and interp_dist() only work when dim_s == 2

@jax.jit
def interp(z, vals):
    return np.interp(z[0], val_z[:,0], vals)

interp_dist = jax.vmap(interp, in_axes=(None,-1))
interp_dist = jax.jit(interp_dist)
