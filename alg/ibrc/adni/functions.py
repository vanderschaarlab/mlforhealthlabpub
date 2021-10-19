import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

from constants import *

def _get_kappa_aux0(x1, s1, omega, tau, u, z, s, lmbda, hyper):
    gamma, alpha, beta, eta = hyper

    ### rho_prime
    aux = z[:,None] * tau[:,u,:] * omega[None,u,:,x1]
    z1 = aux.sum(axis=0) / aux.sum()
    prob_z1 = aux.sum()

    res = -eta * np.log(prob_z1 / tilde_rho_prime) + gamma * interp(z1, lmbda[s1,:])
    return res

_get_kappa_aux0 = jax.vmap(_get_kappa_aux0, in_axes=(0,None,None,None,None,None,None,None,None))
_get_kappa_aux1 = lambda s1, omega, tau, u, z, s, lmbda, hyper: np.sum(omega[u,s1,:] * _get_kappa_aux0(val_x, s1, omega, tau, u, z, s, lmbda, hyper))
_get_kappa_aux1 = jax.vmap(_get_kappa_aux1, in_axes=(0,None,None,None,None,None,None,None))
_get_kappa_aux2 = lambda omega, tau, u, z, s, lmbda, hyper: np.sum(tau[s,u,:] * _get_kappa_aux1(val_s, omega, tau, u, z, s, lmbda, hyper))
_get_kappa_aux2 = jax.vmap(_get_kappa_aux2, in_axes=(0,0,None,None,None,None,None))
_get_kappa_aux3 = lambda u, z, s, lmbda, hyper:  _get_kappa_aux2(val_omega, val_tau, u, z, s, lmbda, hyper)
_get_kappa_aux3 = jax.vmap(_get_kappa_aux3, in_axes=(0,None,None,None,None))
_get_kappa_aux4 = lambda z, s, lmbda, hyper:  _get_kappa_aux3(val_u, z, s, lmbda, hyper)
_get_kappa_aux4 = jax.vmap(_get_kappa_aux4, in_axes=(0,None,None,None))
_get_kappa_aux5 = lambda s, lmbda, hyper:  _get_kappa_aux4(val_z, s, lmbda, hyper)
_get_kappa_aux5 = jax.vmap(_get_kappa_aux5, in_axes=(0,None,None))

get_kappa = lambda lmbda, hyper: _get_kappa_aux5(val_s, lmbda, hyper)
get_kappa = jax.jit(get_kappa)

###

def _get_xi_aux0(ind_xi, u, z, ind_z, kappa, tilde_xi, hyper):
    gamma, alpha, beta, eta = hyper
    return np.log(tilde_xi[ind_xi]) + 1/beta * np.sum(z * kappa[:,ind_z,u,ind_xi])

_get_xi_aux0 = jax.vmap(_get_xi_aux0, in_axes=(0,None,None,None,None,None,None))
_get_xi_aux1 = lambda u, z, ind_z, kappa, tilde_xi, hyper: _get_xi_aux0(val_ind_xi, u, z, ind_z, kappa, tilde_xi, hyper)
_get_xi_aux1 = jax.vmap(_get_xi_aux1, in_axes=(0,None,None,None,None,None))
_get_xi_aux2 = lambda z, ind_z, kappa, tilde_xi, hyper: _get_xi_aux1(val_u, z, ind_z, kappa, tilde_xi, hyper)
_get_xi_aux2 = jax.vmap(_get_xi_aux2, in_axes=(0,0,None,None,None))

@jax.jit
def get_xi(kappa, tilde_xi, hyper):
    aux = _get_xi_aux2(val_z, val_ind_z, kappa, tilde_xi, hyper)
    log_B = logsumexp(aux, axis=-1)
    log_xi = aux - logsumexp(aux, axis=-1, keepdims=True)
    return log_xi, log_B

###

def _get_pi_aux0(u, z, ind_z, nu, tilde_pi, hyper):
    gamma, alpha, beta, eta = hyper
    return np.log(tilde_pi[u]) + 1/alpha * np.sum(z * nu[:,ind_z,u])

_get_pi_aux0 = jax.vmap(_get_pi_aux0, in_axes=(0,None,None,None,None,None))
_get_pi_aux1 = lambda z, ind_z, nu, tilde_pi, hyper: _get_pi_aux0(val_u, z, ind_z, nu, tilde_pi, hyper)
_get_pi_aux1 = jax.vmap(_get_pi_aux1, in_axes=(0,0,None,None,None))

@jax.jit
def get_pi(nu, tilde_pi, hyper):
    aux = _get_pi_aux1(val_z, val_ind_z, nu, tilde_pi, hyper)
    log_A = logsumexp(aux, axis=-1)
    log_pi = aux - logsumexp(aux, axis=-1, keepdims=True)
    return log_pi, log_A

###

@jax.jit
def operator_lmbda(nu, tilde_pi, hyper):
    gamma, alpha, beta, eta = hyper
    log_pi, log_A = get_pi(nu, tilde_pi, hyper)
    pi = np.exp(log_pi)
    pi = pi / pi.sum(axis=-1, keepdims=True)
    res = alpha * np.repeat(log_A[None,:], dim_s, axis=0)
    res = res + np.sum(pi[None,:,:] * (nu - np.sum(val_z.T[:,:,None] * nu, axis=0, keepdims=True)), axis=-1)
    return res, pi

@jax.jit
def operator_nu(kappa, tilde_xi, upsilon, hyper):
    gamma, alpha, beta, eta = hyper
    log_xi, log_B = get_xi(kappa, tilde_xi, hyper)
    xi = np.exp(log_xi)
    xi = xi / xi.sum(axis=-1, keepdims=True)
    res = upsilon[None,:,:]
    res = res + beta * log_B[None,:,:]
    res = res + np.sum(xi[None,:,:,:] * (kappa - np.sum(val_z.T[:,:,None,None] * kappa, axis=0, keepdims=True)), axis=-1)
    return res, xi

@jax.jit
def operator_kappa(lmbda, hyper):
    return get_kappa(lmbda, hyper)

###

def _solve(arg0, arg1):
    lmbda, nu, kappa, pi, xi, tilde_pi, tilde_xi, upsilon, hyper = arg0
    lmbda, pi = operator_lmbda(nu, tilde_pi, hyper)
    nu, xi = operator_nu(kappa, tilde_xi, upsilon, hyper)
    kappa = operator_kappa(lmbda, hyper)
    return (lmbda, nu, kappa, pi, xi, tilde_pi, tilde_xi, upsilon, hyper), None

def solve(tilde_pi, tilde_xi, upsilon, hyper, iter):
    lmbda = np.zeros((dim_s, dim_z))
    nu = np.zeros((dim_s, dim_z, dim_u))
    kappa = np.zeros((dim_s, dim_z, dim_u, dim_xi))
    pi = np.zeros((dim_z, dim_u))
    xi = np.zeros((dim_z, dim_u, dim_xi))
    (lmbda, nu, kappa, pi, xi, *_), _ = jax.lax.scan(_solve,
            (lmbda, nu, kappa, pi, xi, tilde_pi, tilde_xi, upsilon, hyper),
            np.arange(iter, dtype='int32'))
    return lmbda, nu, kappa, pi, xi

solve = jax.jit(solve, static_argnums=4)

###

def _rho(tau, omega, z, u, x1, xi):
    aux = z[:,None] * tau[:,u,:] * omega[None,u,:,x1]
    z1 = aux.sum(axis=0) / aux.sum()
    return z1

_rho = jax.vmap(_rho, in_axes=(0,0,None,None,None,None))

rho = lambda z, u, x1, xi: np.sum(interp_dist(z, xi[:,u,:])[:,None] * _rho(val_tau, val_omega, z, u, x1, xi), axis=0)
rho = jax.jit(rho)

def _rho_batch(arg0, arg1):
    (z, xi), (u, x1) = arg0, arg1
    z1 = rho(z, u, x1, xi)
    return (z1, xi), z

rho_batch = lambda z0, us, x1s, xi: jax.lax.scan(_rho_batch, (z0, xi), (us, x1s))[1]
rho_batch = jax.jit(rho_batch)

rho_batch_batch = jax.vmap(rho_batch, in_axes=(None,0,0,None))
rho_batch_batch = jax.jit(rho_batch_batch)
