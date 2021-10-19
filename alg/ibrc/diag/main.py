import argparse
import dill
import jax
import jax.numpy as np

from constants import *
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='')
args = parser.parse_args()

key = jax.random.PRNGKey(0)

inf = 1e3
eps = 1e-3

###
ext_hyper = dict()

if 'alp' in args.tag:
    ext_hyper['beta'] = inf
    ext_hyper['eta'] = eps
    print('ext_beta = inf, ext_eta = eps')

if 'bet' in args.tag or 'eta' in args.tag:
    ext_hyper['alpha'] = .5
    print('ext_alpha = .5')

with open('diag/data/{}.obj'.format(args.tag), 'rb') as f:
    data = dill.load(f)

_likelihood_aux0 = lambda z, u, pi: (u >= 0) * np.log(interp(z, pi[:,u]))
_likelihood_aux0 = jax.vmap(_likelihood_aux0, in_axes=(0,0,None))
_likelihood_aux1 = lambda us, x1s, pi, xi: _likelihood_aux0(rho_batch(ext_z0, us, x1s, xi), us, pi).sum()
_likelihood_aux1 = jax.vmap(_likelihood_aux1, in_axes=(0,0,None,None))
likelihood = lambda pi, xi: _likelihood_aux1(data['u'], data['x1'], pi, xi).sum()
likelihood = jax.jit(likelihood)

def _sample(arg0, arg1):
    (params, like, step, rate), key = arg0, arg1
    keys = jax.random.split(key, 2)

    _params = params + step * jax.random.normal(keys[0], shape=params.shape)
    _alpha, _beta, _eta = np.exp(_params)
    _alpha = ext_hyper['alpha'] if 'alpha' in ext_hyper else _alpha
    _beta = ext_hyper['beta'] if 'beta' in ext_hyper else _beta
    _eta = ext_hyper['eta'] if 'eta' in ext_hyper else _eta
    *_, _pi, _xi = solve(tilde_pi, tilde_xi, upsilon, (gamma, _alpha, _beta, _eta), 100)
    _like = likelihood(_pi, _xi)

    cond = _like - like > np.log(jax.random.uniform(keys[1]))
    params = jax.lax.select(cond, _params, params)
    like = jax.lax.select(cond, _like, like)
    rate = rate + cond

    return (params, like, step, rate), params

def sample(params, key, step, count):

    alpha, beta, eta = np.exp(params)
    alpha = ext_hyper['alpha'] if 'alpha' in ext_hyper else alpha
    beta = ext_hyper['beta'] if 'beta' in ext_hyper else beta
    eta = ext_hyper['eta'] if 'eta' in ext_hyper else eta
    *_, pi, xi = solve(tilde_pi, tilde_xi, upsilon, (gamma, alpha, beta, eta), 100)
    like = likelihood(pi, xi)

    (params, _, _, rate), hypers = jax.lax.scan(_sample, (params, like, step, 0.), jax.random.split(key, count))
    rate = rate / count
    hypers = np.exp(hypers)

    return rate, params, hypers

sample = jax.jit(sample, static_argnums=[2,3])

###
hypers = np.zeros((0,3))
params = np.zeros(3)
for iter in range(110):
    key, subkey = jax.random.split(key)
    rate, params, _hypers = sample(params, subkey, 0.1, 100)
    hypers = np.concatenate((hypers, _hypers))
    print('iter = {} (x100), rate = {}'.format(iter, rate))

hypers = hypers[1000::10,...]
print(hypers.mean(axis=0))

with open('diag/res/{}.obj'.format(args.tag), 'wb') as f:
    dill.dump(hypers, f)
