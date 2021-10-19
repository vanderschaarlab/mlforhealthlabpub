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

with open('diag/data/{}.obj'.format(args.tag), 'rb') as f:
    data = dill.load(f)

_likelihood_aux0 = lambda z, u, pi: (u >= 0) * np.log(interp(z, pi[:,u]))
_likelihood_aux0 = jax.vmap(_likelihood_aux0, in_axes=(0,0,None))
_likelihood_aux1 = lambda us, x1s, pi, xi: _likelihood_aux0(rho_batch(ext_z0, us, x1s, xi), us, pi).sum()
_likelihood_aux1 = jax.vmap(_likelihood_aux1, in_axes=(0,0,None,None))
likelihood = lambda pi, xi: _likelihood_aux1(data['u'], data['x1'], pi, xi).sum()
likelihood = jax.jit(likelihood)

def _sample(arg0, arg1):
    (upsilon, like, step, rate), key = arg0, arg1
    keys = jax.random.split(key, 2)

    _upsilon = upsilon + step * jax.random.normal(keys[0], shape=upsilon.shape)
    *_, _pi, _xi = solve(tilde_pi, tilde_xi, np.concatenate((_upsilon, -np.ones((2,1))), axis=-1), (gamma, .5, inf, eps), 100)
    _like = likelihood(_pi, _xi)

    cond = _like - like > np.log(jax.random.uniform(keys[1]))
    upsilon = jax.lax.select(cond, _upsilon, upsilon)
    like = jax.lax.select(cond, _like, like)
    rate = rate + cond

    return (upsilon, like, step, rate), upsilon

def sample(upsilon, key, step, count):

    *_, pi, xi = solve(tilde_pi, tilde_xi, np.concatenate((upsilon, -np.ones((2,1))), axis=-1), (gamma, .5, inf, eps), 100)
    like = likelihood(pi, xi)

    (upsilon, _, _, rate), upsilons = jax.lax.scan(_sample, (upsilon, like, step, 0.), jax.random.split(key, count))
    rate = rate / count

    return rate, upsilon, upsilons

sample = jax.jit(sample, static_argnums=[2,3])

###
upsilons = np.zeros((0,dim_s,dim_u-1))
upsilon = np.zeros((dim_s,dim_u-1))
for iter in range(110):
    key, subkey = jax.random.split(key)
    rate, upsilon, _upsilons = sample(upsilon, subkey, 0.1, 100)
    upsilons = np.concatenate((upsilons, _upsilons))
    print('iter = {} (x100), rate = {}'.format(iter, rate))

upsilons = upsilons[1000::10,...]
print(upsilons.mean(axis=0))

with open('diag/res/irl-{}.obj'.format(args.tag), 'wb') as f:
    dill.dump(upsilons, f)
