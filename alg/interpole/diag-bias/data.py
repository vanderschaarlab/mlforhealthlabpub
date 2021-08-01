import argparse
import dill
import jax
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('--bias', action='store_true')
args = parser.parse_args()

key = jax.random.PRNGKey(1209)

S = 2
A = 3
Z = 2

envr_b0 = np.array([.5,.5])
envr_T = np.array([[[1,0],[1,0],[1,0]],[[0,1],[0,1],[0,1]]])
envr_O = np.array([[[1,0],[1,0]],[[0,1],[0,1]],[[.6,.4],[.4,.6]]])

plcy_b0 = envr_b0.copy()
plcy_T = envr_T.copy()
plcy_O = envr_O.copy()
if args.bias:
    plcy_O = np.array([[[1,0],[1,0]],[[0,1],[0,1]],[[.6,.4],[.2,.8]]])
plcy_mu = np.array([[1.3,-0.3],[-0.3,1.3],[.5,.5]]) # 90% confidence

def log_pi(mu, eta, b):
    res = -eta * np.sum((mu - b[None,...])**2, axis=-1)
    return res - np.log(np.sum(np.exp(res)))

data_meta = dict()
data_meta['S'] = S
data_meta['A'] = A
data_meta['Z'] = Z
data_meta['envr_b0'] = envr_b0
data_meta['envr_T'] = envr_T
data_meta['envr_O'] = envr_O
data_meta['plcy_b0'] = plcy_b0
data_meta['plcy_T'] = plcy_T
data_meta['plcy_O'] = plcy_O
data_meta['plcy_mu'] = plcy_mu

with open('data/data{}-meta.obj'.format('-bias' if args.bias else ''), 'wb') as f:
    dill.dump(data_meta, f)

data = list()
for i in range(100):

    data.append(dict())
    data[-1]['a'] = list()
    data[-1]['z'] = list()

    tau = 0
    key, subkey = jax.random.split(key)
    s = jax.random.choice(subkey, range(S), p=envr_b0)
    b = plcy_b0

    while True:

        tau += 1
        key, *subkey = jax.random.split(key, 4)
        a = jax.random.choice(subkey[0], range(A), p=np.exp(log_pi(plcy_mu, 10, b)))
        s = jax.random.choice(subkey[1], range(S), p=envr_T[s,a,:])
        z = jax.random.choice(subkey[2], range(Z), p=envr_O[a,s,:])
        b = plcy_O[a,:,z] * (plcy_T[:,a,:].T @ b)
        b /= b.sum()

        data[-1]['a'].append(a)
        data[-1]['z'].append(z)

        if a == 0 or a == 1:
            break

    data[-1]['tau'] = tau
    if not args.silent:
        print('i = {}, tau = {}'.format(i, tau))

with open('data/data{}.obj'.format('-bias' if args.bias else ''), 'wb') as f:
    dill.dump(data, f)
