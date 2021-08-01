import argparse
import copy
import dill
import jax
import jax.numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('--bias', action='store_true')
args = parser.parse_args()

key = jax.random.PRNGKey(0)
H = 64
L = 64

key = jax.random.PRNGKey(0)

with open('data/data{}-meta.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    data_meta = dill.load(f)
    S = data_meta['S']
    A = data_meta['A']
    Z = data_meta['Z']

with open('data/data{}.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    data = dill.load(f)

n = len(data)
tau = max([d['tau'] for d in data])
data_a = -1 * np.ones((n,tau,A), 'int')
data_z = -1 * np.ones((n,tau,Z), 'int')
for i, traj in zip(range(n), data):
    data_a = data_a.at[i,:traj['tau']].set(jax.nn.one_hot(np.array(traj['a']), A))
    data_z = data_z.at[i,:traj['tau']].set(jax.nn.one_hot(np.array(traj['z']), Z))


def _network(h_c_params, x):
    h, c, params = h_c_params
    f = jax.nn.sigmoid(params['W_f'] @ x + params['U_f'] @ h + params['b_f'])
    i = jax.nn.sigmoid(params['W_i'] @ x + params['U_i'] @ h + params['b_i'])
    o = jax.nn.sigmoid(params['W_o'] @ x + params['U_o'] @ h + params['b_o'])
    c1 = f * c + i * np.tanh(params['W_c'] @ x + params['U_c'] @ h + params['b_c'])
    h1 = o * np.tanh(c1)
    l = np.tanh(params['W_l'] @ h1 + params['b_l'])
    y = jax.nn.softmax(params['W_y'] @ l + params['b_y'])
    return (h1, c1, params), y

def network(params, traj_a, traj_z):
    traj = np.concatenate((traj_a, traj_z), axis=-1)
    _, ys = jax.lax.scan(_network, (np.zeros(H), np.zeros(H), params), traj)
    return ys
network = jax.vmap(network, in_axes=(None,0,0), out_axes=0)
network = jax.jit(network)

def objective(params):
    ys = network(params, data_a[:,:-1,...], data_z[:,:-1,...])
    return -np.sum((data_a[:,1:,...] > 0) * np.log(ys))
objective = jax.jit(objective)

grad_objective = jax.grad(objective)
grad_objective = jax.jit(grad_objective)


params = dict()
key, *subkey = jax.random.split(key, 17)
params['W_f'] = .001 * jax.random.normal(subkey[0], shape=(H,A+Z))
params['W_i'] = .001 * jax.random.normal(subkey[1], shape=(H,A+Z))
params['W_o'] = .001 * jax.random.normal(subkey[2], shape=(H,A+Z))
params['W_c'] = .001 * jax.random.normal(subkey[3], shape=(H,A+Z))
params['W_l'] = .001 * jax.random.normal(subkey[4], shape=(L,H))
params['W_y'] = .001 * jax.random.normal(subkey[5], shape=(A,L))
params['U_f'] = .001 * jax.random.normal(subkey[6], shape=(H,H))
params['U_i'] = .001 * jax.random.normal(subkey[7], shape=(H,H))
params['U_o'] = .001 * jax.random.normal(subkey[8], shape=(H,H))
params['U_c'] = .001 * jax.random.normal(subkey[9], shape=(H,H))
params['b_f'] = .001 * jax.random.normal(subkey[10], shape=(H,))
params['b_i'] = .001 * jax.random.normal(subkey[11], shape=(H,))
params['b_o'] = .001 * jax.random.normal(subkey[12], shape=(H,))
params['b_c'] = .001 * jax.random.normal(subkey[13], shape=(H,))
params['b_l'] = .001 * jax.random.normal(subkey[14], shape=(L,))
params['b_y'] = .001 * jax.random.normal(subkey[15], shape=(A,))

adam_m = dict()
adam_v = dict()
for key in params:
    adam_m[key] = np.zeros(params[key].shape)
    adam_v[key] = np.zeros(params[key].shape)

min_objective = None
min_params = None
objectives = [None] * 100
objectives[0] = objective(params)

for i in range(10000):

    grad = grad_objective(params)
    for key in params:
        adam_m[key] = .1 * grad[key] + .9 * adam_m[key]
        adam_v[key] = .001 * grad[key]**2 + .999 * adam_v[key]
        adam_mhat = adam_m[key] / (1-.9**(i+1))
        adam_vhat = adam_v[key] / (1-.999**(i+1))
        params[key] -= .001 * adam_mhat / (np.sqrt(adam_vhat) + 1e-8)

    objectives[1:] = objectives[:-1]
    objectives[0] = objective(params)
    if min_objective is None or objectives[0] < min_objective:
        min_objective = objectives[0]
        min_params = copy.copy(params)

    if not args.silent:
        print('i = {}, objective = {}'.format(i, objectives[0]))

    if objectives[-1] is not None and objectives[0] - objectives[-1] > -1e-6:
        break

print(min_objective)

res = min_params
res['H'] = H
res['L'] = L
with open('res/res{}-rbc.obj'.format('-bias' if args.bias else ''), 'wb') as f:
    dill.dump(res, f)
