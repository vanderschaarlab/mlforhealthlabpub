import argparse
import dill
import jax
import numpy as np

import pomdp
pomdp.horizon = 25

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true')
parser.add_argument('--bias', action='store_true')
args = parser.parse_args()

with open('data/data{}-meta.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    data_meta = dill.load(f)
    S = data_meta['S']
    A = data_meta['A']
    Z = data_meta['Z']
    envr_b0 = data_meta['envr_b0']
    envr_O = data_meta['envr_O']
    plcy_b0 = data_meta['plcy_b0']
    plcy_T = data_meta['plcy_T']
    plcy_O = data_meta['plcy_O']
    plcy_mu = data_meta['plcy_mu']

def log_pi_il(mu, eta, b):
    res = -eta * np.sum((mu - b[None,...])**2, axis=-1)
    return res - np.log(np.sum(np.exp(res)))

def log_pi_irl(alp, bet, b):
    res = np.zeros(A)
    for a in range(A):
        if alp[a].size == 0:
            res[a] = -1e6
        else:
            res[a] = bet * np.amax(alp[a] @ b)
    return res - np.log(np.sum(np.exp(res)))

algs = list()

with open('res/res{}-interpole.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    res = dill.load(f)
    algs.append(dict())
    algs[-1]['name'] = 'interpole'
    algs[-1]['log_pi'] = log_pi_il
    algs[-1]['b0'] = res['b0']
    algs[-1]['O'] = res['O']
    algs[-1]['tht'] = res['mu']

with open('res/res{}-offpoirl.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    res = dill.load(f)
    res = res['out'][500::10]
    algs.append(dict())
    algs[-1]['name'] = 'offpoirl'
    algs[-1]['log_pi'] = log_pi_irl
    algs[-1]['b0'] = np.zeros(S)
    algs[-1]['O'] = np.zeros((A,S,Z))
    algs[-1]['R'] = np.zeros((S,A))
    for b0, _, O, R in res:
        algs[-1]['b0'] += b0
        algs[-1]['O'] += O
        algs[-1]['R'] += R
    algs[-1]['b0'] /= len(res)
    algs[-1]['O'] /= len(res)
    algs[-1]['R'] /= len(res)
    algs[-1]['tht'] = pomdp.solve(S, A, Z, algs[-1]['b0'], plcy_T, algs[-1]['O'], algs[-1]['R'])

with open('res/res{}-poirl.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    res = dill.load(f)
    res = res['out'][500::10]
    algs.append(dict())
    algs[-1]['name'] = 'poirl'
    algs[-1]['log_pi'] = log_pi_irl
    algs[-1]['b0'] = np.zeros(S)
    algs[-1]['O'] = np.zeros((A,S,Z))
    algs[-1]['R'] = np.zeros((S,A))
    for b0, _, O, R in res:
        algs[-1]['b0'] += b0
        algs[-1]['O'] += O
        algs[-1]['R'] += R
    algs[-1]['b0'] /= len(res)
    algs[-1]['O'] /= len(res)
    algs[-1]['R'] /= len(res)
    algs[-1]['tht'] = pomdp.solve(S, A, Z, algs[-1]['b0'], plcy_T, algs[-1]['O'], algs[-1]['R'])

with open('res/res{}-pombil.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    res = dill.load(f)
    algs.append(dict())
    algs[-1]['name'] = 'pombil'
    algs[-1]['log_pi'] = log_pi_il
    algs[-1]['b0'] = res['b0']
    algs[-1]['O'] = res['O']
    algs[-1]['tht'] = res['mu']

with open('res/res{}-rbc.obj'.format('-bias' if args.bias else ''), 'rb') as f:
    alg_rbc = dill.load(f)

for alg in algs:
    alg['e1'] = list()
    alg['e2'] = list()
    alg['e3'] = list()
e1_rbc = list()
e2_rbc = list()

for i in range(500):
    if not args.silent:
        if i % 100 == 0:
            print('i = {}'.format(i))

    s = np.random.choice(range(S), p=envr_b0)
    b = plcy_b0
    tau = None

    for alg in algs:
        alg['b'] = alg['b0']
        alg['tau'] = None
    h_rbc = np.zeros(alg_rbc['H'])
    c_rbc = np.zeros(alg_rbc['H'])
    tau_rbc = None

    t = 0
    while True:
        t += 1

        pi = np.exp(log_pi_il(plcy_mu, 10, b))
        for alg in algs:
            alg['pi'] = np.exp(alg['log_pi'](alg['tht'], 10, alg['b']))
        if t > 1:
            pi_rbc = np.array(jax.nn.softmax(alg_rbc['W_y'] @ l_rbc + alg_rbc['b_y']))

        if tau is None:
            for alg in algs:
                alg['e2'].append(np.sum(pi * np.log(pi / alg['pi'])))
                alg['e3'].append(np.sum(b * np.log(b / alg['b'])))
            if t > 1:
                e2_rbc.append(np.sum(pi * np.log(pi / pi_rbc)))

        a = np.random.choice(range(A), p=pi)
        for alg in algs:
            alg['a'] = np.random.choice(range(A), p=alg['pi'])
        if t > 1:
            a_rbc = np.random.choice(range(A), p=pi_rbc)

        z = np.random.choice(range(Z), p=envr_O[2,s,:])
        b = plcy_O[2,:,z] * b
        b /= b.sum()
        for alg in algs:
            alg['b'] = alg['O'][2,:,z] * alg['b']
            alg['b'] /= alg['b'].sum()

        x = np.concatenate((np.array([0, 0, 1]), jax.nn.one_hot(z, Z)))
        f = jax.nn.sigmoid(alg_rbc['W_f'] @ x + alg_rbc['U_f'] @ h_rbc + alg_rbc['b_f'])
        i = jax.nn.sigmoid(alg_rbc['W_i'] @ x + alg_rbc['U_i'] @ h_rbc + alg_rbc['b_i'])
        o = jax.nn.sigmoid(alg_rbc['W_o'] @ x + alg_rbc['U_o'] @ h_rbc + alg_rbc['b_o'])
        c_rbc = f * c_rbc + i * np.tanh(alg_rbc['W_c'] @ x + alg_rbc['U_c'] @ h_rbc + alg_rbc['b_c'])
        h_rbc = o * np.tanh(c_rbc)
        l_rbc = np.tanh(alg_rbc['W_l'] @ h_rbc + alg_rbc['b_l'])

        if tau is None and a != 2:
            tau = t
        for alg in algs:
            if alg['tau'] is None and alg['a'] != 2:
                alg['tau'] = t
        if t > 1:
            if tau_rbc is None and a_rbc != 2:
                tau_rbc = t

        brk = tau is not None
        for alg in algs:
            brk = brk and alg['tau'] is not None
        brk = brk and tau_rbc is not None
        if brk:
            break

    for alg in algs:
        alg['e1'].append(np.abs(alg['tau'] - tau))
    e1_rbc.append(np.abs(tau_rbc - tau))

k = len(alg['e1'])
k = k // 5 * 5
for alg in algs:
    alg['e1'] = np.array(alg['e1']).reshape(5,-1).mean(axis=-1)
    alg['e2'] = np.array(alg['e2'][:k]).reshape(5,-1).mean(axis=-1)
    alg['e3'] = np.array(alg['e3'][:k]).reshape(5,-1).mean(axis=-1)
e1_rbc = np.array(e1_rbc).reshape(5,-1).mean(axis=-1)
e2_rbc = np.array(e2_rbc[:k]).reshape(5,-1).mean(axis=-1)

print('alg: STE, policy mismatch, belief mismatch')
for alg in algs:
    print('{}: {} ({}), {} ({}), {} ({})'.format(alg['name'], alg['e1'].mean(), alg['e1'].std(), alg['e2'].mean(), alg['e2'].std(), alg['e3'].mean(), alg['e3'].std()))
print('rbc: {} ({}), {} ({}), n/a'.format(e1_rbc.mean(), e1_rbc.std(), e2_rbc.mean(), e2_rbc.std()))
