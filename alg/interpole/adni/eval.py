import dill
import jax
import numpy as np
import sklearn.metrics as metrics

import pomdp
pomdp.horizon = 25

S = 3
A = 2
Z = 12
with open('data/adni.obj', 'rb') as f:
    data0 = dill.load(f)

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

algs = [dict() for _ in range(4)]
for alg in algs:
    alg['m1'] = list()
    alg['m2'] = list()
    alg['m3'] = list()
m1_rbc = list()
m2_rbc = list()
m3_rbc = list()

for fold in range(5):
    data = data0[len(data0)*fold//5:len(data0)*(fold+1)//5]

    with open('res/adni/res{}-interpole.obj'.format(fold), 'rb') as f:
        res = dill.load(f)
        algs[0]['name'] = 'interpole'
        algs[0]['log_pi'] = log_pi_il
        algs[0]['b0'] = res['b0']
        algs[0]['T'] = res['T']
        algs[0]['O'] = res['O']
        algs[0]['tht'] = res['mu']

    with open('res/adni/res{}-offpoirl.obj'.format(fold), 'rb') as f:
        res = dill.load(f)
        res = res['out'][500::10]
        algs[1]['name'] = 'offpoirl'
        algs[1]['log_pi'] = log_pi_irl
        algs[1]['b0'] = np.zeros(S)
        algs[1]['T'] = np.zeros((S,A,S))
        algs[1]['O'] = np.zeros((A,S,Z))
        algs[1]['R'] = np.zeros((S,A))
        for b0, T, O, R in res:
            algs[1]['b0'] += b0
            algs[1]['T'] += T
            algs[1]['O'] += O
            algs[1]['R'] += R
        algs[1]['b0'] /= len(res)
        algs[1]['T'] /= len(res)
        algs[1]['O'] /= len(res)
        algs[1]['R'] /= len(res)
        algs[1]['tht'] = pomdp.solve(S, A, Z, algs[1]['b0'], algs[1]['T'], algs[1]['O'], algs[1]['R'])

    with open('res/adni/res{}-poirl.obj'.format(fold), 'rb') as f:
        res = dill.load(f)
        res = res['out'][500::10]
        algs[2]['name'] = 'poirl'
        algs[2]['log_pi'] = log_pi_irl
        algs[2]['b0'] = np.zeros(S)
        algs[2]['T'] = np.zeros((S,A,S))
        algs[2]['O'] = np.zeros((A,S,Z))
        algs[2]['R'] = np.zeros((S,A))
        for b0, T, O, R in res:
            algs[2]['b0'] += b0
            algs[2]['T'] += T
            algs[2]['O'] += O
            algs[2]['R'] += R
        algs[2]['b0'] /= len(res)
        algs[2]['T'] /= len(res)
        algs[2]['O'] /= len(res)
        algs[2]['R'] /= len(res)
        algs[2]['tht'] = pomdp.solve(S, A, Z, algs[2]['b0'], algs[2]['T'], algs[2]['O'], algs[2]['R'])

    with open('res/adni/res{}-pombil.obj'.format(fold), 'rb') as f:
        res = dill.load(f)
        algs[3]['name'] = 'pombil'
        algs[3]['log_pi'] = log_pi_il
        algs[3]['b0'] = res['b0']
        algs[3]['T'] = res['T']
        algs[3]['O'] = res['O']
        algs[3]['tht'] = res['mu']

    with open('res/adni/res{}-rbc.obj'.format(fold), 'rb') as f:
        alg_rbc = dill.load(f)

    y_true = list()
    for traj in data:
        for a1 in traj['a'][1:]:
            y_true.append(a1)

    for alg in algs:

        y_score = list()
        for traj in data:
            b = alg['b0']
            for a, z, a1 in zip(traj['a'], traj['z'], traj['a'][1:]):
                b = alg['O'][a,:,z] * (alg['T'][:,a,:].T @ b)
                b /= b.sum()
                pi = np.exp(alg['log_pi'](alg['tht'], 1, b)[1])
                y_score.append(pi)

        alg['m1'].append(metrics.brier_score_loss(y_true, y_score))
        alg['m2'].append(metrics.roc_auc_score(y_true, y_score))
        pre, rec, _ = metrics.precision_recall_curve(y_true, y_score)
        alg['m3'].append(metrics.auc(rec, pre))

    y_score = list()
    for traj in data:
        h = np.zeros(alg_rbc['H'])
        c = np.zeros(alg_rbc['H'])
        for a, z, a1 in zip(traj['a'], traj['z'], traj['a'][1:]):
            x = np.concatenate((jax.nn.one_hot(a, A), jax.nn.one_hot(z, Z)))
            f = jax.nn.sigmoid(alg_rbc['W_f'] @ x + alg_rbc['U_f'] @ h + alg_rbc['b_f'])
            i = jax.nn.sigmoid(alg_rbc['W_i'] @ x + alg_rbc['U_i'] @ h + alg_rbc['b_i'])
            o = jax.nn.sigmoid(alg_rbc['W_o'] @ x + alg_rbc['U_o'] @ h + alg_rbc['b_o'])
            c = f * c + i * np.tanh(alg_rbc['W_c'] @ x + alg_rbc['U_c'] @ h + alg_rbc['b_c'])
            h = o * np.tanh(c)
            l = np.tanh(alg_rbc['W_l'] @ h + alg_rbc['b_l'])
            pi = np.array(jax.nn.softmax(alg_rbc['W_y'] @ l + alg_rbc['b_y']))[1]
            y_score.append(pi)

    m1_rbc.append(metrics.brier_score_loss(y_true, y_score))
    m2_rbc.append(metrics.roc_auc_score(y_true, y_score))
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    m3_rbc.append(metrics.auc(rec, pre))

for alg in algs:
    alg['m1'] = np.array(alg['m1'])
    alg['m2'] = np.array(alg['m2'])
    alg['m3'] = np.array(alg['m3'])
m1_rbc = np.array(m1_rbc)
m2_rbc = np.array(m2_rbc)
m3_rbc = np.array(m3_rbc)

print('alg: calibration, AU-ROC, AU-PR')
for alg in algs:
    print('{}: {} ({}), {} ({}), {} ({})'.format(alg['name'], alg['m1'].mean(), alg['m1'].std(), alg['m2'].mean(), alg['m2'].std(), alg['m3'].mean(), alg['m3'].std()))
print('rbc: {} ({}), {} ({}), {} ({})'.format(m1_rbc.mean(), m1_rbc.std(), m2_rbc.mean(), m2_rbc.std(), m3_rbc.mean(), m3_rbc.std()))
