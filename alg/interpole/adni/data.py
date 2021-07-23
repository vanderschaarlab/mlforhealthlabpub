import dill
import numpy as np
import pandas as pd

state = ['NL', 'MCI', 'Dementia']
state += ['{} to {}'.format(s0, s1) for s0 in state for s1 in state if s0 != s1]
state_dict = {s:i for s,i in zip(state,range(len(state)))}

df = pd.read_csv('data/adni.csv', low_memory=False)
df = df[~df.DX.isna()]
df = df[~df.CDRSB.isna()]

visc = ['bl', 'm06'] + ['m{}'.format(k*6) for k in range(2,20)]
rids = [df[df.VISCODE == vis].RID.unique() for vis in visc]
for i in range(1,len(rids)):
    rids[i] = [rid for rid in rids[i] if rid in rids[i-1]]

df = df[df.VISCODE.isin(visc)]
for vis, rid in zip(visc, rids):
    df = df[df.RID.isin(rid) | (df.VISCODE != vis)]

data = list()
for rid in rids[0]:

    traj = dict()
    traj['s'] = list([None])
    traj['a'] = list()
    traj['z'] = list()
    traj['tau'] = 0

    df1 = df[df.RID == rid]
    for vis in visc:

        df2 = df1[df1.VISCODE == vis]
        if df2.empty:
            break

        s = state_dict[df2.DX.values[0]]
        a = 0 if df2.Hippocampus.isna().values[0] else 1
        z0 = 0 if df2.CDRSB.values[0] == 0 else 1 if df2.CDRSB.values[0] <= 2.5 else 2
        z1 = 0 if df2.Hippocampus.isna().values[0] else 1 if df2.Hippocampus.values[0] < 6642-.5*1225 else 2 if df2.Hippocampus.values[0] <= 6642+.5*1225 else 3

        traj['s'].append(s)
        traj['a'].append(a)
        traj['z'].append(4*z0+z1)
        traj['tau'] += 1

    if traj['s'][-1] == 0 or traj['s'][-1] == 1 or traj['s'][-1] == 2:
        data.append(traj)
        print('n = {}, tau = {}'.format(len(data), traj['tau']))

with open('data/adni.obj', 'wb') as f:
    dill.dump(data, f)
