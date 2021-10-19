import dill
import numpy as np
import pandas as pd

# creates a list of diagnostic states
state = ['NL', 'MCI', 'Dementia']
state += ['{} to {}'.format(s0, s1) for s0 in state for s1 in state if s0 != s1]

# assigns a unique integer to each state
state_dict = {s:i for s,i in zip(state,range(len(state)))}

# there are 9 states in total:
# 0: Normal function (NL)
# 1: Mild cognitive impairment (MCI)
# 2: Dementia
# 3: NL to MCI
# 4: NL to Dementia
# 5: MCI to NL
# 6: MCI to Dementia
# 7: Dementia to NL
# 8: Dementia to MCI

df = pd.read_csv('adni/data/adni.csv', low_memory=False)
df = df[~df.DX.isna()]      # removes enteries with missing diagnostic states
df = df[~df.CDRSB.isna()]   # removes enteries with missing CDR-SB scores

# creates a list of possible viscodes (bl, m06, ml12, ...)
# viscodes mark when a hospital visit has occured
visc = ['bl', 'm06'] + ['m{}'.format(k*6) for k in range(2,20)]

# creates a separate list of rids (patient identifiers) for each viscode
# rids[0] consists of patients that visited the hospital at month 0
# rids[1] consists of patients that visited the hospital at months 0 and 6
# rids[2] consists of patients that visited the hospital at months 0, 6, and 12
# ...
rids = [df[df.VISCODE == vis].RID.unique() for vis in visc]
for i in range(1,len(rids)):
    rids[i] = [rid for rid in rids[i] if rid in rids[i-1]]

# removes visits that occured after a missing hospital visit
# (i.e. if the patient did not visit the hospital at month 12, visits at months 18, 24, ... are removed)
# this ensures that all trajectories are complete with no missing time steps
df = df[df.VISCODE.isin(visc)]
for vis, rid in zip(visc, rids):
    df = df[df.RID.isin(rid) | (df.VISCODE != vis)]

# list of state-action-observation trajectories
data = list()

# dictionary for meta data corresponding to each trajectory
# (e.g. data[0] is the action-observation trajectory corresponding to patient 0, data_meta['age'][0] is their age)
data_meta = dict()
data_meta['age'] = list()
data_meta['apoe'] = list()
data_meta['gender'] = list()

for rid in rids[0]:

    traj = dict()
    traj['s'] = list([None])    # states (note that initial diagnostic state is unknown)
    traj['a'] = list()          # actions
    traj['z'] = list()          # observations
    traj['tau'] = 0             # time horizon (i.e. len(traj['a']))

    df1 = df[df.RID == rid]
    for vis in visc:

        df2 = df1[df1.VISCODE == vis]
        if df2.empty:
            break

        # diagnostic state is encoded using the dictionary created earlier
        s = state_dict[df2.DX.values[0]]

        # the action is whether an MRI test is ordered (a = 1) or not (a = 0)
        # (note that the measurement for the hippocampus volume would be missing when an MRI is not ordered)
        a = 0 if df2.Hippocampus.isna().values[0] else 1

        # continuous measurements for the hippocampus volume are discretized and encoded together with the CDR-SB score
        z0 = 0 if df2.CDRSB.values[0] == 0 else 1 if df2.CDRSB.values[0] <= 2.5 else 2
        z1 = 0 if df2.Hippocampus.isna().values[0] else 1 if df2.Hippocampus.values[0] < 6642-.5*1225 else 2 if df2.Hippocampus.values[0] <= 6642+.5*1225 else 3
        z = 4 * z0 + z1

        # there are 12 possible observations:
        # 0: the CDR-SB score indicates normal function, the hippocampus volume is not measured
        # 1: the CDR-SB score indicates normal function, the hippocampus volume is measured to be below average
        # 2: the CDR-SB score indicates normal function, the hippocampus volume is measured to be average
        # 3: the CDR-SB score indicates normal function, the hippocampus volume is measured to be above average
        # 4: the CDR-SB score indicates qustionable impairment, the hippocampus volume is not measured
        # 5: the CDR-SB score indicates qustionable impairment, the hippocampus volume is measured to be below average
        # 6: the CDR-SB score indicates qustionable impairment, the hippocampus volume is measured to be average
        # 7: the CDR-SB score indicates qustionable impairment, the hippocampus volume is measured to be above average
        # 8: the CDR-SB score indicates mild to severe dementia, the hippocampus volume is not measured
        # 9: the CDR-SB score indicates mild to severe dementia, the hippocampus volume is measured to be below average
        # 10: the CDR-SB score indicates mild to severe dementia, the hippocampus volume is measured to be average
        # 11: the CDR-SB score indicates mild to severe dementia, the hippocampus volume is measured to be above average

        traj['s'].append(s)
        traj['a'].append(a)
        traj['z'].append(z)
        traj['tau'] += 1

    # a trajectory is added to the processed dataset only if it ends with a stable diagnosis (NL, MCI, or Dementia)
    if traj['s'][-1] == 0 or traj['s'][-1] == 1 or traj['s'][-1] == 2:
        data.append(traj)

        df2 = df1[df1.VISCODE == 'bl']
        data_meta['age'].append(df2.AGE.values[0])
        data_meta['apoe'].append(df2.APOE4.values[0])
        data_meta['gender'].append(df2.PTGENDER.values[0])

        print('n = {}, tau = {}'.format(len(data), traj['tau']))

with open('adni/data/adni.obj', 'wb') as f:
    dill.dump(data, f)
with open('adni/data/adni-meta.obj', 'wb') as f:
    dill.dump(data_meta, f)
