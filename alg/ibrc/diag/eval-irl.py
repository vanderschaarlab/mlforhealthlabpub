import dill
import numpy as np

print('cost-benefit ratio')

with open('diag/res/irl-bet.obj', 'rb') as f:
    upsilon = dill.load(f)
    ratios = np.concatenate((upsilon[:,0,1]/upsilon[:,0,0], upsilon[:,1,0]/upsilon[:,1,1]))
    print('  for irl-bet: {} ({})'.format(ratios.mean(), ratios.std()))

with open('diag/res/irl-eta.obj', 'rb') as f:
    upsilon = dill.load(f)
    ratios = np.concatenate((upsilon[:,0,1]/upsilon[:,0,0], upsilon[:,1,0]/upsilon[:,1,1]))
    print('  for irl-eta: {} ({})'.format(ratios.mean(), ratios.std()))
