import dill
import numpy as np

with open('adni/res/res.obj', 'rb') as f:
    hypers0 = dill.load(f)
with open('adni/res/res-age.obj', 'rb') as f:
    hypers1 = dill.load(f)
with open('adni/res/res-apoe.obj', 'rb') as f:
    hypers2 = dill.load(f)
with open('adni/res/res-female.obj', 'rb') as f:
    hypers3 = dill.load(f)

print('beta for all:', hypers0.mean(axis=0)[1])
print('beta for age:', hypers1.mean(axis=0)[1])
print('beta for apoe:', hypers2.mean(axis=0)[1])
print('beta for female', hypers3.mean(axis=0)[1])
