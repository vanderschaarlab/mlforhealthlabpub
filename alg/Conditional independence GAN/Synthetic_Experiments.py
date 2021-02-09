""" Experiments of paper: Conditional independence testing using generative adversarial networks """


import time
from collections import defaultdict
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Choose the sample numbers we will iterate over.
SAMPLE_NUMS = 50


methods = [CRT,GCIT]
z_dim = [50]
alphas = [0.05]



'''
Baseline method computation
'''

RESULTS = defaultdict(list)
type_I_error = defaultdict(int)
type_II_error = defaultdict(int)

#SAVE_FNAME = os.path.join(
#    'independence_test', 'saved_data', dset, 'tmp')
# '{}.pkl'.format(method_name))

for dim in z_dim:
    for alpha in alphas:
        for n_sample in range(SAMPLE_NUMS):
            # Create a conditionally dependent and a conditionally
            # independent version of the dataset.
            np.random.seed(n_sample)
            xd, yd, zd = generate_samples_random(size=100, sType='NI', dx=1, dy=1, dz=dim,
                                                 fixed_function='nonlinear', seed=n_sample,dist_z='laplace')
            xi, yi, zi = generate_samples_random(size=100, sType='CI', dx=1, dy=1, dz=dim,
                                                 fixed_function='nonlinear', seed=n_sample, dist_z = 'laplace')

            if n_sample % 10 == 0:
                print('=' * 70)
                print('Dimension = ', dim, 'Experiment:', n_sample+1)
                print('=' * 70)

            # Run the tests on conditionally-dependent and independent data.
            for method in methods:
                method_name = method.__name__
                key = 'method: {}; num exp: {}; dim of z: {}; alpha: {} '.format(method_name, n_sample, dim, alpha)
                key2 = 'method: {}; dim of z: {}; alpha: {} '.format(method_name, dim, alpha)
                tic = time.time()
                pval_d = method(xd, yd, zd)
                pval_i = method(xi, yi, zi)
                toc = (time.time() - tic) / 2.
                RESULTS[key].append((pval_d, pval_i, toc))

                type_I_error[key2] += int(pval_i < alpha) / SAMPLE_NUMS
                type_II_error[key2] += int(pval_d > alpha) / SAMPLE_NUMS

                #joblib.dump(RESULTS, SAVE_FNAME)
                if n_sample % 100 == 0:
                    print('{}: time={:.2}s, p_d={:.4}, p_i={:.4}.'.format( method_name, toc, pval_d, pval_i))
