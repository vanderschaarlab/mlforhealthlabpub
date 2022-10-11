import time
from collections import defaultdict
import numpy as np
import warnings

warnings.filterwarnings("ignore")

''' 
-------------------------------------------------------------------------
SYNTHETIC EXPERIMENTS
--------------------------------------------------------------------------
'''

def run_synthetic_experiments():

    SAMPLE_NUMS = 50

    methods = [RMD]
    num_obs = [5]
    variances = [0.1]
    sample_size = [50]
    alpha = 0.05
    RESULTS = defaultdict(int)


    for size in sample_size:
        for obs in num_obs:
            for var in variances:
                for n_sample in range(SAMPLE_NUMS):
                    # Create a conditionally dependent and a conditionally
                    # independent version of the dataset.

                    t1, y1 =  generate_random_functions(size=size, num_obs = obs, function= 'zero',
                                                        variance = var)
                    t2, y2 =  generate_random_functions(size=size, num_obs = obs, function= 'zero',
                                                        variance = var)

                    if n_sample % 100 == 0:
                        print('=' * 70)
                        print('Experiment:', n_sample+1)
                        print('=' * 70)

                    # Run the two sample tests.
                    for method in methods:
                        method_name = method.__name__
                        key = 'method: {}; sample size: {}; num_obs: {}; variance: {} '.format(method_name, size, obs, var)
                        tic = time.time()
                        pval = method(t1,y1,t2,y2)
                        toc = (time.time() - tic)

                        RESULTS[key] += int(pval < alpha) / SAMPLE_NUMS

                        #joblib.dump(RESULTS, SAVE_FNAME)
                        if n_sample % 100 == 0:
                            print('{}: time={:.2}s, p={:.4}.'.format( method_name, toc, pval))
    return RESULTS

RESULTS_H0 = run_synthetic_experiments()

''' -------------------------------------------------------------------------
CLIMATE EXPERIMENTS
--------------------------------------------------------------------------
'''

def get_climate_times(year_period=20):
    times = np.zeros(12)  # placeholder for observation times
    for i in range(year_period):
        t = np.linspace(0, 1, 12)
        times = np.vstack((times, t))
    return times[1:]

def run_climate_experiments():
    """
    countries options: ['US','UK','VT','PL','TZ','CA', 'NG']
    :return: p-values
    """
    RESULTS = defaultdict(int)
    countries = ['US','UK','VT']
    num_test_by_country = 7 # this is the number of 20 year intervals we are considering
    methods = [RMD]

    for country in countries:
        for n_test in range(num_test_by_country):

            t = get_climate_times()
            y1, y2 = generate_climate_samples(country)
            y2 = y2[n_test]

            if n_test % 10 == 0:
                print('=' * 70)
                print('Experiment:', n_test)
                print('=' * 70)

            # Run the two sample tests.
            for method in methods:
                method_name = method.__name__
                key = 'Method: {}; Country: {}; Time period: {} '.format(method_name, country, n_test)
                tic = time.time()
                pval = method(t,y1,t,y2)
                toc = (time.time() - tic)

                RESULTS[key] = pval

                #joblib.dump(RESULTS, SAVE_FNAME)
                if n_test % 10 == 0:
                    print('{}: time={:.2}s, p={:.4}.'.format( method_name, toc, pval))
    return RESULTS


Climate_Results = run_climate_experiments()

def peru_experiments():
    """
    countries options: ['US','UK','VT','PL','TZ','CA', 'NG','PR]
    :return: p-values
    """
    RESULTS = defaultdict(int)
    n_samples = [20,40,60,80]
    methods = [RMD]
    num_iter = 10

    for n_sample in n_samples:
        for iter in range(num_iter):

            t1, t2 = get_climate_times(year_period=20), get_climate_times(year_period=n_sample)
            y1, y2 = generate_peru_samples(n_samples=n_sample)

            # Run the two sample tests.
            for method in methods:
                method_name = method.__name__
                key = 'Method: {}; Num of samples: {} '.format(method_name, n_sample)
                tic = time.time()
                pval = method(t1,y1,t2,y2)
                toc = (time.time() - tic)

                RESULTS[key] += pval/ num_iter

                #joblib.dump(RESULTS, SAVE_FNAME)
        print('{}: time={:.2}s, p={:.4}.'.format( method_name, toc, pval))
    return RESULTS


''' 
-------------------------------------------------------------------------
CF EXPERIMENTS
--------------------------------------------------------------------------
'''



def run_CF_experiments():

    RESULTS = defaultdict(int)
    methods = [RMD]
    features = ['Gender']

    for feature in features:

        t0, y0, t1, y1 = generate_CF_samples(outcome = 'FEV', feature = feature)

        # Run the two sample tests.
        for method in methods:
            method_name = method.__name__
            key = 'method: {}; expermient: {} '.format(method_name, feature)
            tic = time.time()
            pval = method(t0,y0,t1,y1)
            toc = (time.time() - tic)

            RESULTS[key] = pval

            print('{}: time={:.2}s, p={:.4}.'.format( method_name, toc, pval))

    return RESULTS