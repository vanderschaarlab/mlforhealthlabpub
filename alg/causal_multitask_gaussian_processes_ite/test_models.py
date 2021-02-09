
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import argparse
import sys
import os
import json

from make_data import sample_IHDP
from utils.metrics import compute_PEHE, mean_confidence_interval
from models.causal_models import CMGP
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def run_experiment(fn_data, mode="CMGP", test_frac=0.1):
    
    train_data, test_data                   = sample_IHDP(fn_data, test_frac=test_frac)
    X_train, W_train, Y_train, T_true_train = train_data[0], train_data[1], train_data[2], train_data[6]
    X_test,  T_true_test                    = test_data[0], test_data[6] 
    
    model = CMGP(dim=25, mode=mode)

    model.fit(X_train, Y_train, W_train)
    
    TE_est_test  = model.predict(X_test)[0]
    TE_est_train = model.predict(X_train)[0]
    
    PEHE_train = compute_PEHE(TE_est_train, T_true_train)
    PEHE_test  = compute_PEHE(TE_est_test, T_true_test)
    
    return PEHE_train, PEHE_test



def main(args, fn_data):

    PEHE_train_ = [] 
    PEHE_test_ = []
    results_d = {}
 
    time_start = time.time()

    for _ in range(args.num_exp):
        
        pehe_train_curr, pehe_test_curr = run_experiment(fn_data, mode=args.mode, test_frac=args.test_frac)
        
        PEHE_train_.append(pehe_train_curr)
        PEHE_test_.append(pehe_test_curr)
    
        print("Experiment: %d (train) \tPEHE: %.3f \t--- (test) \tPEHE: %.3f \t---" % (_, pehe_train_curr, pehe_test_curr))    
    
        # on purpose results are calculated within the loop so intermediate results become available while the experiment is ongoing

        results_d['train'] = PEHE_train_
        results_d['test'] = PEHE_test_

        PEHE_train_np = np.array(PEHE_train_)[~np.isnan(np.array(PEHE_train_))]
        PEHE_test_np  = np.array(PEHE_test_)[~np.isnan(np.array(PEHE_test_))]

        time_exe = time.time() - time_start
    
        results_d['PEHE_train'] = mean_confidence_interval(PEHE_train_np)
        results_d['PEHE_test'] = mean_confidence_interval(PEHE_test_np)

        results_d['time_exe'] = time_exe
        if args.o is not None:
            with open(args.o, 'w') as fp:
                json.dump(results_d, fp)
        
    print('exe time {:0.0f}s'.format(time_exe))
    print("Final results|| Train PEHE = %.3f +/- %.3f --- Test PEHE = %.3f +/- %.3f" % (mean_confidence_interval(PEHE_train_np)[0],
                                                                                        mean_confidence_interval(PEHE_train_np)[1],
                                                                                        mean_confidence_interval(PEHE_test_np)[0],
                                                                                        mean_confidence_interval(PEHE_test_np)[1]))
    return results_d


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Causal Multi-task Gaussian Processes")

    parser.add_argument("-n", "--num-exp", default=10, type=int)
    parser.add_argument("-m", "--mode", default="CMGP", type=str)
    parser.add_argument("-t", "--test-frac", default=0.1, type=float)
    parser.add_argument("-o")

    args = parser.parse_args()

    fn_data = '{}/ihdp/ihdp_covariates.csv'.format(utilmlab.get_data_dir())

    if not os.path.isfile(fn_data):
        print('Error: this implementation requires the IHDP dataset'
              ', please refer to the README.md for more details.')
        sys.exit(0)

    results_d = main(args, fn_data)

    if args.o is not None:
        with open(args.o, 'w') as fp:
            json.dump(results_d, fp)
