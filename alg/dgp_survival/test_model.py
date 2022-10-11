from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np

from data.make_data import load_SEER_data
from utils.metrics import evaluate_performance
from model.DeepGP import DGPSurv
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def run_experiment(prediction_horizon, n_epochs, n_iters, batch_size, lr, n_causes, layer_dim, n_inducing):

    fn_csv = '{}/alg/dgp_survival/data/SEER_breast_cancer_cohort.csv'.format(
        utilmlab.get_proj_dir())

    (X_train, T_train, c_train), (X_test, T_test, c_test) = load_SEER_data(
        fn_csv)

    deepgp = DGPSurv(X_train, T_train, c_train, prediction_horizon=prediction_horizon, 
                     layer_dim=layer_dim, num_inducing=n_inducing, num_causes=n_causes)
    
    deepgp.train(num_epochs=n_epochs, num_iters=n_iters, batch_size=batch_size, learning_rate=lr)

    y_pred = deepgp.predict_survival(X_test)

    evaluate_performance(T_train, c_train, T_test, c_test, 1 - y_pred, prediction_horizon, 
                         num_causes=n_causes, cause_names=["Breast cancer", "Other causes "])


def main(args):
    
    prediction_horizon = 365 * args.horizon
    n_epochs           = args.n_epochs
    n_iters            = args.n_iters
    batch_size         = args.batch_size
    lr                 = args.lr
    n_causes           = args.n_causes
    layer_dim          = args.layer_dim
    n_inducing         = args.n_inducing
    
    run_experiment(prediction_horizon, n_epochs, n_iters, batch_size, lr, n_causes, layer_dim, n_inducing)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Deep Gaussian processes for survival analysis")

    parser.add_argument("-z", "--horizon", default=5, type=int)
    parser.add_argument("-n", "--n-epochs", default=5, type=int)
    parser.add_argument("-t", "--n-iters", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=1000, type=int)
    parser.add_argument("-lr", "--lr", default=0.01, type=float)
    parser.add_argument("-c", "--n-causes", default=2, type=int)
    parser.add_argument("-d", "--layer-dim", default=5, type=int)
    parser.add_argument("-i", "--n-inducing", default=100, type=int)

    args = parser.parse_args()
    
    main(args)    



# python test_model.py -z "horizon" -n "number of epochs" -t "number of iterations" -b "batch size" -lr "learning rate" -c "number of causes" -d "number of hidden dimensions" -i "number of inducing points"



