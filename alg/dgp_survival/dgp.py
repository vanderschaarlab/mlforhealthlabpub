from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
from utils.metrics import evaluate_performance
from model.DeepGP import DGPSurv
import pandas as pd


def run_experiment(
        prediction_horizon,
        n_epochs,
        n_iters,
        batch_size,
        lr,
        n_causes,
        layer_dim,
        n_inducing,
        fn_csv,
        event_col_nm,
        time_col_nm):

    df = pd.read_csv(fn_csv)

    train_size = int(0.7 * len(df))

    train_indexes = np.random.choice(
        list(range(len(df))),
        train_size,
        replace=False)
    test_indexes = np.array(
        list(set(list(range(len(df)))) - set(train_indexes)))

    df_train = df.iloc[train_indexes]
    df_test = df.iloc[test_indexes]

    lbl_T = time_col_nm
    lbl_c = event_col_nm

    features = list(df.columns)
    features.remove(lbl_T)
    features.remove(lbl_c)

    X_train = df_train[features]
    T_train = df_train[lbl_T]
    c_train = df_train[lbl_c]

    X_test = df_test[features]
    T_test = df_test[lbl_T]
    c_test = df_test[lbl_c]

    print('train: {} {}  {} test: {} {} {}'.format(
        X_train.shape, T_train.shape, c_train.shape,
        X_test.shape, T_test.shape, c_test.shape))

    deepgp = DGPSurv(X_train, T_train, c_train, prediction_horizon=prediction_horizon, 
                     layer_dim=layer_dim, num_inducing=n_inducing, num_causes=n_causes)

    deepgp.train(num_epochs=n_epochs, num_iters=n_iters, batch_size=batch_size, learning_rate=lr)

    y_pred = deepgp.predict_survival(X_test)
    evaluate_performance(T_train, c_train, T_test, c_test, 1 - y_pred, prediction_horizon, 
                         num_causes=n_causes)


def main(args):

    prediction_horizon = args.horizon
    n_epochs           = args.n_epochs
    n_iters            = args.n_iters
    batch_size         = args.batch_size
    lr                 = args.lr
    n_causes           = args.n_causes
    layer_dim          = args.layer_dim
    n_inducing         = args.n_inducing
    fn_csv             = args.i
    event_nm           = args.target
    time_nm            = args.time

    run_experiment(
        prediction_horizon,
        n_epochs,
        n_iters,
        batch_size,
        lr,
        n_causes,
        layer_dim,
        n_inducing,
        fn_csv,
        event_nm,
        time_nm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Deep Gaussian processes for survival analysis")

    parser.add_argument("-z", "--horizon", default=5*365, type=int, help="event horizon in days")
    parser.add_argument("-n", "--n-epochs", default=5, type=int)
    parser.add_argument("-t", "--n-iters", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=1000, type=int)
    parser.add_argument("-lr", "--lr", default=0.01, type=float)
    parser.add_argument("-c", "--n-causes", default=1, type=int)
    parser.add_argument("-d", "--layer-dim", default=5, type=int)
    parser.add_argument("--n-inducing", default=100, type=int)
    parser.add_argument("-i", help='input data in csv format')
    parser.add_argument(
        "--target",
        help='name of event column when using csv as input')
    parser.add_argument(
        "--time",
        help='name of event time in days when using csv as input')
    args = parser.parse_args()

    main(args)
