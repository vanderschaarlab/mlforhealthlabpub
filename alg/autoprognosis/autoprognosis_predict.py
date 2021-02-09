import pickle
import model
import argparse
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import initpath_ap
import json
initpath_ap.init_sys_path()
import utilmlab
import logging
import os


from rpy2.robjects import r
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()


def init_r_system():
    r('require("missForest")')
    r('require("MICE")')
    r('require("EMB")')
    r('require("Amelia")')
    r('require("matrix_completion")')
    r('require("softImpute")')


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--startsample", type=int)
    parser.add_argument("--target")
    parser.add_argument("--model")
    parser.add_argument(
        "--separator",
        default=',',
        help="separator to use when writing to csv file")
    parser.add_argument(
        "--evaluate",
        help="evaluate and write results to a json file")
    return parser.parse_args()


if __name__ == '__main__':

    init_r_system()
    
    args = init_arg()
    fn = args.model
    fn_i = args.i
    fn_o = args.o
    lbl = args.target
    fn_eva = args.evaluate
    start_sample = args.startsample
    sep = args.separator
    
    if fn_o is not None:
        logger = utilmlab.init_logger(
            os.path.dirname(fn_o),
            'log_predict_ap.txt')
    else:
        logger = logging.getLogger()

    eva = model.evaluate()
    eva.set_metric('aucprc')

    assert os.path.isfile(fn) and os.path.isfile(fn_i)
    
    with open(fn, "rb") as fp:
        AP_mdl = pickle.load(fp)

    df = pd.read_csv(fn_i, sep=sep)

    if start_sample is not None:
        df = df[start_sample:]
    features = list(df.columns)
    features.remove(lbl)
    X_ = df[features]
    Y_ = df[lbl]

    Y_hat0, Y_hat1 = AP_mdl.predict(X_)

    assert np.array_equal(Y_hat0, Y_hat1)

    if fn_eva is not None:
        eva_d = {
            'aucroc': eva.roc_auc_score(Y_, Y_hat0),
            'aucprc': eva.average_precision_score(Y_, Y_hat0),
            'accuracy': eva.accuracy_score(Y_, None, Y_hat0)
        }
        logger.info('evaluation:{}'.format(eva_d))
        with open(fn_eva, "w") as fp:
            json.dump(eva_d, fp)
        
    if fn_o is not None:
        logger.info('saving:{}'.format(fn_o))
        df = pd.DataFrame(
            Y_hat0,
            columns=['{}_{}'.format(lbl, idx) for idx in range(Y_hat0.shape[1])])
        df.to_csv(fn_o, sep=sep)
