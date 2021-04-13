"""
Author: Alicia Curth
Script to run experiments on Johansson's IHDP dataset (retrieved via https://www.fredjo.com/)
"""
import os
import numpy as onp
import csv

from sklearn import clone

from catenets.experiments.experiment_utils import eval_root_mse, get_model_set

from catenets.models import PSEUDOOUT_NAME, PseudoOutcomeNet
from catenets.models.transformation_utils import RA_TRANSFORMATION

# Some constants
DATA_DIR = 'data/'
IHDP_TRAIN_NAME = 'ihdp_npci_1-100.train.npz'
IHDP_TEST_NAME = 'ihdp_npci_1-100.test.npz'
RESULT_DIR = 'results/ihdp/'
SEP = '_'

# Hyperparameters for experiments on IHDP
LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL_IHDP = 0

MODEL_PARAMS = {'n_layers_out': LAYERS_OUT, 'n_layers_r': LAYERS_R, 'penalty_l2': PENALTY_L2,
                'penalty_orthogonal': PENALTY_ORTHOGONAL_IHDP, 'n_layers_out_t': LAYERS_OUT,
                'n_layers_r_t': LAYERS_R, 'penalty_l2_t': PENALTY_L2}

# get basic models
ALL_MODELS_IHDP = get_model_set(model_selection='all', model_params=MODEL_PARAMS)

COMBINED_MODELS_IHDP = {PSEUDOOUT_NAME + SEP + RA_TRANSFORMATION + SEP + 'S2':
                            PseudoOutcomeNet(n_layers_r=LAYERS_R, n_layers_out=LAYERS_OUT,
                                             penalty_l2=PENALTY_L2, n_layers_r_t=LAYERS_R,
                                             n_layers_out_t=LAYERS_OUT, penalty_l2_t=PENALTY_L2,
                                             transformation=RA_TRANSFORMATION, first_stage_strategy='S2')}

FULL_MODEL_SET_IHDP = dict(**ALL_MODELS_IHDP, **COMBINED_MODELS_IHDP)


def do_ihdp_experiments(n_exp: int = 100, file_name: str = 'ihdp_results_scaled',
                        model_params: dict = None, scale_cate: bool = True,
                        models: dict = None):
    if models is None:
        models = FULL_MODEL_SET_IHDP
    elif type(models) is list or type(models) is str:
        models = get_model_set(models)

    # make path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # get file to write in
    out_file = open(RESULT_DIR + file_name + '.csv', 'w', buffering=1)
    writer = csv.writer(out_file)
    header = [name + '_in' for name in models.keys()] + [name + '_out' for name in models.keys()]
    writer.writerow(header)

    # get data
    data_train = load_data_npz(DATA_DIR + IHDP_TRAIN_NAME, get_po=True)
    data_test = load_data_npz(DATA_DIR + IHDP_TEST_NAME, get_po=True)

    if isinstance(n_exp, int):
        experiment_loop = range(1, n_exp + 1)
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError('n_exp should be either an integer or a list of integers.')

    for i_exp in experiment_loop:
        pehe_in = []
        pehe_out = []

        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=i_exp, get_po=True)

        X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(data_exp, data_exp_test,
                                                                      rescale=scale_cate)

        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)
            if model_params is not None:
                estimator_temp.set_params(**model_params)

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            cate_pred_in = estimator_temp.predict(X, return_po=False)
            cate_pred_out = estimator_temp.predict(X_t, return_po=False)

            pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
            pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))

        writer.writerow(pehe_in + pehe_out)

    out_file.close()


def prepare_ihdp_data(data_train, data_test, rescale: bool = True, return_pos=False):
    """ Prepare data"""

    X, y, w, mu0, mu1 = data_train['X'], data_train['y'], data_train['w'], data_train['mu0'], \
                        data_train['mu1']

    X_t, y_t, w_t, mu0_t, mu1_t = data_test['X'], data_test['y'], data_test['w'], \
                                  data_test['mu0'], data_test['mu1']

    if rescale:
        # rescale all outcomes to have similar scale of CATEs if sd_cate > 1
        cate_in = mu0 - mu1
        sd_cate = onp.sqrt(cate_in.var())

        if sd_cate > 1:
            # training data
            error = y - w * mu1 - (1 - w) * mu0
            mu0 = mu0 / sd_cate
            mu1 = mu1 / sd_cate
            y = w * mu1 + (1 - w) * mu0 + error

            # test data
            mu0_t = mu0_t / sd_cate
            mu1_t = mu1_t / sd_cate

    cate_true_in = mu1 - mu0
    cate_true_out = mu1_t - mu0_t

    if return_pos:
        return X, y, w, cate_true_in, X_t, cate_true_out, mu0, mu1, mu0_t, mu1_t

    return X, y, w, cate_true_in, X_t, cate_true_out


# helper functions
def load_data_npz(fname, get_po: bool = True):
    """ Load data set (adapted from https://github.com/clinicalml/cfrnet)"""
    if fname[-3:] == 'npz':
        data_in = onp.load(fname)
        data = {'X': data_in['x'], 'w': data_in['t'], 'y': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        raise ValueError('This loading function is only for npz files.')

    if get_po:
        data['mu0'] = data_in['mu0']
        data['mu1'] = data_in['mu1']

    data['HAVE_TRUTH'] = not data['ycf'] is None
    data['dim'] = data['X'].shape[1]
    data['n'] = data['X'].shape[0]

    return data


def get_one_data_set(D, i_exp, get_po: bool = True):
    """ Get data for one experiment. Adapted from https://github.com/clinicalml/cfrnet"""
    D_exp = {}
    D_exp['X'] = D['X'][:, :, i_exp - 1]
    D_exp['w'] = D['w'][:, i_exp - 1:i_exp]
    D_exp['y'] = D['y'][:, i_exp - 1:i_exp]
    if D['HAVE_TRUTH']:
        D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
    else:
        D_exp['ycf'] = None

    if get_po:
        D_exp['mu0'] = D['mu0'][:, i_exp - 1:i_exp]
        D_exp['mu1'] = D['mu1'][:, i_exp - 1:i_exp]

    return D_exp
