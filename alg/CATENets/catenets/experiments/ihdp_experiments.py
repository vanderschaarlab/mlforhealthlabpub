"""
Author: Alicia Curth
Script to run experiments on Johansson's IHDP dataset (retrieved via https://www.fredjo.com/)
"""
import csv
import os
from pathlib import Path
from typing import Optional, Union

import catenets.logger as log
from catenets.datasets.dataset_ihdp import get_one_data_set, load_raw, prepare_ihdp_data
from catenets.experiments.experiment_utils import eval_root_mse, get_model_set
from catenets.models import PSEUDOOUT_NAME, PseudoOutcomeNet
from catenets.models.transformation_utils import RA_TRANSFORMATION
from sklearn import clone

# Some constants
DATA_DIR = Path("data/")
RESULT_DIR = Path("results/ihdp/")
SEP = "_"

# Hyperparameters for experiments on IHDP
LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL_IHDP = 0

MODEL_PARAMS = {
    "n_layers_out": LAYERS_OUT,
    "n_layers_r": LAYERS_R,
    "penalty_l2": PENALTY_L2,
    "penalty_orthogonal": PENALTY_ORTHOGONAL_IHDP,
    "n_layers_out_t": LAYERS_OUT,
    "n_layers_r_t": LAYERS_R,
    "penalty_l2_t": PENALTY_L2,
}

# get basic models
ALL_MODELS_IHDP = get_model_set(model_selection="all", model_params=MODEL_PARAMS)

COMBINED_MODELS_IHDP = {
    PSEUDOOUT_NAME
    + SEP
    + RA_TRANSFORMATION
    + SEP
    + "S2": PseudoOutcomeNet(
        n_layers_r=LAYERS_R,
        n_layers_out=LAYERS_OUT,
        penalty_l2=PENALTY_L2,
        n_layers_r_t=LAYERS_R,
        n_layers_out_t=LAYERS_OUT,
        penalty_l2_t=PENALTY_L2,
        transformation=RA_TRANSFORMATION,
        first_stage_strategy="S2",
    )
}

FULL_MODEL_SET_IHDP = dict(**ALL_MODELS_IHDP, **COMBINED_MODELS_IHDP)


def do_ihdp_experiments(
    n_exp: Union[int, list] = 100,
    file_name: str = "ihdp_results_scaled",
    model_params: Optional[dict] = None,
    scale_cate: bool = True,
    models: Union[list, dict, str, None] = None,
) -> None:
    if models is None:
        models = FULL_MODEL_SET_IHDP
    elif isinstance(models, (list, str)):
        models = get_model_set(models)

    # make path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # get file to write in
    out_file = open(RESULT_DIR / (file_name + ".csv"), "w", buffering=1)
    writer = csv.writer(out_file)
    header = [name + "_in" for name in models.keys()] + [
        name + "_out" for name in models.keys()
    ]
    writer.writerow(header)

    # get data
    data_train, data_test = load_raw(DATA_DIR)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError("n_exp should be either an integer or a list of integers.")

    for i_exp in experiment_loop:
        pehe_in = []
        pehe_out = []

        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=i_exp, get_po=True)

        X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(
            data_exp, data_exp_test, rescale=scale_cate
        )

        for model_name, estimator in models.items():
            log.info(f"Experiment {i_exp} with {model_name}")
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
