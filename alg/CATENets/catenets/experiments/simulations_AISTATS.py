"""
Author: Alicia Curth
Script to generate synthetic simulations in AISTATS paper
"""
import csv
import os
from typing import Any, Optional, Union

from sklearn import clone

import catenets.logger as log
from catenets.experiments.experiment_utils import eval_root_mse, get_model_set
from catenets.experiments.simulation_utils import simulate_treatment_setup
from catenets.models import PSEUDOOUT_NAME, PseudoOutcomeNet
from catenets.models.pseudo_outcome_nets import S1_STRATEGY, S_STRATEGY
from catenets.models.snet import DEFAULT_UNITS_R_BIG_S, DEFAULT_UNITS_R_SMALL_S
from catenets.models.transformation_utils import DR_TRANSFORMATION, RA_TRANSFORMATION

# some constants
RESULT_DIR = "results/simulations/"
CSV_STRING = ".csv"
SEP = "_"

# hyperparameters for experiments
LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL = 1 / 100

MODEL_PARAMS_AISTATS = {
    "n_layers_out": LAYERS_OUT,
    "n_layers_r": LAYERS_R,
    "penalty_l2": PENALTY_L2,
    "penalty_orthogonal": PENALTY_ORTHOGONAL,
    "n_layers_out_t": LAYERS_OUT,
    "n_layers_r_t": LAYERS_R,
    "penalty_l2_t": PENALTY_L2,
}

# get basic models
ALL_MODELS_AISTATS = get_model_set(
    model_selection="all", model_params=MODEL_PARAMS_AISTATS
)

# model-twostep combinations
COMBINED_MODELS = {
    PSEUDOOUT_NAME
    + SEP
    + DR_TRANSFORMATION
    + SEP
    + S_STRATEGY: PseudoOutcomeNet(
        transformation=DR_TRANSFORMATION,
        first_stage_strategy=S_STRATEGY,
        n_units_r=DEFAULT_UNITS_R_BIG_S,
        n_units_r_small=DEFAULT_UNITS_R_SMALL_S,
        n_layers_out=LAYERS_OUT,
        n_layers_r=LAYERS_R,
        penalty_l2_t=PENALTY_L2,
        penalty_l2=PENALTY_L2,
        n_layers_out_t=LAYERS_OUT,
        n_layers_r_t=LAYERS_R,
        penalty_orthogonal=PENALTY_ORTHOGONAL,
    ),
    PSEUDOOUT_NAME
    + SEP
    + RA_TRANSFORMATION
    + SEP
    + S_STRATEGY: PseudoOutcomeNet(
        transformation=RA_TRANSFORMATION,
        first_stage_strategy=S_STRATEGY,
        n_units_r=DEFAULT_UNITS_R_BIG_S,
        n_units_r_small=DEFAULT_UNITS_R_SMALL_S,
        penalty_orthogonal=PENALTY_ORTHOGONAL,
        n_layers_out=LAYERS_OUT,
        n_layers_r=LAYERS_R,
        penalty_l2_t=PENALTY_L2,
        penalty_l2=PENALTY_L2,
        n_layers_out_t=LAYERS_OUT,
        n_layers_r_t=LAYERS_R,
    ),
    PSEUDOOUT_NAME
    + SEP
    + DR_TRANSFORMATION
    + SEP
    + S1_STRATEGY: PseudoOutcomeNet(
        transformation=DR_TRANSFORMATION,
        first_stage_strategy=S1_STRATEGY,
        n_layers_out=LAYERS_OUT,
        n_layers_r=LAYERS_R,
        penalty_l2_t=PENALTY_L2,
        penalty_l2=PENALTY_L2,
        n_layers_out_t=LAYERS_OUT,
        n_layers_r_t=LAYERS_R,
    ),
    PSEUDOOUT_NAME
    + SEP
    + RA_TRANSFORMATION
    + SEP
    + S1_STRATEGY: PseudoOutcomeNet(
        transformation=RA_TRANSFORMATION,
        first_stage_strategy=S1_STRATEGY,
        n_layers_out=LAYERS_OUT,
        n_layers_r=LAYERS_R,
        penalty_l2_t=PENALTY_L2,
        penalty_l2=PENALTY_L2,
        n_layers_out_t=LAYERS_OUT,
        n_layers_r_t=LAYERS_R,
    ),
}

FULL_MODEL_SET_AISTATS = dict(**ALL_MODELS_AISTATS, **COMBINED_MODELS)

# some more constants for experiments
NTRAIN_BASE = 2000
NTEST_BASE = 500
D_BASE = 25
BASE_XI = 3
TARGET_PROP_BASE = None

XI_STRING = "xi"
N_STRING = "n"
D_T_STRING = "dim_t"
PROPENSITY_CONSTANT_STRING = "p"
TARGET_STRING = "target_p"


def simulation_experiment_loop(
    range_change: list,
    change_dim: str = N_STRING,
    n_train: int = NTRAIN_BASE,
    n_test: int = NTEST_BASE,
    n_repeats: int = 10,
    d: int = D_BASE,
    n_w: int = 0,
    n_c: int = 5,
    n_o: int = 5,
    n_t: int = 0,
    file_base: str = "results",
    xi: float = BASE_XI,
    mu_1_model: Any = None,
    correlated_x: bool = False,
    mu_1_model_params: Optional[dict] = None,
    mu_0_model_params: Optional[dict] = None,
    models: Any = None,
    nonlinear_prop: bool = True,
    prop_offset: Union[float, str] = "center",
    target_prop: Optional[float] = TARGET_PROP_BASE,
) -> None:
    if change_dim is N_STRING:
        for n in range_change:
            log.info(f"Running experiments for {N_STRING} set to {n}")
            do_one_experiment_repeat(
                n_train=n,
                n_test=n_test,
                n_repeats=n_repeats,
                d=d,
                n_w=n_w,
                n_c=n_c,
                n_o=n_o,
                n_t=n_t,
                file_base=file_base,
                xi=xi,
                mu_1_model=mu_1_model,
                correlated_x=correlated_x,
                models=models,
                mu_1_model_params=mu_1_model_params,
                mu_0_model_params=mu_0_model_params,
                nonlinear_prop=nonlinear_prop,
                prop_offset=prop_offset,
                target_prop=target_prop,
            )
    elif change_dim is XI_STRING:
        for xi_temp in range_change:
            log.info(f"Running experiments for {XI_STRING} set to {xi_temp}")
            do_one_experiment_repeat(
                n_train=n_train,
                n_test=n_test,
                n_repeats=n_repeats,
                d=d,
                n_w=n_w,
                n_c=n_c,
                n_o=n_o,
                n_t=n_t,
                file_base=file_base,
                xi=xi_temp,
                mu_1_model=mu_1_model,
                correlated_x=correlated_x,
                models=models,
                mu_1_model_params=mu_1_model_params,
                mu_0_model_params=mu_0_model_params,
                nonlinear_prop=nonlinear_prop,
                prop_offset=prop_offset,
                target_prop=target_prop,
            )

    elif change_dim is D_T_STRING:
        for d_t_temp in range_change:
            log.info(f"Running experiments for {D_T_STRING} set to {d_t_temp}")
            do_one_experiment_repeat(
                n_train=n_train,
                n_test=n_test,
                n_repeats=n_repeats,
                d=d,
                n_w=n_w,
                n_c=n_c,
                n_o=n_o,
                n_t=d_t_temp,
                file_base=file_base,
                xi=xi,
                mu_1_model=mu_1_model,
                correlated_x=correlated_x,
                models=models,
                mu_1_model_params=mu_1_model_params,
                mu_0_model_params=mu_0_model_params,
                nonlinear_prop=nonlinear_prop,
                prop_offset=prop_offset,
                target_prop=target_prop,
            )

    elif change_dim is TARGET_STRING:
        for target_prop_temp in range_change:
            log.info(
                f"Running experiments for {TARGET_STRING} set to {target_prop_temp}"
            )
            do_one_experiment_repeat(
                n_train=n_train,
                n_test=n_test,
                n_repeats=n_repeats,
                d=d,
                n_w=n_w,
                n_c=n_c,
                n_o=n_o,
                n_t=n_t,
                file_base=file_base,
                xi=xi,
                mu_1_model=mu_1_model,
                correlated_x=correlated_x,
                models=models,
                mu_1_model_params=mu_1_model_params,
                mu_0_model_params=mu_0_model_params,
                nonlinear_prop=nonlinear_prop,
                prop_offset=prop_offset,
                target_prop=target_prop_temp,
            )


def do_one_experiment_repeat(
    n_train: int = NTRAIN_BASE,
    n_test: int = NTEST_BASE,
    n_repeats: int = 10,
    d: int = D_BASE,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    file_base: str = "results",
    xi: float = BASE_XI,
    mu_1_model: Any = None,
    correlated_x: bool = True,
    mu_1_model_params: Optional[dict] = None,
    mu_0_model_params: Optional[dict] = None,
    models: Any = None,
    nonlinear_prop: bool = True,
    range_exp: Optional[list] = None,
    prop_offset: Union[float, str] = 0,
    target_prop: Optional[float] = None,
) -> None:
    # make path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if range_exp is None:
        range_exp = list(range(1, n_repeats + 1))

    if models is None:
        models = FULL_MODEL_SET_AISTATS

    if target_prop is None:
        prop_string = str(prop_offset)
    else:
        prop_string = str(target_prop)

    # create file name and file
    file_name = (
        file_base
        + SEP
        + str(n_train)
        + SEP
        + str(d)
        + SEP
        + str(n_w)
        + SEP
        + str(n_c)
        + SEP
        + str(n_o)
        + SEP
        + str(n_t)
        + SEP
        + str(xi)
        + SEP
        + prop_string
    )

    out_file = open(RESULT_DIR + file_name + CSV_STRING, "w", buffering=1)
    writer = csv.writer(out_file)
    header = [name for name in models.keys()]
    writer.writerow(header)

    for i in range_exp:
        log.info(f"Running experiment {i}.")
        mses = one_simulation_experiment(
            n_train=n_train,
            n_test=n_test,
            d=d,
            n_w=n_w,
            n_c=n_c,
            n_o=n_o,
            n_t=n_t,
            seed=i,
            xi=xi,
            mu_1_model=mu_1_model,
            correlated_x=correlated_x,
            models=models,
            nonlinear_prop=nonlinear_prop,
            mu_0_model_params=mu_0_model_params,
            mu_1_model_params=mu_1_model_params,
            prop_offset=prop_offset,
            target_prop=target_prop,
        )
        writer.writerow(mses)

    out_file.close()
    return None


def one_simulation_experiment(
    n_train: int,
    n_test: int = NTEST_BASE,
    d: int = D_BASE,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    xi: float = BASE_XI,
    seed: int = 42,
    mu_1_model: Any = None,
    propensity_model: Any = None,
    correlated_x: bool = False,
    mu_1_model_params: Optional[dict] = None,
    mu_0_model_params: Optional[dict] = None,
    models: Any = None,
    nonlinear_prop: bool = False,
    prop_offset: Union[float, str] = 0,
    target_prop: Optional[float] = None,
) -> list:
    if models is None:
        models = FULL_MODEL_SET_AISTATS

    # get data
    X, y, w, p, t = simulate_treatment_setup(
        n_train + n_test,
        d=d,
        n_w=n_w,
        n_c=n_c,
        n_o=n_o,
        n_t=n_t,
        propensity_model=propensity_model,
        propensity_model_params={
            "xi": xi,
            "nonlinear": nonlinear_prop,
            "offset": prop_offset,
            "target_prop": target_prop,
        },
        seed=seed,
        mu_1_model=mu_1_model,
        mu_0_model_params=mu_0_model_params,
        mu_1_model_params=mu_1_model_params,
        covariate_model_params={"correlated": correlated_x},
    )
    # split data
    X_train, y_train, w_train, _ = (
        X[:n_train, :],
        y[:n_train],
        w[:n_train],
        p[:n_train],
    )
    X_test, t_test = X[n_train:, :], t[n_train:]

    rmses = []
    for model_name, model in models.items():
        log.info(f"Training model {model_name}")

        estimator = clone(model)
        estimator.fit(X=X_train, y=y_train, w=w_train)

        cate_test = estimator.predict(X_test, return_po=False)
        rmses.append(eval_root_mse(cate_test, t_test))

    return rmses


def main_AISTATS(
    setting: int = 1, models: Any = None, file_name: str = "res", n_repeats: int = 10
) -> None:
    if models is None:
        models = FULL_MODEL_SET_AISTATS
    elif type(models) is list or type(models) is str:
        models = get_model_set(models)

    if setting == 1:
        # no treatment effect, with confounding, by n
        simulation_experiment_loop(
            [1000, 2000, 5000, 10000],
            change_dim="n",
            n_t=0,
            n_w=0,
            n_c=5,
            n_o=5,
            file_base=file_name,
            models=models,
            n_repeats=n_repeats,
        )
    elif setting == 2:
        # with treatment effect, with confounding, by n
        simulation_experiment_loop(
            [1000, 2000, 5000, 10000],
            change_dim="n",
            n_t=5,
            n_w=0,
            n_c=5,
            n_o=5,
            file_base=file_name,
            models=models,
            n_repeats=n_repeats,
        )
    elif setting == 3:
        # Potential outcomes are supported on independent covariates, no confounding, by n
        simulation_experiment_loop(
            [1000, 2000, 5000, 10000],
            change_dim="n",
            n_t=10,
            n_w=0,
            n_c=0,
            n_o=10,
            file_base=file_name,
            models=models,
            xi=0.5,
            mu_1_model_params={"withbase": False},
            n_repeats=n_repeats,
        )
    elif setting == 4:
        # vary number of predictive features at n=2000
        simulation_experiment_loop(
            [0, 1, 3, 5, 7, 10],
            change_dim=D_T_STRING,
            n_train=2000,
            n_c=5,
            n_o=5,
            file_base=file_name,
            models=models,
            n_repeats=n_repeats,
        )
    elif setting == 5:
        # vary percentage treated at n=2000
        simulation_experiment_loop(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            change_dim=TARGET_STRING,
            n_train=2000,
            n_c=5,
            n_o=5,
            n_t=0,
            n_repeats=n_repeats,
            file_base=file_name,
            models=models,
        )
