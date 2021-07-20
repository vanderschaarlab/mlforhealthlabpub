"""
Author: Alicia Curth
Some utils for experiments
"""
from typing import Callable, Optional, Union

import jax.numpy as jnp

from catenets.models import (
    DRNET_NAME,
    PSEUDOOUT_NAME,
    RANET_NAME,
    RNET_NAME,
    SNET1_NAME,
    SNET2_NAME,
    SNET3_NAME,
    SNET_NAME,
    T_NAME,
    XNET_NAME,
    PseudoOutcomeNet,
    get_catenet,
)
from catenets.models.base import check_shape_1d_data
from catenets.models.transformation_utils import (
    DR_TRANSFORMATION,
    PW_TRANSFORMATION,
    RA_TRANSFORMATION,
)

SEP = "_"


def eval_mse_model(
    inputs: jnp.ndarray, targets: jnp.ndarray, predict_fun: Callable, params: dict
) -> jnp.ndarray:
    # evaluate the mse of a model given its function and params
    preds = predict_fun(params, inputs)
    return jnp.mean((preds - targets) ** 2)


def eval_mse(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return jnp.mean((preds - targets) ** 2)


def eval_root_mse(cate_pred: jnp.ndarray, cate_true: jnp.ndarray) -> jnp.ndarray:
    cate_true = check_shape_1d_data(cate_true)
    cate_pred = check_shape_1d_data(cate_pred)
    return jnp.sqrt(eval_mse(cate_pred, cate_true))


def get_model_set(
    model_selection: Union[str, list] = "all", model_params: Optional[dict] = None
) -> dict:
    """Helper function to retrieve a set of models"""
    # get model selection
    models: dict
    if isinstance(model_selection, str):
        if model_selection == "plug":
            models = get_all_plugin_models()
        elif model_selection == "pseudo":
            models = get_all_pseudoout_models()
        elif model_selection == "twostep":
            models = get_all_twostep_models()
        elif model_selection == "all":
            models = dict(**get_all_plugin_models(), **get_all_pseudoout_models())
        else:
            models = {model_selection: get_catenet(model_selection)()}
    elif isinstance(model_selection, list):
        models = {}
        for model in model_selection:
            models.update({model: get_catenet(model)()})
    else:
        raise ValueError("model_selection should be string or list.")

    # set hyperparameters
    if model_params is not None:
        for model in models.values():
            existing_params = model.get_params()
            new_params = {
                key: val
                for key, val in model_params.items()
                if key in existing_params.keys()
            }
            model.set_params(**new_params)

    return models


ALL_PLUGIN_MODELS: list = [T_NAME, SNET1_NAME, SNET2_NAME, SNET3_NAME, SNET_NAME]
ALL_PSEUDOOUT_MODELS: list = [DR_TRANSFORMATION, PW_TRANSFORMATION, RA_TRANSFORMATION]
ALL_TWOSTEP_MODELS: list = [DRNET_NAME, RANET_NAME, XNET_NAME, RNET_NAME]


def get_all_plugin_models() -> dict:
    model_dict = {}
    for name in ALL_PLUGIN_MODELS:
        model_dict.update({name: get_catenet(name)()})
    return model_dict


def get_all_pseudoout_models() -> dict:  # DR, RA, PW learner
    model_dict = {}
    for trans in ALL_PSEUDOOUT_MODELS:
        model_dict.update(
            {PSEUDOOUT_NAME + SEP + trans: PseudoOutcomeNet(transformation=trans)}
        )
    return model_dict


def get_all_twostep_models() -> dict:  # DR, RA, R, X learner
    model_dict = {}
    for name in ALL_TWOSTEP_MODELS:
        model_dict.update({name: get_catenet(name)()})
    return model_dict
