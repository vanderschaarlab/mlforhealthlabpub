"""
Author: Alicia Curth
Some utils for experiments
"""
import jax.numpy as jnp

from catenets.models.base import check_shape_1d_data

from catenets.models import T_NAME, SNET1_NAME, SNET2_NAME, SNET3_NAME, \
    SNET_NAME, PSEUDOOUT_NAME, PseudoOutcomeNet, get_catenet, DRNET_NAME, RANET_NAME, \
    RNET_NAME, XNET_NAME
from catenets.models.transformation_utils import DR_TRANSFORMATION, PW_TRANSFORMATION, \
    RA_TRANSFORMATION

SEP = "_"


def eval_mse_model(inputs, targets, predict_fun, params):
    # evaluate the mse of a model given its function and params
    preds = predict_fun(params, inputs)
    return jnp.mean((preds - targets) ** 2)


def eval_mse(preds, targets):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return jnp.mean((preds - targets) ** 2)


def eval_root_mse(cate_pred, cate_true):
    cate_true = check_shape_1d_data(cate_true)
    cate_pred = check_shape_1d_data(cate_pred)
    return jnp.sqrt(eval_mse(cate_pred, cate_true))


def get_model_set(model_selection = 'all', model_params: dict = None):
    """ Helper function to retrieve a set of models """
    # get model selection
    if type(model_selection) is str:
        if model_selection == 'plug':
            models = get_all_plugin_models()
        elif model_selection == 'pseudo':
            models = get_all_pseudoout_models()
        elif model_selection == 'twostep':
            models = get_all_twostep_models()
        elif model_selection == 'all':
            models = dict(**get_all_plugin_models(), **get_all_pseudoout_models())
        else:
            models = {model_selection: get_catenet(model_selection)()}
    elif type(model_selection) is list:
        models = {}
        for model in model_selection:
            models.update({model: get_catenet(model)()})
    else:
        raise ValueError("model_selection should be string or list.")

    # set hyperparameters
    if model_params is not None:
        for model in models.values():
            existing_params = model.get_params()
            new_params = {key: val for key, val in model_params.items() if
                          key in existing_params.keys()}
            model.set_params(**new_params)

    return models


ALL_PLUGIN_MODELS = [T_NAME,  SNET1_NAME,  SNET2_NAME, SNET3_NAME, SNET_NAME]
ALL_PSEUDOOUT_MODELS = [DR_TRANSFORMATION, PW_TRANSFORMATION, RA_TRANSFORMATION]
ALL_TWOSTEP_MODELS = [DRNET_NAME, RANET_NAME, XNET_NAME, RNET_NAME]



def get_all_plugin_models():
    model_dict = {}
    for name in ALL_PLUGIN_MODELS:
        model_dict.update({name: get_catenet(name)()})
    return model_dict


def get_all_pseudoout_models():  # DR, RA, PW learner
    model_dict = {}
    for trans in ALL_PSEUDOOUT_MODELS:
        model_dict.update({PSEUDOOUT_NAME + SEP + trans: PseudoOutcomeNet(transformation=trans)})
    return model_dict


def get_all_twostep_models():  # DR, RA, R, X learner
    model_dict = {}
    for name in ALL_TWOSTEP_MODELS:
        model_dict.update({name: get_catenet(name)()})
    return model_dict
