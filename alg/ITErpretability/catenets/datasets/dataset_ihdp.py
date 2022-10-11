"""
IHDP (Infant Health and Development Program) dataset
"""
# stdlib
import os
import random
from pathlib import Path
from typing import Any, Tuple

# third party
import numpy as np

import catenets.logger as log

from .network import download_if_needed

np.random.seed(0)
random.seed(0)

TRAIN_DATASET = "ihdp_npci_1-100.train.npz"
TEST_DATASET = "ihdp_npci_1-100.test.npz"
TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


# helper functions
def load_data_npz(fname: Path, get_po: bool = True) -> dict:
    """
    Helper function for loading the IHDP data set (adapted from https://github.com/clinicalml/cfrnet)

    Parameters
    ----------
    fname: Path
        Dataset path

    Returns
    -------
    data: dict
        Raw IHDP dict, with X, w, y and yf keys.
    """
    data_in = np.load(fname)
    data = {"X": data_in["x"], "w": data_in["t"], "y": data_in["yf"]}
    try:
        data["ycf"] = data_in["ycf"]
    except BaseException:
        data["ycf"] = None

    if get_po:
        data["mu0"] = data_in["mu0"]
        data["mu1"] = data_in["mu1"]

    data["HAVE_TRUTH"] = not data["ycf"] is None
    data["dim"] = data["X"].shape[1]
    data["n"] = data["X"].shape[0]

    return data


def prepare_ihdp_data(
    data_train: dict,
    data_test: dict,
    rescale: bool = False,
    setting: str = "C",
    return_pos: bool = False,
) -> Tuple:
    """
    Helper for preprocessing the IHDP dataset.

    Parameters
    ----------
    data_train: pd.DataFrame or dict
        Train dataset
    data_test: pd.DataFrame or dict
        Test dataset
    rescale: bool, default False
        Rescale the outcomes to have similar scale
    setting: str, default C
        Experiment setting
    return_pos: bool
        Return potential outcomes

    Returns
    -------
    X: dict or pd.DataFrame
        Training Feature set
    y: pd.DataFrame or list
        Outcome list
    t: pd.DataFrame or list
        Treatment list
    cate_true_in: pd.DataFrame or list
        Average treatment effects for the training set
    X_t: pd.Dataframe or list
        Test feature set
    cate_true_out: pd.DataFrame of list
        Average treatment effects for the testing set
    """

    X, y, w, mu0, mu1 = (
        data_train["X"],
        data_train["y"],
        data_train["w"],
        data_train["mu0"],
        data_train["mu1"],
    )

    X_t, _, _, mu0_t, mu1_t = (
        data_test["X"],
        data_test["y"],
        data_test["w"],
        data_test["mu0"],
        data_test["mu1"],
    )
    if setting == "D":
        y[w == 1] = y[w == 1] + mu0[w == 1]
        mu1 = mu0 + mu1
        mu1_t = mu0_t + mu1_t

    if rescale:
        # rescale all outcomes to have similar scale of CATEs if sd_cate > 1
        cate_in = mu0 - mu1
        sd_cate = np.sqrt(cate_in.var())

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


def get_one_data_set(D: dict, i_exp: int, get_po: bool = True) -> dict:
    """
    Helper for getting the IHDP data for one experiment. Adapted from https://github.com/clinicalml/cfrnet

    Parameters
    ----------
    D: dict or pd.DataFrame
        All the experiment
    i_exp: int
        Experiment number

    Returns
    -------
    data: dict or pd.Dataframe
        dict with the experiment
    """
    D_exp = {}
    D_exp["X"] = D["X"][:, :, i_exp - 1]
    D_exp["w"] = D["w"][:, i_exp - 1 : i_exp]
    D_exp["y"] = D["y"][:, i_exp - 1 : i_exp]
    if D["HAVE_TRUTH"]:
        D_exp["ycf"] = D["ycf"][:, i_exp - 1 : i_exp]
    else:
        D_exp["ycf"] = None

    if get_po:
        D_exp["mu0"] = D["mu0"][:, i_exp - 1 : i_exp]
        D_exp["mu1"] = D["mu1"][:, i_exp - 1 : i_exp]

    return D_exp


def load(data_path: Path, exp: int = 1, rescale: bool = False, **kwargs: Any) -> Tuple:
    """
    Get IHDP train/test datasets with treatments and labels.

    Parameters
    ----------
    data_path: Path
        Path to the dataset csv. If the data is missing, it will be downloaded.


    Returns
    -------
    X: pd.Dataframe or array
        The training feature set
    w: pd.DataFrame or array
        Training treatment assignments.
    y: pd.Dataframe or array
        The training labels
    training potential outcomes: pd.DataFrame or array.
        Potential outcomes for the training set.
    X_t: pd.DataFrame or array
        The testing feature set
    testing potential outcomes: pd.DataFrame of array
        Potential outcomes for the testing set.
    """
    data_train, data_test = load_raw(data_path)

    data_exp = get_one_data_set(data_train, i_exp=exp, get_po=True)
    data_exp_test = get_one_data_set(data_test, i_exp=exp, get_po=True)

    (
        X,
        y,
        w,
        cate_true_in,
        X_t,
        cate_true_out,
        mu0,
        mu1,
        mu0_t,
        mu1_t,
    ) = prepare_ihdp_data(
        data_exp,
        data_exp_test,
        rescale=rescale,
        return_pos=True,
    )

    return (
        X,
        w,
        y,
        np.asarray([mu0, mu1]).squeeze().T,
        X_t,
        np.asarray([mu0_t, mu1_t]).squeeze().T,
    )


def load_raw(data_path: Path) -> Tuple:
    """
    Get IHDP raw train/test sets.

    Parameters
    ----------
    data_path: Path
        Path to the dataset csv. If the data is missing, it will be downloaded.

    Returns
    -------

    data_train: dict or pd.DataFrame
        Training data
    data_test: dict or pd.DataFrame
        Testing data
    """

    try:
        os.mkdir(data_path)
    except BaseException:
        pass

    train_csv = data_path / TRAIN_DATASET
    test_csv = data_path / TEST_DATASET

    log.debug(f"load raw dataset {train_csv}")

    download_if_needed(train_csv, http_url=TRAIN_URL)
    download_if_needed(test_csv, http_url=TEST_URL)

    data_train = load_data_npz(train_csv, get_po=True)
    data_test = load_data_npz(test_csv, get_po=True)

    return data_train, data_test
