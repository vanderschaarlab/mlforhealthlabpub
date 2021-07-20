# stdlib
import os
import random
from pathlib import Path
from typing import Tuple

import cmgp.logger as log

# third party
import numpy as np

from .network import download_if_needed

np.random.seed(0)
random.seed(0)

TRAIN_DATASET = "ihdp_npci_1-100.train.npz"
TEST_DATASET = "ihdp_npci_1-100.test.npz"
TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


# helper functions
def load_data_npz(fname: Path, get_po: bool = True) -> dict:
    """Load data set (adapted from https://github.com/clinicalml/cfrnet)"""
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
    data_train: dict, data_test: dict, rescale: bool = True, return_pos: bool = False
) -> Tuple:
    """Prepare data"""

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
    """Get data for one experiment. Adapted from https://github.com/clinicalml/cfrnet"""
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


def load(data_path: Path, train_split: float = 0.8) -> Tuple:
    """
    Download the dataset if needed.
    Load the dataset.
    Preprocess the data.
    Return train/test split.
    """
    data_train, data_test = load_raw(data_path)

    exp = 1
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
        rescale=True,
        return_pos=True,
    )

    return (
        X,
        w,
        y,
        cate_true_in,
        X_t,
        np.asarray([mu0_t, mu1_t]).squeeze().T,
    )


def load_raw(data_path: Path) -> Tuple:
    """
    Download the dataset if needed.
    Load the dataset.
    """

    try:
        os.mkdir(data_path)
    except BaseException:
        pass

    train_csv = data_path / TRAIN_DATASET
    test_csv = data_path / TEST_DATASET

    log.debug(f"load raw dataset f{train_csv}")

    download_if_needed(train_csv, TRAIN_URL)
    download_if_needed(test_csv, TEST_URL)

    data_train = load_data_npz(train_csv, get_po=True)
    data_test = load_data_npz(test_csv, get_po=True)

    return data_train, data_test
