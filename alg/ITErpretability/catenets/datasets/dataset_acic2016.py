"""
ACIC2016 dataset
"""
# stdlib
import random
from pathlib import Path
from typing import Any, Tuple
import glob

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import catenets.logger as log

from .network import download_if_needed

np.random.seed(0)
random.seed(0)

FILE_ID = "0B7pG5PPgj6A3N09ibmFwNWE1djA"
PREPROCESSED_FILE_ID = "1iOfEAk402o3jYBs2Prfiz6oaailwWcR5"

NUMERIC_COLS = [
    0,
    3,
    4,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    24,
    25,
    30,
    31,
    32,
    33,
    39,
    40,
    41,
    53,
    54,
]
N_NUM_COLS = len(NUMERIC_COLS)


def get_acic_covariates(
        fn_csv: Path, keep_categorical: bool = False, preprocessed: bool = True
) -> np.ndarray:
    X = pd.read_csv(fn_csv)
    if not keep_categorical:
        X = X.drop(columns=["x_2", "x_21", "x_24"])
    else:
        # encode categorical features
        feature_list = []
        for cols_ in X.columns:
            if type(X.loc[X.index[0], cols_]) not in [np.int64, np.float64]:

                enc = OneHotEncoder(drop="first")

                enc.fit(np.array(X[[cols_]]).reshape((-1, 1)))

                for k in range(len(list(enc.get_feature_names()))):
                    X[cols_ + list(enc.get_feature_names())[k]] = enc.transform(
                        np.array(X[[cols_]]).reshape((-1, 1))
                    ).toarray()[:, k]

                feature_list.append(cols_)

        X.drop(feature_list, axis=1, inplace=True)

    if preprocessed:
        X_t = X.values
    else:
        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)
    return X_t


def preprocess_simu(
        fn_csv: Path,
        n_0: int = 2000,
        n_1: int = 200,
        n_test: int = 500,
        error_sd: float = 1,
        sp_lin: float = 0.6,
        sp_nonlin: float = 0.3,
        prop_gamma: float = 0,
        prop_omega: float = 0,
        ate_goal: float = 0,
        inter: bool = True,
        i_exp: int = 0,
        keep_categorical: bool = False,
        preprocessed: bool = True,
) -> Tuple:
    X = get_acic_covariates(
        fn_csv, keep_categorical=keep_categorical, preprocessed=preprocessed
    )
    np.random.seed(i_exp)

    # shuffle indices
    n_total, n_cov = X.shape
    ind = np.arange(n_total)
    np.random.shuffle(ind)
    ind_test = ind[-n_test:]
    ind_1 = ind[n_0: (n_0 + n_1)]

    # create treatment indicator (treatment assignment does not matter in test set)
    w = np.zeros(n_total).reshape((-1, 1))
    w[ind_1] = 1

    # create dgp
    coeffs_ = [0, 1]
    # sample baseline coefficients
    beta_0 = np.random.choice(coeffs_, size=n_cov, replace=True, p=[1 - sp_lin, sp_lin])
    intercept = np.random.choice([x for x in np.arange(-1, 1.25, 0.25)])

    # sample treatment effect coefficients
    gamma = np.random.choice(
        coeffs_, size=n_cov, replace=True, p=[1 - prop_gamma, prop_gamma]
    )
    omega = np.random.choice(
        [0, 1], replace=True, size=n_cov, p=[prop_omega, 1 - prop_omega]
    )

    # simulate mu_0 and mu_1
    mu_0 = (intercept + np.dot(X, beta_0)).reshape((-1, 1))
    mu_1 = (intercept + np.dot(X, gamma + beta_0 * omega)).reshape((-1, 1))
    if sp_nonlin > 0:
        coefs_sq = [0, 0.1]
        beta_sq = np.random.choice(
            coefs_sq, size=N_NUM_COLS, replace=True, p=[1 - sp_nonlin, sp_nonlin]
        )
        omega = np.random.choice(
            [0, 1], replace=True, size=N_NUM_COLS, p=[prop_omega, 1 - prop_omega]
        )
        X_sq = X[:, NUMERIC_COLS] ** 2
        mu_0 = mu_0 + np.dot(X_sq, beta_sq).reshape((-1, 1))
        mu_1 = mu_1 + np.dot(X_sq, beta_sq * omega).reshape((-1, 1))

        if inter:
            # randomly add some interactions
            ind_c = np.arange(n_cov)
            np.random.shuffle(ind_c)
            inter_list = list()
            for i in range(0, n_cov - 2, 2):
                inter_list.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]])

            X_inter = np.array(inter_list).T
            n_inter = X_inter.shape[1]
            beta_inter = np.random.choice(
                coefs_sq, size=n_inter, replace=True, p=[1 - sp_nonlin, sp_nonlin]
            )
            omega = np.random.choice(
                [0, 1], replace=True, size=n_inter, p=[prop_omega, 1 - prop_omega]
            )
            mu_0 = mu_0 + np.dot(X_inter, beta_inter).reshape((-1, 1))
            mu_1 = mu_1 + np.dot(X_inter, beta_inter * omega).reshape((-1, 1))

    ate = np.mean(mu_1 - mu_0)
    mu_1 = mu_1 - ate + ate_goal

    y = (
            w * mu_1
            + (1 - w) * mu_0
            + np.random.normal(0, error_sd, n_total).reshape((-1, 1))
    )

    X_train, y_train, w_train, mu_0_train, mu_1_train = (
        X[ind[: (n_0 + n_1)], :],
        y[ind[: (n_0 + n_1)]],
        w[ind[: (n_0 + n_1)]],
        mu_0[ind[: (n_0 + n_1)]],
        mu_1[ind[: (n_0 + n_1)]],
    )
    X_test, y_test, w_test, mu_0_t, mu_1_t = (
        X[ind_test, :],
        y[ind_test],
        w[ind_test],
        mu_0[ind_test],
        mu_1[ind_test],
    )

    return (
        X_train,
        w_train,
        y_train,
        np.asarray([mu_0_train, mu_1_train]).squeeze().T,
        X_test,
        w_test,
        y_test,
        np.asarray([mu_0_t, mu_1_t]).squeeze().T,
    )


def get_acic_orig_filenames(data_path: Path, simu_num: int) -> list:
    return sorted(glob.glob((data_path / ("data_cf_all/" + str(simu_num) +
                                          '/zymu_*.csv')).__str__()))


def get_acic_orig_outcomes(data_path: Path,
                           simu_num: int,
                           i_exp: int) -> Tuple:
    file_list = get_acic_orig_filenames(data_path=data_path,
                                        simu_num=simu_num)

    out = pd.read_csv(file_list[i_exp])
    w = out['z']
    y = w * out['y1'] + (1 - w) * out['y0']
    mu_0, mu_1 = out['mu0'], out['mu1']
    return y.values, w.values, mu_0.values, mu_1.values


def preprocess_acic_orig(fn_csv: Path,
                         data_path: Path,
                         preprocessed: bool = False,
                         keep_categorical: bool = True,
                         simu_num: int = 1,
                         i_exp: int = 0,
                         train_size: int = 4000,
                         random_split: bool = False
                         )-> Tuple:
    X = get_acic_covariates(
        fn_csv, keep_categorical=keep_categorical, preprocessed=preprocessed
    )

    y, w, mu_0, mu_1 = get_acic_orig_outcomes(data_path=data_path, simu_num=simu_num, i_exp=i_exp)

    if not random_split:
        X_train, y_train, w_train, mu_0_train, mu_1_train = X[:train_size, :], y[:train_size], \
                                                            w[:train_size], mu_0[:train_size], \
                                                            mu_1[:train_size]
        X_test, y_test, w_test, mu_0_test, mu_1_test = X[train_size:, :], y[train_size:], \
                                                       w[train_size:], mu_0[train_size:], \
                                                       mu_1[train_size:]
    else:
        X_train, X_test, y_train, y_test, w_train, w_test, \
        mu_0_train, mu_0_test, mu_1_train, mu_1_test = train_test_split(X, y, w, mu_0, mu_1,
                                                                        test_size=1 - train_size,
                                                                        random_state=i_exp)

    return (
        X_train,
        w_train,
        y_train,
        np.asarray([mu_0_train, mu_1_train]).squeeze().T,
        X_test,
        w_test,
        y_test,
        np.asarray([mu_0_test, mu_1_test]).squeeze().T,
    )


def preprocess(fn_csv: Path,
               data_path: Path,
               preprocessed: bool = True,
               original_acic_outcomes: bool = False,
               **kwargs: Any,
               ) -> Tuple:
    if not original_acic_outcomes:
        return preprocess_simu(fn_csv=fn_csv, preprocessed=preprocessed, **kwargs)
    else:
        return preprocess_acic_orig(fn_csv=fn_csv, preprocessed=preprocessed,
                                    data_path=data_path, **kwargs)


def load(
        data_path: Path,
        preprocessed: bool = True,
        original_acic_outcomes: bool = False,
        **kwargs: Any,
) -> Tuple:
    """
    ACIC2016 dataset dataloader.
        - Download the dataset if needed.
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.

    Parameters
    ----------
    data_path: Path
        Path to the CSV. If it is missing, it will be downloaded.
    preprocessed: bool
        Switch between the raw and preprocessed versions of the dataset.
    original_acic_outcomes: bool
        Switch between new simulations (Inductive bias paper) and original acic outcomes

    Returns
    -------
    train_x: array or pd.DataFrame
        Features in training data.
    train_t: array or pd.DataFrame
        Treatments in training data.
    train_y: array or pd.DataFrame
        Observed outcomes in training data.
    train_potential_y: array or pd.DataFrame
        Potential outcomes in training data.
    test_x: array or pd.DataFrame
        Features in testing data.
    test_potential_y: array or pd.DataFrame
        Potential outcomes in testing data.
    """
    if preprocessed:
        csv = data_path / "x_trans.csv"

        download_if_needed(csv, file_id=PREPROCESSED_FILE_ID)
    else:
        arch = data_path / "data_cf_all.tar.gz"

        download_if_needed(
            arch, file_id=FILE_ID, unarchive=True, unarchive_folder=data_path
        )

        csv = data_path / "data_cf_all/x.csv"
    log.debug(f"load dataset {csv}")

    return preprocess(csv, data_path=data_path, preprocessed=preprocessed,
                      original_acic_outcomes=original_acic_outcomes,
                      **kwargs)
