# stdlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import catenets.logger as log

# third party
import numpy as np
from scipy.special import expit

from .network import download_if_needed

np.random.seed(0)
random.seed(0)

DATASET = "Twin_Data.csv.gz"
URL = "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"  # noqa: E501


def preprocess(fn_csv: Path) -> List[np.ndarray]:
    """
    Load the dataset and preprocess it.
    Input:
        fn_csv: Path to the input CSV to load
    Outputs:
        X: Feature Vector
        T: Treatment Vector
        Y: Observable Outcomes
        Opt_Y: Potential Outcomes
    """
    # Data Input (11400 patients, 30 features, 2 potential outcomes)
    Data = np.loadtxt(fn_csv, delimiter=",", skiprows=1)

    # Features
    X = Data[:, :30]

    # Feature dimensions and patient numbers
    Dim = len(X[0])
    No = len(X)

    # Labels
    Opt_Y = Data[:, 30:]

    for i in range(2):
        idx = np.where(Opt_Y[:, i] > 365)
        Opt_Y[idx, i] = 365

    Opt_Y = 1 - (Opt_Y / 365.0)

    # Patient Treatment Assignment
    coef = 0 * np.random.uniform(-0.01, 0.01, size=[Dim, 1])
    Temp = expit(np.matmul(X, coef) + np.random.normal(0, 0.01, size=[No, 1]))

    Temp = Temp / (2 * np.mean(Temp))

    Temp[Temp > 1] = 1

    T = np.random.binomial(1, Temp, [No, 1])
    T = T.reshape(
        [
            No,
        ]
    )

    # Observable outcomes
    Y = np.zeros([No, 1])

    # Output
    Y = np.transpose(T) * Opt_Y[:, 1] + np.transpose(1 - T) * Opt_Y[:, 0]
    Y = np.transpose(Y)
    Y = np.reshape(
        Y,
        [
            No,
        ],
    )

    return [
        X,
        T,
        Y,
        Opt_Y,
    ]


def train_test_split(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    Opt_Y: np.ndarray,
    train_rate: float = 0.8,
    downsample: Optional[int] = None,
) -> Tuple:
    """
    Split the dataset for train and test.
    Input:
        X: Feature Vector
        T: Treatment Vector
        Y: Observable Outcomes
        Opt_Y: Potential Outcomes
        train_rate: Train/Test split
    Outputs:
        - Train_X, Test_X: Train and Test features
        - Train_Y: Observable outcomes
        - Train_T: Assigned treatment
        - Opt_Train_Y, Test_Y: Potential outcomes.
    """
    No = len(X)

    temp = np.random.permutation(No)
    Train_No = int(train_rate * No)
    train_idx = temp[:Train_No]
    test_idx = temp[Train_No:No]

    Train_X = X[train_idx, :]
    Train_T = T[train_idx]
    Train_Y = Y[train_idx]
    Opt_Train_Y = Opt_Y[train_idx, :]

    Test_X = X[test_idx, :]
    Test_Y = Opt_Y[test_idx, :]

    if downsample:
        if len(Train_X) > downsample:
            Train_X = Train_X[:downsample]
            Train_Y = Train_Y[:downsample]
            Train_T = Train_T[:downsample]
            Opt_Train_Y = Opt_Train_Y[:downsample]
        if len(Test_X) > downsample:
            Test_X = Test_X[:downsample]
            Test_Y = Test_Y[:downsample]

    return (
        Train_X,
        Train_T,
        Train_Y,
        Opt_Train_Y,
        Test_X,
        Test_Y,
    )


def load(
    data_path: Path, train_split: float = 0.8, downsample: Optional[int] = None
) -> Tuple:
    """
    Download the dataset if needed.
    Load the dataset.
    Preprocess the data.
    Return train/test split.
    """
    csv = data_path / DATASET

    download_if_needed(csv, URL)

    log.debug(f"load dataset {csv}")

    [X, T, Y, Opt_Y] = preprocess(csv)

    return train_test_split(X, T, Y, Opt_Y, train_split, downsample)
