"""
Twins dataset
Load real-world individualized treatment effects estimation datasets

- Reference: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html
"""
# stdlib
import random
from pathlib import Path
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import catenets.logger as log

from .network import download_if_needed


DATASET = "Twin_Data.csv.gz"
URL = "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"  # noqa: E501


def preprocess(
    fn_csv: Path,
    train_ratio: float = 0.8,
    treatment_type: str = "rand",
    seed: int = 42,
    treat_prop: float = 0.5,
) -> Tuple:
    """Helper for preprocessing the Twins dataset.

    Parameters
    ----------
    fn_csv: Path
        Dataset CSV file path.
    train_ratio: float
        The ratio of training data.
    treatment_type: string
        The treatment selection strategy.
    seed: float
        Random seed.

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
    np.random.seed(seed)
    random.seed(seed)
    
    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    df = pd.read_csv(fn_csv)

    cleaned_columns = []
    for col in df.columns:
        cleaned_columns.append(col.replace("'", "").replace("’", ""))
    df.columns = cleaned_columns

    feat_list = list(df)

    # 8: factor not on certificate, 9: factor not classifiable --> np.nan --> mode imputation
    medrisk_list = [
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
    ]
    # 99: missing
    other_list = ["cigar", "drink", "wtgain", "gestat", "dmeduc", "nprevist"]

    other_list2 = ["pldel", "resstatb"]  # but no samples are missing..

    bin_list = ["dmar"] + medrisk_list
    con_list = ["dmage", "mpcb"] + other_list
    cat_list = ["adequacy"] + other_list2

    for feat in medrisk_list:
        df[feat] = df[feat].apply(lambda x: df[feat].mode()[0] if x in [8, 9] else x)

    for feat in other_list:
        df.loc[df[feat] == 99, feat] = df.loc[df[feat] != 99, feat].mean()

    df_features = df[con_list + bin_list]

    for feat in cat_list:
        df_features = pd.concat(
            [df_features, pd.get_dummies(df[feat], prefix=feat)], axis=1
        )

    # Define features
    feat_list = [
        "dmage",
        "mpcb",
        "cigar",
        "drink",
        "wtgain",
        "gestat",
        "dmeduc",
        "nprevist",
        "dmar",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "adequacy_1",
        "adequacy_2",
        "adequacy_3",
        "pldel_1",
        "pldel_2",
        "pldel_3",
        "pldel_4",
        "pldel_5",
        "resstatb_1",
        "resstatb_2",
        "resstatb_3",
        "resstatb_4",
    ]

    x = np.asarray(df_features[feat_list])
    y0 = np.asarray(df[["outcome(t=0)"]]).reshape((-1,))
    y0 = np.array(y0 < 9999, dtype=int)

    y1 = np.asarray(df[["outcome(t=1)"]]).reshape((-1,))
    y1 = np.array(y1 < 9999, dtype=int)

    # Preprocessing
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    no, dim = x.shape

    if treatment_type == "rand":
        # assign with p=0.5
        prob = np.ones(x.shape[0]) * treat_prop
    elif treatment_type == "logistic":
        # assign with logistic prob
        coef = np.random.uniform(-0.1, 0.1, size=[np.shape(x)[1], 1])
        prob = 1 / (1 + np.exp(-np.matmul(x, coef)))

    w = np.random.binomial(1, prob)
    y = y1 * w + y0 * (1 - w)

    potential_y = np.vstack((y0, y1)).T

    # Train/test division
    if train_ratio < 1:
        idx = np.random.permutation(no)
        train_idx = idx[: int(train_ratio * no)]
        test_idx = idx[int(train_ratio * no):]

        train_x = x[train_idx, :]
        train_w = w[train_idx]
        train_y = y[train_idx]
        train_potential_y = potential_y[train_idx, :]

        test_x = x[test_idx, :]
        test_potential_y = potential_y[test_idx, :]
    else:
        train_x = x
        train_w = w
        train_y = y
        train_potential_y = potential_y
        test_x = None
        test_potential_y = None

    return train_x, train_w, train_y, train_potential_y, test_x, test_potential_y


def load(
    data_path: Path,
    train_ratio: float = 0.8,
    treatment_type: str = "rand",
    seed: int = 42,
    treat_prop: float = 0.5,
) -> Tuple:
    """
    Twins dataset dataloader.
        - Download the dataset if needed.
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.

    Parameters
    ----------
    data_path: Path
        Path to the CSV. If it is missing, it will be downloaded.
    train_ratio: float
        Train/test ratio
    treatment_type: str
        Treatment generation strategy
    seed: float
        Random seed
    treat_prop: float
        Treatment proportion

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
    csv = data_path / DATASET

    download_if_needed(csv, http_url=URL)

    log.debug(f"load dataset {csv}")

    return preprocess(
        csv,
        train_ratio=train_ratio,
        treatment_type=treatment_type,
        seed=seed,
        treat_prop=treat_prop,
    )
