# stdlib
import os
from pathlib import Path
from typing import Any, Tuple

from . import dataset_acic2016, dataset_ihdp, dataset_twins

DATA_PATH = Path(os.path.dirname(__file__)) / Path("data")

try:
    os.mkdir(DATA_PATH)
except BaseException:
    pass


def load(dataset: str, *args: Any, **kwargs: Any) -> Tuple:
    """
    Input:
        dataset: the name of the dataset to load
    Outputs:
        - Train_X, Test_X: Train and Test features
        - Train_Y: Observable outcomes
        - Train_T: Assigned treatment
        - Test_Y: Potential outcomes.
    """
    if dataset == "twins":
        return dataset_twins.load(DATA_PATH, *args, **kwargs)
    if dataset == "ihdp":
        return dataset_ihdp.load(DATA_PATH, *args, **kwargs)
    if dataset == "acic2016":
        return dataset_acic2016.load(DATA_PATH, *args, **kwargs)
    else:
        raise Exception("Unsupported dataset")


__all__ = ["dataset_ihdp", "dataset_twins", "dataset_acic2016", "load"]
