import sys
from typing import Tuple

import cmgp.logger as log
import numpy as np
import pytest
from cmgp import CMGP
from cmgp.datasets import load
from cmgp.utils.metrics import sqrt_PEHE_with_diff

log.add(sink=sys.stderr, level="DEBUG")


def downsample(
    X: np.ndarray, W: np.ndarray, Y: np.ndarray, downsample: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X[:downsample]
    W = W[:downsample]
    Y = Y[:downsample]
    return X, W, Y


def test_model_sanity() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")

    X_train, W_train, Y_train = downsample(X_train, W_train, Y_train, 500)

    model = CMGP(X_train, W_train, Y_train, max_gp_iterations=7, mode="CMGP")

    assert model.max_gp_iterations == 7
    assert model.mode == "CMGP"

    pred = model.predict(X_test)

    assert len(pred) == len(Y_test)


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_training(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)

    X_train, W_train, Y_train = downsample(X_train, W_train, Y_train, 500)

    model = CMGP(X_train, W_train, Y_train, max_gp_iterations=100)

    pred = model.predict(X_test)

    pehe = sqrt_PEHE_with_diff(Y_test, pred)
    print(f"PEHE score for CMGP on {dataset} = {pehe}")

    assert pehe < pehe_threshold
