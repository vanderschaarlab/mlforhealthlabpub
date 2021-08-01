import sys

import ganite.logger as log
import pytest
from ganite import Ganite
from ganite.datasets import load
from ganite.utils.metrics import sqrt_PEHE_with_diff

log.add(sink=sys.stderr, level="DEBUG")


def test_model_sanity() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")

    model = Ganite(
        X_train,
        W_train,
        Y_train,
        num_iterations=12,
        dim_hidden=3,
        depth=2,
        alpha=0.5,
        beta=0.1,
        minibatch_size=16,
        num_discr_iterations=7,
    )

    assert model.minibatch_size == 16
    assert model.alpha == 0.5
    assert model.beta == 0.1
    assert model.depth == 2
    assert model.num_discr_iterations == 7
    assert model.num_iterations == 12

    pred = model(X_test).numpy()

    assert len(pred) == len(Y_test)


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_training(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)

    model = Ganite(X_train, W_train, Y_train, num_iterations=1000)

    pred = model(X_test).cpu().numpy()

    pehe = sqrt_PEHE_with_diff(Y_test, pred)
    print(f"PEHE score for GANITE on {dataset} = {pehe}")

    assert pehe < pehe_threshold
