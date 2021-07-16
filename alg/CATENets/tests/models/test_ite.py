import sys
from copy import deepcopy

import catenets.logger as log
import numpy as np
import pytest
from catenets.datasets import load
from catenets.experiments.experiment_utils import get_model_set

LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL_IHDP = 0

MODEL_PARAMS = {
    "n_layers_out": LAYERS_OUT,
    "n_layers_r": LAYERS_R,
    "penalty_l2": PENALTY_L2,
    "penalty_orthogonal": PENALTY_ORTHOGONAL_IHDP,
    "n_layers_out_t": LAYERS_OUT,
    "n_layers_r_t": LAYERS_R,
    "penalty_l2_t": PENALTY_L2,
}

ALL_MODELS = get_model_set(model_selection="all", model_params=MODEL_PARAMS)

log.add(sink=sys.stderr, level="DEBUG")


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
@pytest.mark.parametrize("model_name", list(ALL_MODELS.keys()))
def test_model_sanity(dataset: str, pehe_threshold: float, model_name: str) -> None:
    model = deepcopy(ALL_MODELS[model_name])

    X_train, W_train, Y_train, X_test, Y_test = load(dataset)

    model.fit(X=X_train, y=Y_train, w=W_train)

    cate_pred = model.predict(X_test, return_po=False)

    pehe = sqrt_PEHE(Y_test, cate_pred)
    print(model_name, pehe)
    assert pehe < pehe_threshold
