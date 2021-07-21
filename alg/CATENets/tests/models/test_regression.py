import os
import sys
from copy import deepcopy

import catenets.logger as log
import numpy as np
import pytest
from catenets.experiments.experiment_utils import get_model_set
from sklearn.model_selection import train_test_split

try:
    from medicaldata.Twins import download, load

    medicaldata_exists = True
except BaseException:
    medicaldata_exists = False

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

models = list(ALL_MODELS.keys())
models.remove("PseudoOutcomeNet_PW")


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


@pytest.mark.skipif(not medicaldata_exists, reason="Missing medicaldata package")
@pytest.mark.parametrize("model_name", models)
def test_model_fit_issue_twins_version(model_name: str) -> None:
    csv_path = "test.csv"

    if not os.path.isfile(csv_path):
        download(csv_path)

    X, W, Y = load(csv_path)

    model = deepcopy(ALL_MODELS[model_name])

    X_train, X_test, W_train, W_test, Y_train, Y_test = train_test_split(X, W, Y)

    model.fit(
        X=X_train.to_numpy(), y=Y_train["outcome"].to_numpy(), w=W_train.to_numpy()
    )

    cate_pred = model.predict(X_test, return_po=False)

    pehe = sqrt_PEHE(Y_test.to_numpy(), cate_pred)
    print(f"PEHE score for model {model_name} on Twins dataset = {pehe}")
