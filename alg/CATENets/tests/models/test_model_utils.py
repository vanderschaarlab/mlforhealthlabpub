from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from catenets.models.model_utils import (
    check_shape_1d_data,
    check_X_is_np,
    make_val_split,
)


@pytest.mark.parametrize("data", [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])])
def test_check_shape_1d_data_sanity(data: np.ndarray) -> None:
    out = check_shape_1d_data(data)

    assert len(out.shape) == 2


@pytest.mark.parametrize("data", [np.array([1, 2, 3]), pd.DataFrame([1, 2])])
def test_check_X_is_np_sanity(data: Any) -> None:
    out = check_X_is_np(data)

    assert isinstance(out, jnp.ndarray)


def test_make_val_split_sanity() -> None:
    X = np.random.rand(1000, 5)
    y = np.random.randint(0, 1, size=1000)
    w = np.random.randint(0, 1, size=1000)

    X_t, y_t, w_t, X_val, y_val, w_val, VALIDATION_STRING = make_val_split(X, y, w)

    assert X_t.shape[0] == 700
    assert y_t.shape[0] == 700
    assert w_t.shape[0] == 700
    assert X_val.shape[0] == 300
    assert y_val.shape[0] == 300
    assert w_val.shape[0] == 300
    assert VALIDATION_STRING == "validation"
