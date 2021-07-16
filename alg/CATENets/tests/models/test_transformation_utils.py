from typing import Callable

import numpy as np
import pytest

from catenets.models.transformation_utils import (
    ALL_TRANSFORMATIONS,
    DR_TRANSFORMATION,
    PW_TRANSFORMATION,
    RA_TRANSFORMATION,
    _get_transformation_function,
    aipw_te_transformation,
    ht_te_transformation,
    ra_te_transformation,
)


def test_get_transformation_function_sanity() -> None:
    expected_fns = [ht_te_transformation, aipw_te_transformation, ra_te_transformation]

    for tr, expected in zip(ALL_TRANSFORMATIONS, expected_fns):
        assert _get_transformation_function(tr) is expected

    with pytest.raises(ValueError):
        _get_transformation_function("invalid")


@pytest.mark.parametrize(
    "fn", [aipw_te_transformation, _get_transformation_function(DR_TRANSFORMATION)]
)
def test_aipw_te_transformation_sanity(fn: Callable) -> None:
    res = fn(
        y=np.array([0, 1]),
        w=np.array([1, 0]),
        p=None,
        mu_0=np.array([0.4, 0.6]),
        mu_1=np.array([0.6, 0.4]),
    )
    assert res.shape[0] == 2


@pytest.mark.parametrize(
    "fn", [ht_te_transformation, _get_transformation_function(PW_TRANSFORMATION)]
)
def test_ht_te_transformation_sanity(fn: Callable) -> None:
    res = fn(
        y=np.array([0, 1]),
        w=np.array([1, 0]),
    )
    assert res.shape[0] == 2


@pytest.mark.parametrize(
    "fn", [ra_te_transformation, _get_transformation_function(RA_TRANSFORMATION)]
)
def test_ra_te_transformation_sanity(fn: Callable) -> None:
    res = fn(
        y=np.array([0, 1]),
        w=np.array([1, 0]),
        p=None,
        mu_0=np.array([0.4, 0.6]),
        mu_1=np.array([0.6, 0.4]),
    )
    assert res.shape[0] == 2
