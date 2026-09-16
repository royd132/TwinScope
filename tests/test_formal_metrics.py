import numpy as np
import pytest

from formal.metrics import regression_metrics


def test_mbe_is_prediction_minus_observation():
    got = regression_metrics(
        np.array([2.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        capacity=2.0,
    )

    assert got["mbe_physical"] == pytest.approx(1.0)
    assert got["nmbe_capacity"] == pytest.approx(0.5)


def test_r2_uses_sum_of_squared_errors():
    got = regression_metrics(
        np.array([1.0, 2.0, 4.0]),
        np.array([1.0, 3.0, 5.0]),
        capacity=5.0,
    )

    assert got["r2"] == pytest.approx(0.75)


def test_metrics_reject_nonpositive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        regression_metrics(np.array([1.0]), np.array([1.0]), capacity=0.0)
