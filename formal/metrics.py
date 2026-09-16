"""One authoritative implementation of formal regression metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(prediction, observation, capacity: float) -> dict[str, float]:
    if capacity <= 0.0:
        raise ValueError("capacity must be positive")
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    observation = np.asarray(observation, dtype=np.float64).reshape(-1)
    if prediction.shape != observation.shape:
        raise ValueError("prediction and observation must have the same shape")
    finite = np.isfinite(prediction) & np.isfinite(observation)
    if not finite.any():
        raise ValueError("metrics need at least one finite pair")
    error = prediction[finite] - observation[finite]
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    mbe = float(np.mean(error))
    centered = observation[finite] - float(np.mean(observation[finite]))
    denominator = float(np.sum(centered**2))
    r2 = (
        float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
    )
    return {
        "rmse_physical": rmse,
        "mae_physical": mae,
        "mbe_physical": mbe,
        "nrmse_capacity": rmse / capacity,
        "nmae_capacity": mae / capacity,
        "nmbe_capacity": mbe / capacity,
        "r2": r2,
    }
