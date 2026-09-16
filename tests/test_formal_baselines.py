"""Deterministic baselines skip training and consume audited future geometry."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formal.config import (
    DETERMINISTIC_MODELS,
    JobSpec,
    load_experiment_config,
)
from formal.data import load_dataset
from formal.engine import ForecastDataset, run_job

ROOT = Path(__file__).resolve().parents[1]
SEQ_LEN = 96
HORIZON = 24


def _write_geometry_series(path, rows=1000):
    daylight = (np.arange(rows) % 96 > 24).astype(np.float32)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="15min"),
            "ClearSkyGHIProxy": daylight
            * (0.5 + 0.5 * np.cos(np.arange(rows) / 96 * 2 * np.pi)),
            "SolarElevationMask": daylight,
            "TargetObserved": np.ones(rows),
            "Target": np.linspace(0.0, 10.0, rows, dtype=np.float32),
        }
    )
    frame.to_csv(path, index=False)


def test_future_geometry_window_uses_only_future_rows(tmp_path):
    path = tmp_path / "series_geometry.csv"
    _write_geometry_series(path)
    bundle = load_dataset(path, seq_len=SEQ_LEN, horizon=HORIZON)
    dataset = ForecastDataset(bundle, bundle.test.starts)

    history, target, _cycle, future_solar = dataset[0]
    start = int(bundle.test.starts[0])
    target_start = start + SEQ_LEN
    expected = bundle.values[
        target_start : target_start + HORIZON,
        list(bundle.solar_future_indices),
    ]

    assert tuple(future_solar.shape) == (HORIZON, 2)
    assert tuple(history.shape) == (SEQ_LEN, bundle.raw.shape[1])
    assert tuple(target.shape) == (HORIZON,)
    np.testing.assert_array_equal(future_solar.numpy(), expected)
    # The audited channels are deterministic geometry, never the target.
    target_index = bundle.raw.shape[1] - 1
    assert target_index not in bundle.solar_future_indices


@pytest.mark.parametrize("model", list(DETERMINISTIC_MODELS))
def test_deterministic_baselines_run_without_training(model):
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    job = JobSpec("pvod_station00", model, SEQ_LEN, HORIZON, 2026)

    result = run_job(job, config)

    assert result["status"] == "complete"
    assert result["epochs_ran"] == 0
    assert result["best_epoch"] == 0
    assert result["early_stopped"] is False
    assert result["stop_reason"] == "deterministic_baseline"
    assert result["parameter_source"] == "deterministic_baseline"
    assert result["training_history"] == []
    metrics = result["test"]
    for key in ("rmse_physical", "mae_physical", "mbe_physical", "r2"):
        assert np.isfinite(metrics[key])
