import numpy as np
import pandas as pd
import pytest

from formal.data import load_dataset


def _write_series(path, rows=1000, test_outlier=False):
    target = np.linspace(0.0, 10.0, rows, dtype=np.float32)
    if test_outlier:
        target[-1] = 1000.0
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="15min"),
            "irradiance": np.maximum(0.0, np.sin(np.arange(rows) / 96 * 2 * np.pi)),
            "TargetObserved": np.ones(rows),
            "Target": target,
        }
    )
    frame.to_csv(path, index=False)


def test_deterministic_solar_geometry_columns_are_indexed(tmp_path):
    rows = 1000
    path = tmp_path / "series_geometry.csv"
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

    bundle = load_dataset(path, seq_len=96, horizon=24)

    assert bundle.feature_names[bundle.clear_sky_index] == "ClearSkyGHIProxy"
    assert bundle.feature_names[bundle.daylight_index] == "SolarElevationMask"
    # Clear sky precedes daylight in the audited future channel ordering.
    assert bundle.solar_future_indices == (
        bundle.clear_sky_index,
        bundle.daylight_index,
    )


def test_solar_geometry_absent_without_clear_sky_column(tmp_path):
    path = tmp_path / "series_plain.csv"
    _write_series(path)

    bundle = load_dataset(path, seq_len=96, horizon=24)

    assert bundle.clear_sky_index is None
    # The appended daylight clock heuristic alone cannot drive smart
    # persistence: a deterministic clear-sky trajectory is still required.
    assert bundle.solar_future_indices[0] == bundle.daylight_index


def test_constant_nominal_clear_sky_column_is_audited_out(tmp_path):
    rows = 1000
    path = tmp_path / "series_fake_clear.csv"
    daylight = (np.arange(rows) % 96 > 24).astype(np.float32)
    trajectory = np.where(
        daylight > 0.5,
        np.maximum(0.0, np.sin(np.arange(rows) / 96 * 2 * np.pi)),
        0.0,
    )
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="15min"),
            # Constant nominal irradiance, identical day and night: must be
            # rejected even though it matches the preferred column name.
            "ClearSkyGHIProxy": np.full(rows, 1180.0, dtype=np.float32),
            "SolarGeometryProxy": trajectory.astype(np.float32),
            "SolarElevationMask": daylight,
            "TargetObserved": np.ones(rows),
            "Target": np.linspace(0.0, 10.0, rows, dtype=np.float32),
        }
    )
    frame.to_csv(path, index=False)

    bundle = load_dataset(path, seq_len=96, horizon=24)

    assert bundle.feature_names[bundle.clear_sky_index] == "SolarGeometryProxy"
    assert bundle.solar_future_indices[0] == bundle.clear_sky_index


def test_windows_do_not_cross_target_split_boundaries(tmp_path):
    path = tmp_path / "series.csv"
    _write_series(path)

    bundle = load_dataset(path, seq_len=96, horizon=48)

    assert bundle.train.target_end.max() <= bundle.train_end
    assert bundle.validation.target_start.min() >= bundle.train_end
    assert bundle.validation.target_end.max() <= bundle.validation_end
    assert bundle.test.target_start.min() >= bundle.validation_end


def test_scaler_and_capacity_only_use_training_rows(tmp_path):
    path = tmp_path / "series.csv"
    _write_series(path, test_outlier=True)

    bundle = load_dataset(path, seq_len=96, horizon=48)

    assert bundle.capacity == pytest.approx(bundle.raw[: bundle.train_end, -1].max())
    assert bundle.capacity < 1000.0
    assert bundle.feature_mu[-1] == pytest.approx(
        bundle.raw[: bundle.train_end, -1].mean(), rel=1e-6
    )


def test_discontinuous_rows_are_not_joined_into_a_window(tmp_path):
    path = tmp_path / "series.csv"
    _write_series(path, rows=500)
    frame = pd.read_csv(path)
    frame.loc[250:, "date"] = pd.date_range(
        "2020-02-01", periods=len(frame) - 250, freq="15min"
    )
    frame.to_csv(path, index=False)

    bundle = load_dataset(path, seq_len=32, horizon=16)

    all_starts = np.concatenate(
        [bundle.train.starts, bundle.validation.starts, bundle.test.starts]
    )
    assert not any(start < 250 < start + 48 for start in all_starts)
