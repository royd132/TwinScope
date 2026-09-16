"""Time-safe CSV loading and forecasting-window splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMNS = (
    "date",
    "date.1",
    "timestamp",
    "Timestamp",
    "Time",
    "interval_datetime",
)
QUALITY_COLUMNS = {"TargetObserved", "ObservedFlag", "QualityFlag", "DataQuality"}
# Deterministic solar geometry: functions of timestamp only, so future values
# inside the forecast horizon are known in advance and auditable for the
# smart persistence baseline (never observed power or NWP).
CLEAR_SKY_CANDIDATES = (
    "ClearSkyGHIProxy",
    "ClearSkyGHI",
    "ClearSkyGTI",
    "ghic",
    "gtic",
    "SolarGeometryProxy",
    "SolarZenithCos",
    "ClearSkyProxy",
)
DAYLIGHT_CANDIDATES = ("SolarElevationMask", "DaylightHeuristic")


@dataclass(frozen=True)
class WindowSplit:
    starts: np.ndarray
    seq_len: int
    horizon: int

    @property
    def history_start(self) -> np.ndarray:
        return self.starts

    @property
    def target_start(self) -> np.ndarray:
        return self.starts + self.seq_len

    @property
    def target_end(self) -> np.ndarray:
        return self.starts + self.seq_len + self.horizon


@dataclass(frozen=True)
class DatasetBundle:
    path: Path
    raw: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    target_observed: np.ndarray
    timestamps: np.ndarray
    sample_hours: float
    capacity: float
    train_end: int
    validation_end: int
    train: WindowSplit
    validation: WindowSplit
    test: WindowSplit
    clear_sky_index: int | None
    daylight_index: int | None
    solar_future_indices: tuple[int, ...]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", on_bad_lines="skip")


def _infer_sample_hours(timestamps: pd.Series) -> float:
    differences = timestamps.sort_values().diff().dt.total_seconds().div(3600)
    positive = differences[(differences > 0) & np.isfinite(differences)]
    if positive.empty:
        raise ValueError("dataset needs at least two valid timestamps")
    return float(positive.median())


def _continuous_starts(
    timestamps: np.ndarray, sample_hours: float, total_len: int
) -> np.ndarray:
    starts = np.arange(len(timestamps) - total_len + 1, dtype=np.int64)
    if not len(starts):
        return starts
    stamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    differences = np.diff(stamp_ns) / 3_600_000_000_000.0
    bad = np.zeros(len(timestamps), dtype=np.int64)
    bad[1:] = ~np.isfinite(differences) | (
        np.abs(differences - sample_hours) > max(sample_hours * 0.05, 1e-6)
    )
    prefix = np.concatenate(([0], np.cumsum(bad)))
    return starts[(prefix[starts + total_len] - prefix[starts + 1]) == 0]


def _select_clear_sky_name(
    names: list[str],
    raw: np.ndarray,
    train_end: int,
    daylight_index: int | None,
) -> str | None:
    """Pick the first candidate that behaves like a daylight-only trajectory."""
    night = (
        raw[:train_end, daylight_index] < 0.5
        if daylight_index is not None
        else None
    )
    for candidate in CLEAR_SKY_CANDIDATES:
        if candidate not in names:
            continue
        column = raw[:train_end, names.index(candidate)]
        peak = float(column.max())
        if peak <= 0.0:
            continue
        if night is not None:
            # Night rows exist: the proxy must be near zero on all of them.
            if float(column[night].max()) < 0.05 * peak:
                return candidate
        elif float(column.min()) < 0.02 * peak:
            # No daylight mask: a diurnal trajectory must reach near zero.
            return candidate
    return None


def load_dataset(
    path: Path,
    seq_len: int,
    horizon: int,
    train_fraction: float = 0.70,
    validation_end_fraction: float = 0.85,
) -> DatasetBundle:
    """Load one dataset without using validation/test rows for fitted state."""
    path = Path(path).resolve()
    frame = _read_csv(path)
    if "Target" not in frame:
        raise ValueError(f"Target column is missing in {path}")
    time_name = next((name for name in TIME_COLUMNS if name in frame), None)
    if time_name is None:
        raise ValueError(f"timestamp column is missing in {path}")
    timestamps = pd.to_datetime(frame[time_name], errors="coerce")
    sample_hours = _infer_sample_hours(timestamps)
    target = pd.to_numeric(frame["Target"], errors="coerce")
    observed_name = next((name for name in QUALITY_COLUMNS if name in frame), None)
    observed = (
        pd.to_numeric(frame[observed_name], errors="coerce").fillna(0).to_numpy() > 0.5
        if observed_name
        else target.notna().to_numpy()
    )

    excluded = {"Target", *TIME_COLUMNS}
    numeric = frame[[name for name in frame.columns if name not in excluded]].apply(
        pd.to_numeric, errors="coerce"
    )
    names = [name for name in numeric if numeric[name].notna().mean() >= 0.80]
    features = numeric[names].to_numpy(np.float32)

    minute = (
        timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60.0
    ).to_numpy()
    phase = minute / 1440.0
    clock = np.column_stack(
        [
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            np.maximum(0.0, -np.cos(2 * np.pi * phase)),
            ((minute >= 360) & (minute <= 1200)).astype(np.float32),
        ]
    ).astype(np.float32)
    features = np.column_stack((features, clock)).astype(np.float32)
    names.extend(("ClockSin", "ClockCos", "ClockSolarProxy", "DaylightHeuristic"))
    raw = np.column_stack((features, target.to_numpy(np.float32))).astype(np.float32)
    names.append("Target")

    valid = np.isfinite(raw).all(axis=1) & timestamps.notna().to_numpy()
    raw = raw[valid]
    observed = observed[valid]
    timestamp_values = timestamps[valid].to_numpy(dtype="datetime64[ns]")
    if not 0.0 < train_fraction < validation_end_fraction < 1.0:
        raise ValueError("split fractions must satisfy 0 < train < validation < 1")
    train_end = int(train_fraction * len(raw))
    validation_end = int(validation_end_fraction * len(raw))
    if train_end <= seq_len + horizon or validation_end <= train_end:
        raise ValueError("dataset is too short for the requested split and horizon")

    training_values = raw[:train_end].astype(np.float64)
    feature_mu = training_values.mean(axis=0)
    feature_sd = training_values.std(axis=0) + 1e-6
    for index, name in enumerate(names):
        if name in QUALITY_COLUMNS:
            feature_mu[index] = 0.0
            feature_sd[index] = 1.0
    values = ((raw - feature_mu) / feature_sd).astype(np.float32)
    capacity = float(raw[:train_end, -1].max())
    if capacity <= 0.0:
        raise ValueError("training target capacity must be positive")

    daylight_name = next((n for n in DAYLIGHT_CANDIDATES if n in names), None)
    daylight_index = names.index(daylight_name) if daylight_name is not None else None
    # A valid deterministic clear-sky trajectory must vanish at night; some
    # inputs ship a constant nominal-irradiance column under a clear-sky name
    # (e.g. DKASC), which is audited out here using training rows only.
    clear_name = _select_clear_sky_name(
        names, raw, train_end, daylight_index
    )
    clear_sky_index = names.index(clear_name) if clear_name is not None else None
    solar_future_names = ([clear_name] if clear_name is not None else [])
    if daylight_name is not None:
        solar_future_names.append(daylight_name)
    solar_future_indices = tuple(names.index(n) for n in solar_future_names)

    starts = _continuous_starts(timestamp_values, sample_hours, seq_len + horizon)
    if len(starts):
        observed_prefix = np.concatenate(([0], np.cumsum(observed.astype(np.int64))))
        target_start = starts + seq_len
        target_observed = (
            observed_prefix[target_start + horizon] - observed_prefix[target_start]
        ) == horizon
        starts = starts[target_observed]

    train_starts = starts[starts + seq_len + horizon <= train_end]
    validation_starts = starts[
        (starts + seq_len >= train_end) & (starts + seq_len + horizon <= validation_end)
    ]
    test_starts = starts[starts + seq_len >= validation_end]
    if not all(map(len, (train_starts, validation_starts, test_starts))):
        raise ValueError("one or more chronological splits contain no valid windows")

    return DatasetBundle(
        path=path,
        raw=raw,
        values=values,
        feature_names=tuple(names),
        feature_mu=feature_mu,
        feature_sd=feature_sd,
        target_observed=observed.astype(np.float32),
        timestamps=timestamp_values,
        sample_hours=sample_hours,
        capacity=capacity,
        train_end=train_end,
        validation_end=validation_end,
        train=WindowSplit(train_starts, seq_len, horizon),
        validation=WindowSplit(validation_starts, seq_len, horizon),
        test=WindowSplit(test_starts, seq_len, horizon),
        clear_sky_index=clear_sky_index,
        daylight_index=daylight_index,
        solar_future_indices=solar_future_indices,
    )
