"""Preprocess new DKASC Alice Springs sites (1A and 19) into the H16 protocol.

Identity (verified against dkasolarcentre.com.au):
  data/91-Site_DKA-M9_B-Phase.csv -> Site 1A, Trina 10.5 kW, mono-Si, dual
      tracker (post 2013-08-14 two-tracker configuration), B-phase.
  data/98-Site_DKA-M8_B-Phase.csv -> Site 19, Sungrid 5.0 kW, mono-Si, fixed.

The recipe mirrors the frozen DKASC_site31 / DKASC_site9a 15-min artefacts so
all four DKASC stations share the same feature semantics, windowing rule and
train/val/test calendar position:

  * window fixed to 2018-02-01 .. 2020-01-31 (the exact two calendar years used
    by DKASC_site9a; chosen ex-ante for cross-site comparability; 2014-2016 was
    100% complete but would not share weather years with sites 31/9a, and
    2023-2025 has ~12% weather gaps).
  * 5-minute samples are averaged into label-left 15-minute slots (:00 slot =
    raw :00,:05,:10).
  * Target = mean Active_Power clipped at zero. TargetObserved = 1 only when
    all three 5-minute power samples exist; 1-2 samples -> 0 with the mean
    kept; zero samples -> NaN Target (dropped by the formal loader, same as
    site9a).
  * SolarGeometryProxy is the GHI copy used historically.
  * SolarZenithCos = clip(GHI / c, 0, 1); c is the 0.95 quantile of clock-time
    11:00-13:00 GHI (clear midday typical maximum). The legacy constants
    recovered from site31/site9a (1174.55614 / 1181.33686) are matched within
    0.6% by this documented rule; the column is a linear GHI rescale and is
    z-scored before modelling.
  * ClearSkyGHIProxy = trailing 24 h rolling maximum of GHI (not zeroed at
    night), identical to the legacy DKASC artefacts; the formal loader audits
    it out as a clear-sky trajectory on every DKASC site.
  * SolarDaySin/Cos use clock time only (legacy artefacts match with RMSE 0).
  * SolarElevationMask = (SolarZenithCos > 0.02).

Usage:
    python preprocess_dkasc_new_sites.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_START = "2018-02-01 00:00:00"
WINDOW_END = "2020-01-31 23:45:00"

RAW_COLUMNS = [
    "timestamp",
    "Active_Power",
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Wind_Direction",
    "Weather_Daily_Rainfall",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

WEATHER_MEAN = [
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Wind_Direction",
    "Weather_Daily_Rainfall",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

OUTPUT_COLUMNS = [
    "date",
    "Target",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Wind_Direction",
    "Wind_Speed",
    "Weather_Daily_Rainfall",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "TargetObserved",
    "SolarZenithCos",
    "SolarGeometryProxy",
    "ClearSkyGHIProxy",
    "SolarDaySin",
    "SolarDayCos",
    "SolarElevationMask",
]

SITES = [
    ("data/91-Site_DKA-M9_B-Phase.csv", "data/DKASC_site1a_15min.csv", "dkasc_site1a"),
    ("data/98-Site_DKA-M8_B-Phase.csv", "data/DKASC_site19_15min.csv", "dkasc_site19"),
]


def _slot_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame["slot"] = frame["timestamp"].dt.floor("15min")

    grouped = frame.groupby("slot")
    weather = grouped[WEATHER_MEAN].mean()
    power = grouped["Active_Power"]
    target = power.mean().clip(lower=0.0).rename("Target")
    valid_count = power.count().rename("_valid")
    out = weather.join(target).join(valid_count)
    out["TargetObserved"] = (out["_valid"] == 3).astype(np.float64)
    partial = (out["_valid"] > 0) & (out["_valid"] < 3)
    out.loc[partial, "TargetObserved"] = 0.0
    out.loc[out["_valid"] == 0, "Target"] = np.nan
    out = out.drop(columns="_valid")
    return out


def preprocess(raw_path: str, output_path: str, label: str) -> None:
    frame = pd.read_csv(raw_path, usecols=RAW_COLUMNS)
    slots = _slot_aggregate(frame)

    # Clear-sky envelope is computed over the whole available history before
    # slicing, so the first window day has a real trailing 24 h envelope.
    slots["ClearSkyGHIProxy"] = (
        slots["Global_Horizontal_Radiation"]
        .rolling(96, min_periods=1)
        .max()
    )

    window = slots.loc[
        pd.Timestamp(WINDOW_START) : pd.Timestamp(WINDOW_END)
    ].copy()
    full_index = pd.date_range(WINDOW_START, WINDOW_END, freq="15min")
    window = window.reindex(full_index)
    assert len(window) == 70080, len(window)

    ghi = window["Global_Horizontal_Radiation"]
    clock_minutes = (
        window.index.hour * 60 + window.index.minute
    ).to_numpy(dtype=np.float64)
    midday = (clock_minutes >= 660) & (clock_minutes <= 780)
    scale = float(np.nanquantile(ghi.to_numpy()[midday], 0.95))
    window["SolarGeometryProxy"] = ghi
    window["SolarZenithCos"] = (ghi / scale).clip(0.0, 1.0)
    angle = 2.0 * np.pi * clock_minutes / 1440.0
    window["SolarDaySin"] = np.sin(angle)
    window["SolarDayCos"] = np.cos(angle)
    window["SolarElevationMask"] = (
        window["SolarZenithCos"] > 0.02
    ).astype(np.float64)

    window = window.reset_index(names="date")
    window["date"] = window["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    window = window[OUTPUT_COLUMNS]
    window.to_csv(output_path, index=False)

    observed = window["TargetObserved"].mean()
    dropped = int(window["Target"].isna().sum())
    print(
        f"{label}: {Path(output_path).name} rows={len(window)} "
        f"observed={observed:.4f} empty_slots={dropped} ghi_scale={scale:.3f}"
    )
    for col in WEATHER_MEAN:
        frac = float(window[col].notna().mean())
        if frac < 0.999:
            print(f"  note: {col} finite fraction {frac:.4f}")


if __name__ == "__main__":
    for raw, out, label in SITES:
        preprocess(raw, out, label)
