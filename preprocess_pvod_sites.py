"""One-off preprocessing for the two additional PVOD sites (2026-09-14).

Raw files supplied by the user (upstream PVOD 15-min export, UTC Timestamp plus
a Beijing-time ``Time`` column, NWP/LMD measurements, raw ``power`` in kW):

* ``data/S-3.csv`` -> Location ``station07`` (36.64403 N, 113.64187 E)
* ``data/S-4.csv`` -> Location ``station08`` (36.70761 N, 113.89999 E)

They are converted into the exact processed schema used by
``data/PVOD_station00_15min.csv`` / ``data/PVOD_station01_15min.csv`` (16
columns, local-time index, LMD measurements only, raw power as Target,
cyclic wind encoding, deterministic solar geometry + clear-sky proxy).

The solar recipe was reverse-engineered against the two existing processed
PVOD files (RMSE at CSV rounding precision: zenith cosine 1e-4, solar-day
sin/cos < 1e-4, clear-sky GHI < 1 W/m^2, daylight mask 1 boundary row in
~30k):

* local time = upstream ``Time`` column (Timestamp UTC + 8 h);
* equation of time: Spencer (1971), declination: Cooper's approximation;
* apparent solar time = clock minutes + 4*(lon-120 deg) + EoT minutes;
* SolarZenithCos = SolarGeometryProxy = max(0, cos zenith);
* SolarDaySin/Cos cyclic encoding of the apparent solar time;
* SolarElevationMask = SolarZenithCos > 0.02;
* ClearSkyGHIProxy = max(0, 986.32 z + 180.38 z^2 - 94.91 z^3 - 39.02),
  the station-independent clear-sky relation fitted on station00/01
  (cross-station transfer RMSE 0.87 W/m^2).

Outputs (naming continues the anonymised station00/01 series; the upstream
Location ids are preserved in this docstring):

* ``data/PVOD_station02_15min.csv``  <- S-3.csv / station07
* ``data/PVOD_station03_15min.csv``  <- S-4.csv / station08

Chronological splits are applied downstream by formal.data (70/15/15,
contiguous-window, train-only normalisation), exactly as for every other
site; this script only reshapes the raw export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JOBS = (
    ("data/S-3.csv", "data/PVOD_station02_15min.csv"),
    ("data/S-4.csv", "data/PVOD_station03_15min.csv"),
)

OUTPUT_COLUMNS = [
    "date",
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_windspeed",
    "TargetObserved",
    "Target",
    "WindDirectionSin",
    "WindDirectionCos",
    "SolarZenithCos",
    "SolarGeometryProxy",
    "ClearSkyGHIProxy",
    "SolarDaySin",
    "SolarDayCos",
    "SolarElevationMask",
]


def equation_of_time_minutes(day_of_year: np.ndarray) -> np.ndarray:
    """Spencer (1971) equation of time, in minutes."""
    gamma = 2.0 * np.pi * (day_of_year - 1) / 365.0
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )


def solar_declination_radians(day_of_year: np.ndarray) -> np.ndarray:
    """Cooper's approximation."""
    return np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + day_of_year) / 365.0)))


def preprocess(raw_path: str, output_path: str) -> None:
    frame = pd.read_csv(raw_path)
    required = {
        "Time",
        "lmd_totalirrad",
        "lmd_diffuseirrad",
        "lmd_temperature",
        "lmd_pressure",
        "lmd_winddirection",
        "lmd_windspeed",
        "power",
        "Longitude",
        "Latitude",
        "Location",
    }
    missing = required.difference(frame)
    if missing:
        raise ValueError(f"{raw_path} misses columns: {sorted(missing)}")

    location = str(frame["Location"].iloc[0])
    longitude = float(frame["Longitude"].iloc[0])
    latitude = float(frame["Latitude"].iloc[0])

    local_time = pd.to_datetime(frame["Time"])
    if local_time.isna().any():
        raise ValueError(f"{raw_path}: unparseable local timestamps")
    if not local_time.is_monotonic_increasing or local_time.duplicated().any():
        raise ValueError(f"{raw_path}: local timestamps must be ordered and unique")

    day_of_year = local_time.dt.dayofyear.to_numpy(dtype=np.float64)
    clock_minutes = (
        local_time.dt.hour * 60
        + local_time.dt.minute
        + local_time.dt.second / 60.0
    ).to_numpy(dtype=np.float64)

    eot = equation_of_time_minutes(day_of_year)
    declination = solar_declination_radians(day_of_year)
    solar_minutes = clock_minutes + 4.0 * (longitude - 120.0) + eot
    hour_angle = np.radians(15.0 * (solar_minutes / 60.0 - 12.0))

    sin_lat = np.sin(np.radians(latitude))
    cos_lat = np.cos(np.radians(latitude))
    zenith_cos = np.clip(
        sin_lat * np.sin(declination)
        + cos_lat * np.cos(declination) * np.cos(hour_angle),
        0.0,
        None,
    )
    solar_day_sin = np.sin(2.0 * np.pi * solar_minutes / 1440.0)
    solar_day_cos = np.cos(2.0 * np.pi * solar_minutes / 1440.0)
    clear_sky = np.clip(
        986.32 * zenith_cos
        + 180.38 * zenith_cos**2
        - 94.91 * zenith_cos**3
        - 39.02,
        0.0,
        None,
    )
    daylight_mask = (zenith_cos > 0.02).astype(np.float64)

    wind_radians = np.radians(frame["lmd_winddirection"].to_numpy(dtype=np.float64))
    power = pd.to_numeric(frame["power"], errors="coerce")
    observed = power.notna().astype(np.float64)
    target = power.fillna(0.0).clip(lower=0.0)

    processed = pd.DataFrame(
        {
            "date": local_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "lmd_totalirrad": pd.to_numeric(frame["lmd_totalirrad"], errors="coerce"),
            "lmd_diffuseirrad": pd.to_numeric(frame["lmd_diffuseirrad"], errors="coerce"),
            "lmd_temperature": pd.to_numeric(frame["lmd_temperature"], errors="coerce"),
            "lmd_pressure": pd.to_numeric(frame["lmd_pressure"], errors="coerce"),
            "lmd_windspeed": pd.to_numeric(frame["lmd_windspeed"], errors="coerce"),
            "TargetObserved": observed,
            "Target": target,
            "WindDirectionSin": np.sin(wind_radians),
            "WindDirectionCos": np.cos(wind_radians),
            "SolarZenithCos": zenith_cos,
            "SolarGeometryProxy": zenith_cos,
            "ClearSkyGHIProxy": clear_sky,
            "SolarDaySin": solar_day_sin,
            "SolarDayCos": solar_day_cos,
            "SolarElevationMask": daylight_mask,
        }
    )[OUTPUT_COLUMNS]

    if processed.isna().any().any():
        raise ValueError(f"{raw_path}: processed frame contains NaNs")
    processed.to_csv(output_path, index=False)
    print(
        f"[preprocess] {raw_path} ({location}, {latitude:.5f}N {longitude:.5f}E) "
        f"-> {output_path}: {len(processed)} rows"
    )


def main() -> int:
    for raw_path, output_path in JOBS:
        if not Path(raw_path).is_file():
            raise FileNotFoundError(raw_path)
        preprocess(raw_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
