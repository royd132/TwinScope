"""Deterministic case-study window selection from prediction dumps.

No random selection, no overlap between showcase windows.  "Daytime" is
defined by observed generation (no site-local timezone assumptions), and
two kinds of difficult regimes are chosen: the windows with the largest
mean backbone absolute error, and the windows containing the steepest
observed 15-min power ramp (cloud-transition analogue).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.artifacts import DEFAULT_FINAL_DIR, SINGLE_SEED, load_dump
from analysis.horizon import DATASETS, STATION_LABELS

STEP = np.timedelta64(15, "m")
MIN_SEPARATION_STEPS = 96  # selected windows must be >= 24 h apart
DAYTIME_FRACTION = 0.20
CASES_PER_KIND = 2


def _candidate_frame(dump) -> pd.DataFrame:
    target_mean = dump.y.mean(axis=1)
    backbone_mae = np.abs(dump.yhat0 - dump.y).mean(axis=1)
    ramp = np.abs(np.diff(dump.y, axis=1)).max(axis=1)
    return pd.DataFrame(
        {
            "window": np.arange(dump.y.shape[0]),
            "window_start": dump.timestamps,
            "daytime": target_mean > DAYTIME_FRACTION * dump.capacity,
            "backbone_mae": backbone_mae,
            "ramp": ramp,
        }
    )


def _greedy_pick(frame: pd.DataFrame, column: str, count: int) -> list[int]:
    """Pick the strongest windows, forbidding near-duplicate/overlapping cases."""
    ordered = frame.sort_values(
        [column, "window"], ascending=[False, True]
    )["window"].to_numpy()
    chosen: list[int] = []
    for candidate in ordered:
        if all(abs(int(candidate) - prior) >= MIN_SEPARATION_STEPS for prior in chosen):
            chosen.append(int(candidate))
        if len(chosen) >= count:
            break
    return chosen


def select_cases(
    dataset: str,
    horizon: int = 24,
    seed: int = SINGLE_SEED,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> pd.DataFrame:
    dump = load_dump(dataset, "psrc", seed, horizon=horizon, final_dir=final_dir)
    frame = _candidate_frame(dump)
    daytime = frame[frame["daytime"]]
    error_cases = _greedy_pick(daytime, "backbone_mae", CASES_PER_KIND)
    ramp_pool = daytime[
        daytime["window"].apply(
            lambda window: all(
                abs(int(window) - prior) >= MIN_SEPARATION_STEPS
                for prior in error_cases
            )
        )
    ]
    ramp_cases = _greedy_pick(ramp_pool, "ramp", CASES_PER_KIND)
    # One representative medium-error daytime window: closest to the median
    # daytime backbone MAE, kept away from the already chosen dates.
    excluded = set(error_cases) | set(ramp_cases)
    remaining = daytime[~daytime["window"].isin(excluded)]
    median_error = float(remaining["backbone_mae"].median())
    typical = remaining.assign(
        distance=(remaining["backbone_mae"] - median_error).abs()
    ).sort_values(["distance", "window"])
    typical_case = None
    for row in typical.itertuples():
        if all(
            abs(int(row.window) - prior) >= MIN_SEPARATION_STEPS
            for prior in excluded
        ):
            typical_case = int(row.window)
            break
    cases = []
    for window in error_cases:
        cases.append((window, "high backbone error"))
    for window in ramp_cases:
        cases.append((window, "steepest observed power ramp"))
    if typical_case is not None:
        cases.append((typical_case, "median-error daytime"))
    rows = []
    for window, reason in cases:
        record = frame.loc[frame["window"] == window].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "window": int(window),
                "reason": reason,
                "window_start": str(record["window_start"]),
                "backbone_mae": float(record["backbone_mae"]),
                "max_power_ramp_mw": float(record["ramp"]),
            }
        )
    return pd.DataFrame(rows)


def plot_case(dump, window: int, reason: str, figures: Path) -> None:
    start = dump.timestamps[window]
    stamps = start + np.arange(dump.horizon) * STEP
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 5.2), sharex=True, height_ratios=[3, 1]
    )
    ax1.plot(stamps, dump.y[window], color="black", lw=2.0, label="Observed")
    ax1.plot(
        stamps,
        dump.yhat0[window],
        color="#7f7f7f",
        lw=1.6,
        label="Numerical backbone",
    )
    ax1.plot(
        stamps,
        dump.yhat[window],
        color="#1f4e79",
        lw=1.8,
        label="PSRC",
    )
    ax1.set_ylabel("Power (MW)")
    title = f"{STATION_LABELS[dump.dataset]} | {reason} | start {start}"
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.plot(stamps, dump.gate[window], color="#c0392b", lw=1.6)
    ax2.set_ylabel("gate $g$")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    filename = f"case_{dump.dataset}_w{window:06d}.png"
    fig.savefig(figures / filename, dpi=160)
    plt.close(fig)


def run(
    output_dir: Path,
    horizon: int = 24,
    seed: int = SINGLE_SEED,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, object]:
    figures = output_dir / "figures" / "cases"
    figures.mkdir(parents=True, exist_ok=True)
    all_frames = []
    plotted = 0
    for dataset in datasets:
        cases = select_cases(
            dataset, horizon=horizon, seed=seed, final_dir=final_dir
        )
        all_frames.append(cases)
        dump = load_dump(
            dataset, "psrc", seed, horizon=horizon, final_dir=final_dir
        )
        for row in cases.itertuples():
            plot_case(dump, row.window, row.reason, figures)
            plotted += 1
    table = pd.concat(all_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "case_study_windows.csv", index=False)
    return {"cases": json.loads(table.to_json(orient="records")), "figures": plotted}
