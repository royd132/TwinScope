"""H1-H24 error attribution: numerical backbone vs PSRC vs LSTM/GRU.

Produces, per station:
  * RMSE_h / MAE_h curves for the backbone (yhat0), PSRC (three seeds),
    LSTM and GRU (anchor seed only);
  * Gain_h = (RMSE_base,h - RMSE_psrc,h) / RMSE_base,h * 100 %;
  * the mean per-horizon gate g_h overlaid on the gain panel.

The mechanism claim examined here: h up => g_h up => PSRC gain up.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.artifacts import (
    DATASETS,
    DEFAULT_FINAL_DIR,
    PSRC_SEEDS,
    SINGLE_SEED,
    PredictionDump,
    load_dump,
)

STATION_LABELS = {
    "dkasc_site31": "DKASC Site 31",
    "dkasc_site9a": "DKASC Site 9A",
    "dkasc_site1a": "DKASC Site 1A",
    "dkasc_site19": "DKASC Site 19",
    "pvod_station00": "PVOD Station 00",
    "pvod_station01": "PVOD Station 01",
    "pvod_station02": "PVOD Station 02",
    "pvod_station03": "PVOD Station 03",
    "hkust": "HKUST",
}


def horizon_errors(y: np.ndarray, yhat: np.ndarray) -> dict[str, np.ndarray]:
    error = yhat - y
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "mbe": np.mean(error, axis=0),
    }


def horizon_table_for_station(
    dataset: str,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> pd.DataFrame:
    rows = []
    psrc_dumps = [
        load_dump(dataset, "psrc", seed, horizon=horizon, final_dir=final_dir)
        for seed in seeds
    ]
    hours = psrc_dumps[0].hours_ahead

    for seed, dump in zip(seeds, psrc_dumps):
        metrics0 = horizon_errors(dump.y, dump.yhat0)
        metrics = horizon_errors(dump.y, dump.yhat)
        for h in range(dump.horizon):
            rows.append(
                {
                    "dataset": dataset,
                    "model": "backbone",
                    "seed": seed,
                    "horizon": h + 1,
                    "hours_ahead": hours[h],
                    "rmse": metrics0["rmse"][h],
                    "mae": metrics0["mae"][h],
                    "mbe": metrics0["mbe"][h],
                    "nrmse": metrics0["rmse"][h] / dump.capacity,
                    "gate_mean": np.nan,
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": "psrc",
                    "seed": seed,
                    "horizon": h + 1,
                    "hours_ahead": hours[h],
                    "rmse": metrics["rmse"][h],
                    "mae": metrics["mae"][h],
                    "mbe": metrics["mbe"][h],
                    "nrmse": metrics["rmse"][h] / dump.capacity,
                    "gate_mean": np.nanmean(dump.gate[:, h]),
                }
            )

    for model in ("lstm", "gru"):
        dump = load_dump(
            dataset, model, SINGLE_SEED, horizon=horizon, final_dir=final_dir
        )
        metrics = horizon_errors(dump.y, dump.yhat)
        for h in range(dump.horizon):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": SINGLE_SEED,
                    "horizon": h + 1,
                    "hours_ahead": hours[h],
                    "rmse": metrics["rmse"][h],
                    "mae": metrics["mae"][h],
                    "mbe": metrics["mbe"][h],
                    "nrmse": metrics["rmse"][h] / dump.capacity,
                    "gate_mean": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_horizon_tables(
    output_dir: Path,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> pd.DataFrame:
    frames = [
        horizon_table_for_station(
            dataset, horizon=horizon, seeds=seeds, final_dir=final_dir
        )
        for dataset in datasets
    ]
    table = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "horizon_metrics.csv", index=False)
    summary = (
        table.groupby(["dataset", "model", "horizon", "hours_ahead"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            gate_mean=("gate_mean", "mean"),
        )
        .fillna(0.0)
    )
    summary.to_csv(output_dir / "horizon_metrics_summary.csv", index=False)
    return table


def _series(table: pd.DataFrame, model: str, metric: str):
    group = (
        table[table["model"] == model]
        .groupby("horizon")[metric]
        .agg(["mean", "std"])
        .reindex(range(1, 25))
        .fillna(0.0)
    )
    return group["mean"].to_numpy(), group["std"].to_numpy()


def plot_station(
    dataset: str,
    table: pd.DataFrame,
    output_dir: Path,
    horizon: int = 24,
) -> None:
    station = table[table["dataset"] == dataset]
    hours = np.arange(1, horizon + 1) * 0.25
    base_mean, base_std = _series(station, "backbone", "rmse", horizon)
    psrc_mean, psrc_std = _series(station, "psrc", "rmse", horizon)
    lstm_mean, _ = _series(station, "lstm", "rmse", horizon)
    gru_mean, _ = _series(station, "gru", "rmse", horizon)
    gate = (
        station[station["model"] == "psrc"]
        .groupby("horizon")["gate_mean"]
        .mean()
        .reindex(range(1, horizon + 1))
        .to_numpy()
    )
    gain = (base_mean - psrc_mean) / np.maximum(base_mean, 1e-12) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(hours, base_mean, color="#7f7f7f", lw=2, label="Numerical backbone")
    ax1.fill_between(
        hours, base_mean - base_std, base_mean + base_std, color="#7f7f7f", alpha=0.15
    )
    ax1.plot(hours, psrc_mean, color="#1f4e79", lw=2, label="PSRC")
    ax1.fill_between(
        hours, psrc_mean - psrc_std, psrc_mean + psrc_std, color="#1f4e79", alpha=0.18
    )
    ax1.plot(hours, lstm_mean, color="#d98c00", lw=1.6, ls="--", label="LSTM (s2026)")
    ax1.plot(hours, gru_mean, color="#5b9bd5", lw=1.6, ls="--", label="GRU (s2026)")
    ax1.set_xlabel("Forecast horizon (hours ahead)")
    ax1.set_ylabel("RMSE (MW)")
    ax1.set_title(STATION_LABELS[dataset])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    color_gain = "#2e7d32"
    ax2.plot(hours, gain, color=color_gain, lw=2, label="PSRC gain over backbone")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("Forecast horizon (hours ahead)")
    ax2.set_ylabel("RMSE gain (%)", color=color_gain)
    ax2.tick_params(axis="y", labelcolor=color_gain)
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(hours, gate, color="#c0392b", lw=1.6, ls=":", label="mean gate $g_h$")
    ax2b.set_ylabel("mean gate $g_h$", color="#c0392b")
    ax2b.tick_params(axis="y", labelcolor="#c0392b")
    ax2b.set_ylim(0, 1)
    ax2.set_title("Calibration gain and gate vs horizon")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"horizon_{dataset}.png", dpi=160)
    plt.close(fig)


def plot_combined_gain(
    table: pd.DataFrame, output_dir: Path, horizon: int = 24
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    hours = np.arange(1, horizon + 1) * 0.25
    for dataset in [name for name in DATASETS if name in set(table["dataset"])]:
        station = table[table["dataset"] == dataset]
        base_mean, _ = _series(station, "backbone", "rmse", horizon)
        psrc_mean, _ = _series(station, "psrc", "rmse", horizon)
        gain = (base_mean - psrc_mean) / np.maximum(base_mean, 1e-12) * 100.0
        ax.plot(hours, gain, lw=1.8, label=STATION_LABELS[dataset])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("PSRC RMSE gain over backbone (%)")
    ax.set_title("Per-horizon gain across stations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_gain_all_stations.png", dpi=160)
    plt.close(fig)


def run(
    output_dir: Path,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, object]:
    table = build_horizon_tables(
        output_dir, horizon=horizon, seeds=seeds, final_dir=final_dir,
        datasets=datasets,
    )
    figures = output_dir / "figures"
    for dataset in datasets:
        plot_station(dataset, table, figures, horizon=horizon)
    plot_combined_gain(table, figures, horizon=horizon)
    per_seed_rmse = (
        table.groupby(["dataset", "model", "seed"], as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "seed_rmse"})
    )

    def _seed_std(values: pd.Series) -> float:
        return float(values.std()) if len(values) > 1 else 0.0

    overall = (
        per_seed_rmse.groupby(["dataset", "model"], as_index=False)
        .agg(rmse_mean=("seed_rmse", "mean"), rmse_seed_std=("seed_rmse", _seed_std))
    )
    overall_path = output_dir / "horizon_overall.json"
    payload = json.loads(overall.to_json(orient="records"))
    overall_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"rows": int(len(table)), "overall": payload}
