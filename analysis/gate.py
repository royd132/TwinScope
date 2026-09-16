"""Gate behaviour diagnosis (primarily PVOD Station 01).

Questions answered, from PSRC dumps only (no tuning follows automatically):
  * Does a high gate coincide with high numerical-backbone error?
  * Does calibration reduce error inside high-gate regimes?
  * How do these relationships evolve with the forecast horizon?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.artifacts import DATASETS, DEFAULT_FINAL_DIR, PSRC_SEEDS
from analysis.horizon import STATION_LABELS


def _seed_frame(
    dataset: str, seed: int, horizon: int = 24, final_dir: Path = DEFAULT_FINAL_DIR
) -> pd.DataFrame:
    from analysis.artifacts import load_dump

    dump = load_dump(dataset, "psrc", seed, horizon=horizon, final_dir=final_dir)
    n_windows, horizon = dump.y.shape
    horizon_index, window_index = np.meshgrid(
        np.arange(horizon), np.arange(n_windows)
    )
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "seed": seed,
            "window": window_index.reshape(-1),
            "horizon": (horizon_index.reshape(-1) + 1),
            "gate": dump.gate.reshape(-1),
            "base_abs_err": np.abs(dump.yhat0 - dump.y).reshape(-1),
            "psrc_abs_err": np.abs(dump.yhat - dump.y).reshape(-1),
        }
    )
    frame["improved"] = frame["psrc_abs_err"] < frame["base_abs_err"]
    return frame


def station_diagnosis(
    dataset: str,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> dict[str, object]:
    frames = [
        _seed_frame(dataset, seed, horizon=horizon, final_dir=final_dir)
        for seed in seeds
    ]
    table = pd.concat(frames, ignore_index=True)

    def _corr(frame: pd.DataFrame) -> dict[str, float]:
        corr_base = spearmanr(frame["gate"], frame["base_abs_err"]).statistic
        corr_psrc = spearmanr(frame["gate"], frame["psrc_abs_err"]).statistic
        return {
            "spearman_gate_vs_base_abs_err": float(corr_base),
            "spearman_gate_vs_psrc_abs_err": float(corr_psrc),
        }

    per_seed = [_corr(frame) for frame in frames]
    pooled = _corr(table)

    # Gate deciles within each seed, then average the bin statistics.
    binned_rows = []
    for frame in frames:
        bins = pd.qcut(frame["gate"], q=10, labels=False, duplicates="drop")
        grouped = frame.groupby(bins, observed=True)
        for bin_id, part in grouped:
            binned_rows.append(
                {
                    "bin": int(bin_id),
                    "gate_mean": float(part["gate"].mean()),
                    "base_mae": float(part["base_abs_err"].mean()),
                    "psrc_mae": float(part["psrc_abs_err"].mean()),
                    "improved_share": float(part["improved"].mean()),
                }
            )
    binned = (
        pd.DataFrame(binned_rows)
        .groupby("bin", as_index=False)
        .mean()
        .sort_values("bin")
    )

    per_horizon_rows = []
    for h in range(1, frames[0]["horizon"].max() + 1):
        rows_h = []
        for frame in frames:
            part = frame[frame["horizon"] == h]
            rows_h.append(
                {
                    "horizon": h,
                    "corr_base": spearmanr(
                        part["gate"], part["base_abs_err"]
                    ).statistic,
                    "corr_psrc": spearmanr(
                        part["gate"], part["psrc_abs_err"]
                    ).statistic,
                    "gate_mean": part["gate"].mean(),
                    "improved_share": part["improved"].mean(),
                }
            )
        per_horizon_rows.append(
            pd.DataFrame(rows_h).groupby("horizon", as_index=False).mean().iloc[0]
            .to_dict()
        )
    per_horizon = pd.DataFrame(per_horizon_rows)

    high = table[table["gate"] >= table["gate"].quantile(0.9)]
    summary = {
        "dataset": dataset,
        "pooled": pooled,
        "per_seed": per_seed,
        "gate_global_mean": float(table["gate"].mean()),
        "improved_share_all": float(table["improved"].mean()),
        "improved_share_top_gate_decile": float(
            binned.iloc[-1]["improved_share"]
        ),
        "base_mae_top_gate_decile": float(binned.iloc[-1]["base_mae"]),
        "psrc_mae_top_gate_decile": float(binned.iloc[-1]["psrc_mae"]),
        "high_gate_regime": {
            "threshold_gate_p90": float(table["gate"].quantile(0.9)),
            "base_mae": float(high["base_abs_err"].mean()),
            "psrc_mae": float(high["psrc_abs_err"].mean()),
            "improved_share": float(high["improved"].mean()),
        },
    }
    return {
        "summary": summary,
        "binned": binned,
        "per_horizon": per_horizon,
    }


def plot_diagnosis(dataset: str, diagnosis: dict[str, object], figures: Path) -> None:
    binned = diagnosis["binned"]
    per_horizon = diagnosis["per_horizon"]
    hours = per_horizon["horizon"].to_numpy() * 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    axes[0].plot(binned["gate_mean"], binned["base_mae"], "o-", label="backbone MAE")
    axes[0].plot(
        binned["gate_mean"], binned["psrc_mae"], "s-", label="PSRC MAE"
    )
    axes[0].set_xlabel("Gate decile (mean gate)")
    axes[0].set_ylabel("MAE (MW)")
    axes[0].set_title("Error vs gate regime")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].bar(binned["gate_mean"], binned["improved_share"] * 100.0, width=0.03)
    axes[1].set_xlabel("Gate decile (mean gate)")
    axes[1].set_ylabel("Windows improved by calibration (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Correction success rate")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].plot(
        hours, per_horizon["corr_base"], lw=1.8, label=r"corr($g$, $|\hat y_0-y|$)"
    )
    axes[2].plot(
        hours,
        per_horizon["corr_psrc"],
        lw=1.8,
        ls="--",
        label=r"corr($g$, $|\hat y_{PSRC}-y|$)",
    )
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_xlabel("Forecast horizon (hours ahead)")
    axes[2].set_ylabel("Spearman correlation")
    axes[2].set_title("Gate-error correlation vs horizon")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(STATION_LABELS[dataset])
    fig.tight_layout()
    fig.savefig(figures / f"gate_diagnosis_{dataset}.png", dpi=160)
    plt.close(fig)


def run(
    output_dir: Path,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, object]:
    tables_dir = output_dir / "gate_tables"
    figures = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in datasets:
        diagnosis = station_diagnosis(
            dataset, horizon=horizon, seeds=seeds, final_dir=final_dir
        )
        summaries.append(diagnosis["summary"])
        diagnosis["binned"].to_csv(
            tables_dir / f"gate_deciles_{dataset}.csv", index=False
        )
        diagnosis["per_horizon"].to_csv(
            tables_dir / f"gate_horizon_{dataset}.csv", index=False
        )
        plot_diagnosis(dataset, diagnosis, figures)
    (output_dir / "gate_diagnosis.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return {"stations": summaries}
