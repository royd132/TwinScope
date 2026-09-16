"""Paired moving-block bootstrap tests on forecast loss differentials.

For two aligned models A (PSRC) and B (backbone or strongest baseline) on
the same test windows::

    d_i = mean_h L(e_A,i,h) - mean_h L(e_B,i,h)

Windows stride at 15 min and overlap across h, so the d_i sequence is
serially correlated.  A moving block bootstrap over contiguous windows
preserves that dependence.  We report the percentage reduction in mean
loss with a 95% percentile confidence interval.  Aggregating 4 stations x
4 metrics into one Wilcoxon/DM count is deliberately avoided.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.artifacts import (
    DATASETS,
    DEFAULT_FINAL_DIR,
    PSRC_SEEDS,
    SINGLE_SEED,
    assert_aligned,
    load_dump,
)
from analysis.horizon import STATION_LABELS

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
PRIMARY_BLOCK = 96  # one day of 15-min windows
SENSITIVITY_BLOCKS = (48, 96, 192)

# Strongest external baselines per the pre-fix formal table; the runner-up
# is included at the stations where the top spot is contested.
EXTERNAL_TARGETS = {
    "dkasc_site31": [("ampdnet", SINGLE_SEED)],
    "dkasc_site9a": [("lstm", SINGLE_SEED), ("gru", SINGLE_SEED)],
    "pvod_station00": [("timemixer", SINGLE_SEED), ("gru", SINGLE_SEED)],
    "pvod_station01": [("lstm", SINGLE_SEED)],
}


def _window_loss(dump, loss: str) -> np.ndarray:
    error = dump.yhat - dump.y
    point = error**2 if loss == "squared" else np.abs(error)
    return point.mean(axis=1)


def block_bootstrap_reduction(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    block: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Percent reduction of A's mean loss relative to B, block bootstrap CI."""
    if loss_a.shape != loss_b.shape:
        raise ValueError("paired loss series must share shape")
    n = loss_a.shape[0]
    if not 0 < block <= n:
        raise ValueError("block length must lie in (0, n_windows]")
    differential = loss_a - loss_b
    observed_diff = float(differential.mean())
    observed_base = float(loss_b.mean())
    observed = -100.0 * observed_diff / observed_base

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    reductions = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        index = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        mean_diff = float(differential[index].mean())
        mean_base = float(loss_b[index].mean())
        reductions[replicate] = -100.0 * mean_diff / max(mean_base, 1e-12)
    low, high = np.percentile(reductions, [2.5, 97.5])
    # Two-sided bootstrap p-value against zero reduction.
    positive_fraction = float(np.mean(reductions >= 0.0))
    p_value = 2.0 * min(positive_fraction, 1.0 - positive_fraction)
    return {
        "block_windows": int(block),
        "observed_reduction_pct": observed,
        "ci95_low_pct": float(low),
        "ci95_high_pct": float(high),
        "p_value_two_sided": min(1.0, p_value),
        "replicates": replicates,
    }


def compare_pair(
    dataset: str,
    model_a: str,
    seed_a: int,
    model_b: str,
    seed_b: int,
    loss: str,
    blocks: tuple[int, ...] = SENSITIVITY_BLOCKS,
    horizon: int = 24,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> dict[str, object]:
    dump_a = load_dump(
        dataset, model_a, seed_a, horizon=horizon, final_dir=final_dir
    )
    dump_b = load_dump(
        dataset, model_b, seed_b, horizon=horizon, final_dir=final_dir
    )
    assert_aligned(dump_a, dump_b)
    loss_a = _window_loss(dump_a, loss)
    loss_b = _window_loss(dump_b, loss)
    return {
        "dataset": dataset,
        "model_a": model_a,
        "seed_a": seed_a,
        "model_b": model_b,
        "seed_b": seed_b,
        "loss": loss,
        "n_windows": int(loss_a.shape[0]),
        "blocks": [
            block_bootstrap_reduction(loss_a, loss_b, block) for block in blocks
        ],
    }


def backbone_comparisons(
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> list[dict[str, object]]:
    """PSRC vs the numerical backbone inside the same trained model."""
    results = []
    for dataset in datasets:
        for seed in seeds:
            dump = load_dump(
                dataset, "psrc", seed, horizon=horizon, final_dir=final_dir
            )
            for loss in ("squared", "absolute"):
                error = dump.yhat - dump.y
                base_error = dump.yhat0 - dump.y
                point = error**2 if loss == "squared" else np.abs(error)
                base_point = (
                    base_error**2 if loss == "squared" else np.abs(base_error)
                )
                results.append(
                    {
                        "dataset": dataset,
                        "model_a": "psrc",
                        "seed_a": seed,
                        "model_b": "backbone",
                        "seed_b": seed,
                        "loss": loss,
                        "n_windows": int(dump.y.shape[0]),
                        "blocks": [
                            block_bootstrap_reduction(
                                point.mean(axis=1),
                                base_point.mean(axis=1),
                                block,
                            )
                            for block in SENSITIVITY_BLOCKS
                        ],
                    }
                )
    return results


def external_comparisons(
    horizon: int = 24,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> list[dict[str, object]]:
    results = []
    for dataset in datasets:
        for model_b, seed_b in EXTERNAL_TARGETS.get(dataset, []):
            for loss in ("squared", "absolute"):
                results.append(
                    compare_pair(
                        dataset,
                        "psrc",
                        SINGLE_SEED,
                        model_b,
                        seed_b,
                        loss,
                        horizon=horizon,
                        final_dir=final_dir,
                    )
                )
    return results


def plot_primary_intervals(payload: dict[str, object], figures: Path) -> None:
    rows = []
    for record in payload["comparisons"]:
        if record["loss"] != "squared":
            continue
        primary = next(
            block for block in record["blocks"] if block["block_windows"] == PRIMARY_BLOCK
        )
        if record["model_b"] == "backbone" and record["seed_a"] != SINGLE_SEED:
            continue
        rows.append(
            {
                "dataset": record["dataset"],
                "label": (
                    "vs backbone"
                    if record["model_b"] == "backbone"
                    else f"vs {record['model_b']}"
                ),
                "reduction": primary["observed_reduction_pct"],
                "low": primary["ci95_low_pct"],
                "high": primary["ci95_high_pct"],
            }
        )
    lookup = {(row["dataset"], row["label"]): row for row in rows}
    datasets = list(DATASETS)
    labels = sorted({row["label"] for row in rows})
    x = np.arange(len(datasets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for index, label in enumerate(labels):
        selected = [lookup.get((dataset, label)) for dataset in datasets]
        reductions = [row["reduction"] if row else np.nan for row in selected]
        lower = [
            row["reduction"] - row["low"] if row else np.nan for row in selected
        ]
        upper = [
            row["high"] - row["reduction"] if row else np.nan for row in selected
        ]
        positions = x + (index - 0.5) * width
        ax.bar(positions, reductions, width, label=label)
        ax.errorbar(
            positions,
            reductions,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1.0,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([STATION_LABELS[d].replace("DKASC ", "").replace("PVOD ", "") for d in datasets])
    ax.set_ylabel("Squared-error reduction of PSRC (%)")
    ax.set_title("Paired block-bootstrap reductions (block = 96 windows, 95% CI)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "bootstrap_reduction_ci.png", dpi=160)
    plt.close(fig)


def run(
    output_dir: Path,
    horizon: int = 24,
    seeds: tuple[int, ...] = PSRC_SEEDS,
    final_dir: Path = DEFAULT_FINAL_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, object]:
    payload = {
        "method": (
            "paired moving block bootstrap on per-window mean loss "
            "differentials; 10000 replicates; seed 2026; primary block 96 "
            "windows (1 day) with 48/192 sensitivity"
        ),
        "comparisons": backbone_comparisons(
            horizon=horizon, seeds=seeds, final_dir=final_dir, datasets=datasets
        )
        + external_comparisons(
            horizon=horizon, final_dir=final_dir, datasets=datasets
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bootstrap_paired_tests.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_primary_intervals(payload, figures)
    significant = [
        {
            "dataset": record["dataset"],
            "comparison": f"psrc-{record['model_b']}",
            "loss": record["loss"],
            "seed": record["seed_a"],
            **next(
                block
                for block in record["blocks"]
                if block["block_windows"] == PRIMARY_BLOCK
            ),
        }
        for record in payload["comparisons"]
    ]
    return {"paired_tests": significant}
