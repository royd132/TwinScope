"""Average-rank table and Friedman tests (descriptive, supplementary).

The primary inferential evidence is the paired block bootstrap on forecast
loss sequences (``significance.py``).  Here blocks are dataset x metric
summary scores, following the TFCNet-style rank protocol; with only four
stations the test has low power and the result is reported descriptively.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parents[1]
FORMAL_REPORT = ROOT / "outputs" / "final" / "formal_report.json"

DATASET_NAME_MAP = {
    "DKASC_SITE31": "dkasc_site31",
    "DKASC_SITE9A": "dkasc_site9a",
    "DKASC_SITE1A": "dkasc_site1a",
    "DKASC_SITE19": "dkasc_site19",
    "PVOD_STATION00": "pvod_station00",
    "PVOD_STATION01": "pvod_station01",
    "PVOD_STATION02": "pvod_station02",
    "PVOD_STATION03": "pvod_station03",
    "HKUST": "hkust",
}
BLOCK_METRICS = ("rmse", "mae", "r2")
HIGHER_IS_BETTER = {"rmse": False, "mae": False, "r2": True}


def load_score_frame(
    report_path: Path = FORMAL_REPORT,
) -> tuple[pd.DataFrame, list[str]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    models = list(report["models"])
    records = []
    for row in report["rows"]:
        dataset = DATASET_NAME_MAP[row["dataset"]]
        for model in models:
            entry = row["values"][model]
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **{
                        metric: float(entry[metric]["mean"])
                        for metric in ("rmse", "mae", "mbe", "r2")
                    },
                }
            )
    return pd.DataFrame(records), models


def rank_matrix(score_frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    block_rows = []
    for dataset in sorted(score_frame["dataset"].unique()):
        for metric in BLOCK_METRICS:
            part = score_frame[score_frame["dataset"] == dataset].set_index("model")
            scores = part.loc[models, metric]
            ascending = not HIGHER_IS_BETTER[metric]
            ranks = scores.rank(ascending=ascending, method="average")
            for model, rank in ranks.items():
                block_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model": model,
                        "score": float(scores.loc[model]),
                        "rank": float(rank),
                    }
                )
    return pd.DataFrame(block_rows)


def friedman_for_metric(block_table: pd.DataFrame, metric: str) -> dict[str, object]:
    part = block_table[block_table["metric"] == metric]
    datasets = sorted(part["dataset"].unique())
    models = sorted(part["model"].unique())
    matrix = (
        part.pivot(index="dataset", columns="model", values="rank")
        .reindex(index=datasets, columns=models)
    )
    # friedmanchisquare wants one array per block (dataset).
    statistic, p_value = friedmanchisquare(
        *[matrix.loc[dataset].to_numpy() for dataset in datasets]
    )
    return {
        "metric": metric,
        "n_blocks": int(len(datasets)),
        "n_models": int(len(models)),
        "friedman_statistic": float(statistic),
        "p_value": float(p_value),
        "note": (
            f"descriptive only: {len(datasets)} station blocks, low power"
        ),
    }


def run(
    output_dir: Path, report_path: Path | None = None
) -> dict[str, object]:
    # Resolve the default at call time so tests can monkeypatch FORMAL_REPORT.
    score_frame, models = load_score_frame(
        FORMAL_REPORT if report_path is None else Path(report_path)
    )
    block_table = rank_matrix(score_frame, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_table.to_csv(output_dir / "rank_blocks.csv", index=False)

    average = (
        block_table.groupby("model")["rank"]
        .mean()
        .sort_values()
        .reset_index(name="average_rank")
    )
    average["average_rank"] = average["average_rank"].round(4)
    average.to_csv(output_dir / "average_rank.csv", index=False)
    per_metric = (
        block_table.groupby(["metric", "model"])["rank"]
        .mean()
        .reset_index(name="average_rank")
    )
    per_metric.to_csv(output_dir / "average_rank_by_metric.csv", index=False)

    tests = [friedman_for_metric(block_table, metric) for metric in BLOCK_METRICS]
    all_block_tests = {
        "metric": "rmse+mae+r2 blocks (TFCNet-style)",
        "n_blocks": int(len(block_table) / len(models)),
        "n_models": int(len(models)),
    }
    # Combine all 12 blocks for one supplementary Friedman test.
    pivot = block_table.pivot_table(
        index=["dataset", "metric"], columns="model", values="rank"
    )
    arrays = [pivot.iloc[i].to_numpy() for i in range(len(pivot))]
    statistic, p_value = friedmanchisquare(*arrays)
    all_block_tests.update(
        {"friedman_statistic": float(statistic), "p_value": float(p_value)}
    )
    payload = {
        "protocol": (
            "ranks per station x metric block from formal_report.json "
            "multi-seed means; Friedman is supplementary (few blocks), "
            "paired window-level tests live in bootstrap_paired_tests.json"
        ),
        "per_metric_tests": tests,
        "all_blocks_test": all_block_tests,
        "average_rank": average.to_dict(orient="records"),
    }
    (output_dir / "ranking_friedman.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    psrc_rank = average.loc[average["model"] == "psrc", "average_rank"]
    return {
        "average_rank_table": average.to_dict(orient="records"),
        "psrc_average_rank": float(psrc_rank.iloc[0]),
        "friedman": tests + [all_block_tests],
    }
