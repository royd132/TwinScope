"""Tests for the prediction-dump-based mechanism analyses."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import case_study, gate, horizon, ranking, significance
from analysis.artifacts import PredictionDump, assert_aligned


def make_fake_dump(dataset="dkasc_site31", model="psrc", seed=2026, n=200):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 1.0, size=(n, 24))
    base_error = rng.normal(0.0, 0.2, size=(n, 24))
    yhat0 = y + base_error
    # Gate tracks backbone error magnitude; PSRC removes most of the error.
    gate_values = np.clip(np.abs(base_error) * 3.0, 0.0, 1.0)
    yhat = y + base_error * (1.0 - gate_values) * 0.3
    window_start = (
        np.datetime64("2020-01-01T08:00:00", "ns")
        + np.arange(n) * np.timedelta64(15, "m")
    ).astype("datetime64[ns]").astype("int64")
    return PredictionDump(
        dataset=dataset,
        model=model,
        seed=seed,
        horizon=24,
        y=y,
        yhat0=yhat0,
        yhat=yhat,
        delta=yhat - yhat0,
        gate=gate_values,
        window_start=window_start,
        capacity=1.0,
        artifact_path=None,
    )


def test_horizon_errors_known_values():
    y = np.zeros((4, 2))
    yhat = np.full((4, 2), 0.5)
    metrics = horizon.horizon_errors(y, yhat)
    assert metrics["rmse"].shape == (2,)
    np.testing.assert_allclose(metrics["rmse"], [0.5, 0.5])
    np.testing.assert_allclose(metrics["mae"], [0.5, 0.5])
    np.testing.assert_allclose(metrics["mbe"], [0.5, 0.5])


def test_gain_sign_and_series(monkeypatch, tmp_path):
    dump = make_fake_dump()

    def fake_load(dataset, model, seed, horizon=24, final_dir=None):
        return make_fake_dump(dataset=dataset, model=model, seed=seed)

    monkeypatch.setattr("analysis.horizon.load_dump", fake_load)
    table = horizon.horizon_table_for_station("dkasc_site31")
    base = table[(table["model"] == "backbone") & (table["seed"] == 2026)]
    psrc = table[(table["model"] == "psrc") & (table["seed"] == 2026)]
    base_rmse = base.sort_values("horizon")["rmse"].to_numpy()
    psrc_rmse = psrc.sort_values("horizon")["rmse"].to_numpy()
    # Corrected predictions must improve over the backbone on this construct.
    assert np.all(psrc_rmse < base_rmse)
    gain = (base_rmse - psrc_rmse) / base_rmse * 100.0
    assert np.all(gain > 0.0)
    gate_rows = psrc.sort_values("horizon")["gate_mean"].to_numpy()
    assert np.all((gate_rows >= 0.0) & (gate_rows <= 1.0))


def test_block_bootstrap_clear_improvement():
    n = 240
    rng = np.random.default_rng(0)
    loss_b = rng.uniform(0.5, 1.5, size=n)
    loss_a = loss_b - 0.3 + rng.normal(0.0, 0.01, size=n)
    result = significance.block_bootstrap_reduction(
        loss_a, loss_b, block=48, replicates=1000, seed=2026
    )
    assert result["ci95_low_pct"] > 0.0
    assert result["observed_reduction_pct"] > 20.0
    assert result["p_value_two_sided"] < 0.05


def test_block_bootstrap_zero_differential_crosses_zero():
    n = 240
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 1.5, size=n)
    differential = rng.choice([-0.1, 0.1], size=n)
    loss_b = base
    loss_a = base + differential
    result = significance.block_bootstrap_reduction(
        loss_a, loss_b, block=48, replicates=1000, seed=2026
    )
    low, high = result["ci95_low_pct"], result["ci95_high_pct"]
    assert low < 1e-9 and high > -1e-9 and low < high


def test_block_bootstrap_validates_block():
    with pytest.raises(ValueError):
        significance.block_bootstrap_reduction(
            np.ones(10), np.ones(10), block=0, replicates=10
        )
    with pytest.raises(ValueError):
        significance.block_bootstrap_reduction(
            np.ones(10), np.ones(10), block=11, replicates=10
        )


def test_aligned_dump_assertions():
    dump_a = make_fake_dump(n=50)
    dump_b = make_fake_dump(n=50)
    assert_aligned(dump_a, dump_b)
    dump_c = make_fake_dump(n=49)
    with pytest.raises(ValueError):
        assert_aligned(dump_a, dump_c)
    window_start = dump_b.window_start.copy()
    window_start[0] += 1
    dump_d = SimpleNamespace(
        window_start=window_start, horizon=24, model="other"
    )
    with pytest.raises(ValueError):
        assert_aligned(dump_a, dump_d)


def test_gate_diagnosis_relationships(monkeypatch):
    monkeypatch.setattr(
        "analysis.artifacts.load_dump",
        lambda dataset, model, seed, horizon=24, final_dir=None: make_fake_dump(
            dataset=dataset, seed=seed
        ),
    )
    diagnosis = gate.station_diagnosis("pvod_station01")
    summary = diagnosis["summary"]
    # Constructed gate is monotone in backbone |error|.
    assert summary["pooled"]["spearman_gate_vs_base_abs_err"] > 0.9
    assert summary["pooled"]["spearman_gate_vs_psrc_abs_err"] < (
        summary["pooled"]["spearman_gate_vs_base_abs_err"]
    )
    assert 0.0 <= summary["improved_share_all"] <= 1.0
    binned = diagnosis["binned"]
    assert len(binned) == 10
    assert binned["gate_mean"].is_monotonic_increasing
    per_horizon = diagnosis["per_horizon"]
    assert list(per_horizon["horizon"]) == list(range(1, 25))


def test_case_selection_is_deterministic_and_nonoverlapping(monkeypatch):
    monkeypatch.setattr(
        case_study,
        "load_dump",
        lambda dataset, model, seed, horizon=24, final_dir=None: make_fake_dump(
            dataset=dataset, seed=seed
        ),
    )
    first = case_study.select_cases("dkasc_site31")
    second = case_study.select_cases("dkasc_site31")
    pd.testing.assert_frame_equal(first, second)
    windows = sorted(first["window"].tolist())
    for left, right in zip(windows[:-1], windows[1:]):
        assert right - left >= case_study.MIN_SEPARATION_STEPS
    assert set(first["reason"]) <= {
        "high backbone error",
        "steepest observed power ramp",
        "median-error daytime",
    }


def test_greedy_pick_enforces_separation():
    frame = pd.DataFrame(
        {
            "window": [0, 5, 100, 150, 200],
            "score": [10, 9.5, 9, 8, 7],
        }
    )
    picked = case_study._greedy_pick(frame, "score", 3)
    assert picked == [0, 100, 200]
    for left, right in zip(picked[:-1], picked[1:]):
        assert right - left >= case_study.MIN_SEPARATION_STEPS


def _minimal_report(path: Path):
    datasets = [
        "DKASC_SITE31",
        "DKASC_SITE9A",
        "PVOD_STATION00",
        "PVOD_STATION01",
    ]
    models = ["psrc", "good", "weak"]
    rows = []
    for block, dataset in enumerate(datasets):
        values = {}
        for model in models:
            base = {"psrc": 1.0, "good": 1.2, "weak": 1.5}[model]
            score = base + 0.01 * block
            # Last block swaps the middle and weak models so ranks vary
            # across blocks (Friedman needs between-block variation).
            if block == len(datasets) - 1 and model == "good":
                score += 0.5
            values[model] = {
                "rmse": {"mean": score, "std": 0.0},
                "mae": {"mean": score, "std": 0.0},
                "mbe": {"mean": 0.0, "std": 0.0},
                "r2": {"mean": 10.0 - score, "std": 0.0},
            }
        rows.append({"dataset": dataset, "values": values})
    report = {"models": models, "rows": rows}
    path.write_text(json.dumps(report), encoding="utf-8")


def test_ranking_and_friedman(monkeypatch, tmp_path):
    report_path = tmp_path / "formal_report.json"
    _minimal_report(report_path)
    monkeypatch.setattr(ranking, "FORMAL_REPORT", report_path)
    result = ranking.run(tmp_path)
    ranks = {row["model"]: row["average_rank"] for row in result["average_rank_table"]}
    assert ranks["psrc"] == 1.0
    assert ranks["good"] == 2.25
    assert ranks["weak"] == 2.75
    rmse_test = next(
        test for test in result["friedman"] if isinstance(test, dict) and test.get("metric") == "rmse"
    )
    assert rmse_test["n_blocks"] == 4
    assert 0.0 <= rmse_test["p_value"] <= 1.0
    saved = json.loads((tmp_path / "ranking_friedman.json").read_text())
    assert saved["average_rank"][0]["model"] == "psrc"
