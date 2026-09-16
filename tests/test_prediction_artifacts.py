"""Prediction dumps and checkpoints accompany the single headline evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import (
    WARMUP_EPOCHS,
    _evaluate,
    _loader,
    _make_scheduler,
    run_job,
    save_checkpoint,
    save_prediction_artifacts,
)
from formal.runner import should_skip

ROOT = Path(__file__).resolve().parents[1]
SEQ_LEN = 96
HORIZON = 24

DUMP_KEYS = ("y", "yhat0", "yhat_psrc", "delta", "gate", "window_start")


def _write_series(path, rows=1000):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="15min"),
            "TargetObserved": np.ones(rows),
            "Target": np.linspace(0.0, 10.0, rows, dtype=np.float32),
        }
    )
    frame.to_csv(path, index=False)


class _FakePSRC(nn.Module):
    """Mimics the PSRC inference attributes without the full architecture."""

    def __init__(self, horizon):
        super().__init__()
        self.proj = nn.Linear(1, 1)
        self.horizon = horizon

    def forward(self, history, cycle, future_solar, *args):
        batch = history.shape[0]
        base = torch.full((batch, self.horizon), 0.5)
        gate = torch.full((batch, self.horizon), 0.37)
        corrected = base + 0.1
        self.last_psrc_gate = gate
        self.last_semantic_decomposition = (base, corrected - base)
        return corrected.unsqueeze(-1)


def test_deterministic_run_persists_dump_without_checkpoint(tmp_path):
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    job = JobSpec("pvod_station00", "persistence", SEQ_LEN, HORIZON, 2026)
    artifacts = tmp_path / "pvod_persistence"

    result = run_job(job, config, artifacts_dir=artifacts)

    assert result["predictions_artifact"] == "predictions.npz"
    assert result["checkpoint_artifact"] is None
    dump = np.load(artifacts / "predictions.npz")
    for key in DUMP_KEYS:
        assert key in dump.files
    n_windows = dump["y"].shape[0]
    assert dump["y"].shape == (n_windows, HORIZON)
    assert dump["window_start"].shape == (n_windows,)
    np.testing.assert_array_equal(dump["yhat0"], dump["yhat_psrc"])
    np.testing.assert_array_equal(dump["delta"], 0.0)
    assert np.isnan(dump["gate"]).all()
    np.testing.assert_allclose(
        np.sqrt(np.mean((dump["yhat_psrc"] - dump["y"]) ** 2)),
        result["test"]["rmse_physical"],
        rtol=1e-6,
    )
    assert not (artifacts / "checkpoint.pt").exists()


def test_run_without_artifacts_dir_has_no_artifact_fields():
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    job = JobSpec("pvod_station00", "persistence", SEQ_LEN, HORIZON, 2026)

    result = run_job(job, config)

    assert result["predictions_artifact"] is None
    assert result["checkpoint_artifact"] is None


def test_psrc_collect_keeps_base_gate_and_reproduces_metrics(tmp_path):
    path = tmp_path / "series.csv"
    _write_series(path)
    bundle = load_dataset(path, seq_len=SEQ_LEN, horizon=HORIZON)
    model = _FakePSRC(HORIZON)
    loader = _loader(bundle, bundle.test.starts, batch_size=64, shuffle=False, seed=7)

    metrics, arrays = _evaluate(model, loader, bundle, torch.device("cpu"), collect=True)
    dump_path = save_prediction_artifacts(
        tmp_path, arrays, loader, SEQ_LEN, bundle, metrics
    )
    checkpoint_path = save_checkpoint(tmp_path, model)

    dump = np.load(dump_path)
    assert np.isfinite(dump["gate"]).all()
    np.testing.assert_allclose(dump["gate"], 0.37)
    np.testing.assert_allclose(dump["yhat_psrc"] - dump["yhat0"], dump["delta"])
    assert dump["window_start"].shape == (bundle.test.starts.shape[0],)
    # Timestamps must line up with the deterministic test window order.
    target_starts = bundle.test.starts + SEQ_LEN
    expected = bundle.timestamps[target_starts].astype("datetime64[ns]").astype(np.int64)
    np.testing.assert_array_equal(dump["window_start"], expected)
    loaded = torch.load(checkpoint_path, weights_only=True)
    assert set(loaded) == {"state_dict"}


def test_warmup_scheduler_ramps_then_decays():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    scheduler = _make_scheduler(
        {"scheduler": "cosine_warmup"}, optimizer, epochs
    )
    rates = []
    for _ in range(epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    warmup = rates[: WARMUP_EPOCHS]
    assert warmup[0] < warmup[-1]
    assert np.all(np.diff(np.round(warmup, 12)) >= 0)
    assert rates[WARMUP_EPOCHS] >= rates[-1]
    assert rates[-1] < rates[WARMUP_EPOCHS]


def _write_result(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_result(**overrides):
    result = {
        "status": "complete",
        "config_fingerprint": "fp",
        "headline_test_evaluation_passes": 1,
        "test": {"rmse_physical": 1.0},
        "predictions_artifact": None,
        "checkpoint_artifact": None,
    }
    result.update(overrides)
    return result


def test_resume_requires_artifacts_only_when_requested(tmp_path):
    result_path = tmp_path / "job" / "result.json"
    _write_result(result_path, _complete_result())

    assert should_skip(result_path, "fp") is True
    assert should_skip(result_path, "fp", require_artifacts=True) is False

    (result_path.parent / "predictions.npz").write_bytes(b"x")
    # Deterministic-style record (no checkpoint) is complete with the dump.
    _write_result(
        result_path, _complete_result(predictions_artifact="predictions.npz")
    )
    assert should_skip(result_path, "fp", require_artifacts=True) is True

    # A trained-model record must also carry its checkpoint.
    _write_result(
        result_path,
        _complete_result(
            predictions_artifact="predictions.npz",
            checkpoint_artifact="checkpoint.pt",
        ),
    )
    assert should_skip(result_path, "fp", require_artifacts=True) is False
    (result_path.parent / "checkpoint.pt").write_bytes(b"x")
    assert should_skip(result_path, "fp", require_artifacts=True) is True
