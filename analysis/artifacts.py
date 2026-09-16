"""Read prediction dumps and archived result metadata for mechanism analyses.

Every analysis consumes the same ``predictions.npz`` artifacts produced by
the single formal test evaluation, so plotted numbers and report tables
share one authoritative pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_DIR = ROOT / "outputs" / "final"
DEFAULT_ANALYSIS_DIR = ROOT / "outputs" / "analysis"
DATASETS = (
    "dkasc_site31",
    "dkasc_site9a",
    "dkasc_site1a",
    "dkasc_site19",
    "pvod_station00",
    "pvod_station01",
    "pvod_station02",
    "pvod_station03",
    "hkust",
)
PSRC_SEEDS = (2026, 2027, 2028)
SINGLE_SEED = 2026

DUMP_FIELDS = ("y", "yhat0", "yhat_psrc", "delta", "gate", "window_start")


@dataclass(frozen=True)
class PredictionDump:
    dataset: str
    model: str
    seed: int
    horizon: int
    y: np.ndarray
    yhat0: np.ndarray
    yhat: np.ndarray
    delta: np.ndarray
    gate: np.ndarray
    window_start: np.ndarray
    capacity: float
    artifact_path: Path

    @property
    def hours_ahead(self) -> np.ndarray:
        return (np.arange(self.horizon) + 1) * 0.25

    @property
    def timestamps(self) -> np.ndarray:
        return self.window_start.astype("datetime64[ns]")


def job_directory(
    dataset: str,
    model: str,
    seed: int,
    horizon: int = 24,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> Path:
    return Path(final_dir) / f"{dataset}_L96_H{horizon}_{model}_seed{seed}"


def load_result(
    dataset: str,
    model: str,
    seed: int,
    horizon: int = 24,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> dict[str, object]:
    path = job_directory(dataset, model, seed, horizon, final_dir) / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_dump(
    dataset: str,
    model: str,
    seed: int,
    horizon: int = 24,
    final_dir: Path = DEFAULT_FINAL_DIR,
) -> PredictionDump:
    directory = job_directory(dataset, model, seed, horizon, final_dir)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    artifact_name = result.get("predictions_artifact")
    if not artifact_name:
        raise FileNotFoundError(f"{dataset}/{model}/seed{seed} has no dump record")
    path = directory / str(artifact_name)
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction dump: {path}")
    archive = np.load(path)
    missing = [field for field in DUMP_FIELDS if field not in archive.files]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    y = archive["y"]
    if y.shape[1] != horizon:
        raise ValueError(f"{path} horizon mismatch: {y.shape[1]} != {horizon}")
    return PredictionDump(
        dataset=dataset,
        model=model,
        seed=seed,
        horizon=horizon,
        y=np.asarray(archive["y"], dtype=np.float64),
        yhat0=np.asarray(archive["yhat0"], dtype=np.float64),
        yhat=np.asarray(archive["yhat_psrc"], dtype=np.float64),
        delta=np.asarray(archive["delta"], dtype=np.float64),
        gate=np.asarray(archive["gate"], dtype=np.float64),
        window_start=np.asarray(archive["window_start"], dtype=np.int64),
        capacity=float(result["capacity"]),
        artifact_path=path,
    )


def assert_aligned(*dumps: PredictionDump) -> None:
    """All models in one paired comparison must share test window identity."""
    if not dumps:
        raise ValueError("at least one dump required")
    reference = dumps[0].window_start
    for dump in dumps[1:]:
        if dump.window_start.shape != reference.shape:
            raise ValueError(
                f"window count mismatch: {dump.model} vs {dumps[0].model}"
            )
        if not np.array_equal(dump.window_start, reference):
            raise ValueError(f"windows not aligned: {dump.model} vs {dumps[0].model}")
        if dump.horizon != dumps[0].horizon:
            raise ValueError("horizon mismatch among paired dumps")
