"""Validated configuration for the formal H16 comparison and H24 archive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_NAMES = (
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
MODEL_NAMES = (
    "psrc",
    "timemixer",
    "ampdnet",
    "crossunet",
    "timesnet",
    "patchtst",
    "itransformer",
    "dlinear",
    "cyclenet",
    "patchmlp",
    "tcn",
    "frets",
    "lstm",
    "gru",
    "timexer",
    "persistence",
    "seasonal_naive",
    "smart_persistence",
)
# Parameter-free reference baselines are evaluated once per seed but never
# trained; every other model follows the locked 100-epoch protocol.
DETERMINISTIC_MODELS = ("persistence", "seasonal_naive", "smart_persistence")
# Public-configuration baselines copied from the reference repository are run
# once at the anchor seed only, exactly as published (no per-seed retuning).
SINGLE_SEED_MODELS = ("tcn", "frets", "lstm", "gru", "timexer")
# Models whose architecture hyperparameters come from the reference
# repository's published configuration instead of the shared baseline block.
# PatchTST keeps the full three-seed matrix; the five imported baselines run
# once at the anchor seed only.
PUBLIC_BASELINE_MODELS = ("patchtst",) + SINGLE_SEED_MODELS
SEEDS = (2026, 2027, 2028)
ALLOWED_HORIZONS = (16, 24)
ALLOWED_SEEDS = SEEDS


def seeds_for_model(model: str) -> tuple[int, ...]:
    if model in SINGLE_SEED_MODELS:
        return (SEEDS[0],)
    return SEEDS


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    model: str
    seq_len: int
    horizon: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset}_L{self.seq_len}_H{self.horizon}_{self.model}_seed{self.seed}"


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    datasets: tuple[DatasetSpec, ...]
    models: tuple[str, ...]
    seq_len: int
    horizons: tuple[int, ...]
    seeds: tuple[int, ...]
    epochs: int
    patience: int
    train_fraction: float
    validation_end_fraction: float
    output_projection: str
    train_windows_per_epoch: int
    psrc: dict[str, object]
    baseline: dict[str, object]
    public_baselines: dict[str, dict[str, object]]
    best_parameters: Path

    def jobs(self) -> list[JobSpec]:
        return [
            JobSpec(dataset.name, model, self.seq_len, horizon, seed)
            for dataset in self.datasets
            for horizon in self.horizons
            for model in self.models
            for seed in self.seeds_for(model)
        ]

    def dataset(self, name: str) -> DatasetSpec:
        return next(dataset for dataset in self.datasets if dataset.name == name)

    def seeds_for(self, model: str) -> tuple[int, ...]:
        """Seeds a model is evaluated under.

        Public-configuration baselines always run once at the anchor seed
        (2026); every tuned/protocol model is repeated across the seeds
        selected for this experiment run.
        """
        allowed = seeds_for_model(model)
        if model in SINGLE_SEED_MODELS:
            return allowed
        return tuple(seed for seed in self.seeds if seed in allowed)


def _require_exact(actual, expected, label: str) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{label} must be {list(expected)}, got {list(actual)}")


def _require_subset(actual, allowed, label: str) -> None:
    actual_t = tuple(actual)
    allowed_t = tuple(allowed)
    if not actual_t:
        raise ValueError(f"{label} must be a non-empty subset of {list(allowed_t)}")
    unexpected = [value for value in actual_t if value not in allowed_t]
    if unexpected:
        raise ValueError(
            f"{label} contains {unexpected}; allowed values are {list(allowed_t)}"
        )
    if len(set(actual_t)) != len(actual_t):
        raise ValueError(f"{label} contains duplicate entries: {list(actual_t)}")


def load_best_parameters(
    path: Path, required_datasets: tuple[str, ...] = DATASET_NAMES
) -> dict[str, object]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    for dataset, record in records.items():
        if not isinstance(record.get("source"), str) or not isinstance(
            record.get("params"), dict
        ):
            raise ValueError(f"invalid best parameter record: {dataset}")
    missing = [dataset for dataset in required_datasets if dataset not in records]
    if missing:
        raise ValueError(f"missing best parameter records for: {missing}")
    return records


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    datasets = tuple(
        DatasetSpec(item["name"], (root / item["path"]).resolve())
        for item in payload["datasets"]
    )
    models = tuple(payload["models"])
    horizons = tuple(int(value) for value in payload["horizons"])
    seeds = tuple(int(value) for value in payload["seeds"])
    # The historical H24 config pins the full four-station matrix exactly;
    # newer campaigns may select allow-listed subsets/add the fifth station.
    _require_subset((item.name for item in datasets), DATASET_NAMES, "datasets")
    _require_exact(models, MODEL_NAMES, "models")
    _require_subset(horizons, ALLOWED_HORIZONS, "horizons")
    _require_subset(seeds, ALLOWED_SEEDS, "seeds")
    if int(payload["seq_len"]) != 96:
        raise ValueError("seq_len must be 96")
    if int(payload["epochs"]) != 100 or int(payload["patience"]) != 10:
        raise ValueError("formal budget must be epochs=100 and patience=10")
    train = float(payload["train_fraction"])
    validation = float(payload["validation_end_fraction"])
    if not 0.0 < train < validation < 1.0:
        raise ValueError("split fractions must satisfy 0 < train < validation < 1")
    public_baselines = payload.get("public_baselines", {})
    _require_exact(
        public_baselines, PUBLIC_BASELINE_MODELS, "public baseline models"
    )
    for model, record in public_baselines.items():
        if not isinstance(record, dict) or not record:
            raise ValueError(f"public baseline config must be a non-empty mapping: {model}")
    config = ExperimentConfig(
        output_dir=(root / payload["output_dir"]).resolve(),
        datasets=datasets,
        models=models,
        seq_len=96,
        horizons=horizons,
        seeds=seeds,
        epochs=100,
        patience=10,
        train_fraction=train,
        validation_end_fraction=validation,
        output_projection=str(payload["output_projection"]),
        train_windows_per_epoch=int(payload["train_windows_per_epoch"]),
        psrc=dict(payload["psrc"]),
        baseline=dict(payload["baseline"]),
        public_baselines={
            model: dict(record) for model, record in public_baselines.items()
        },
        best_parameters=(root / payload["best_parameters"]).resolve(),
    )
    load_best_parameters(
        config.best_parameters,
        required_datasets=tuple(dataset.name for dataset in datasets),
    )
    return config
