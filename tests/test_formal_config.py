import json
from dataclasses import replace
from pathlib import Path

from formal.config import (
    PUBLIC_BASELINE_MODELS,
    SINGLE_SEED_MODELS,
    load_experiment_config,
    seeds_for_model,
)


CONFIG_PATH = "configs/formal_experiment.json"
EXPECTED_MODELS = {
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
}
EXPECTED_SEEDS = {2026, 2027, 2028}


def test_checked_in_config_is_the_locked_h16_matrix():
    config = load_experiment_config(CONFIG_PATH)

    assert [dataset.name for dataset in config.datasets] == [
        "dkasc_site31", "dkasc_site9a", "dkasc_site1a", "dkasc_site19",
        "pvod_station00", "pvod_station01", "pvod_station02", "pvod_station03",
        "hkust",
    ]
    assert set(config.models) == EXPECTED_MODELS
    assert config.seeds == (2026,)
    assert config.horizons == (16,)
    assert config.output_dir.name == "final_h16"
    assert config.seq_len == 96
    assert config.epochs == 100
    assert config.patience == 10
    assert config.train_windows_per_epoch == 6000


def test_checked_in_config_documents_a_single_audited_setting():
    raw = json.loads(open(CONFIG_PATH, encoding="utf-8").read())

    assert raw["psrc"]["gate"] is True
    # Every training baseline shares one fixed hyperparameter block; PSRC is
    # the only model allowed station-specific tuned parameters. Published
    # architecture overrides live in the public_baselines block so that the
    # shared fingerprint stays untouched for the remaining baselines.
    assert raw["baseline"] == {
        "d_model": 64,
        "batch": 128,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adam",
        "scheduler": "none",
        "loss_base": "mse",
    }
    # Public-configuration models are pinned exactly to PatchTST plus the five
    # imported baselines, keeping the published architecture values.
    assert set(raw["public_baselines"]) == set(PUBLIC_BASELINE_MODELS)
    assert raw["public_baselines"]["patchtst"] == {
        "patchtst_d_model": 128,
        "patchtst_d_ff": 256,
        "patchtst_heads": 16,
        "patchtst_layers": 3,
        "patchtst_dropout": 0.2,
        "patchtst_patch_len": 16,
        "patchtst_stride": 8,
    }
    # The reference repository trains every imported baseline with a plain
    # Adam(lr=1e-3) optimizer and no weight decay; weight_decay=0 also avoids
    # a weight_norm/Adam numerical blow-up observed for TCN.
    assert raw["public_baselines"]["lstm"] == {
        "lstm_d_model": 128,
        "lstm_layers": 1,
        "weight_decay": 0.0,
    }
    assert raw["public_baselines"]["tcn"] == {
        "tcn_channels": 128,
        "tcn_layers": 3,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.1,
        "weight_decay": 0.0,
    }


def test_job_plan_covers_the_full_model_dataset_seed_matrix():
    config = load_experiment_config(CONFIG_PATH)
    jobs = config.jobs()

    assert len(jobs) == 9 * len(EXPECTED_MODELS)
    assert {job.dataset for job in jobs} == {
        dataset.name for dataset in config.datasets
    }
    assert {job.model for job in jobs} == EXPECTED_MODELS
    assert {job.seed for job in jobs} == {2026}
    assert {(job.seq_len, job.horizon) for job in jobs} == {(96, 16)}


def test_public_baselines_run_only_at_the_anchor_seed():
    config = load_experiment_config(CONFIG_PATH)

    assert seeds_for_model("tcn") == (2026,)
    assert seeds_for_model("psrc") == (2026, 2027, 2028)
    jobs = config.jobs()
    for model in SINGLE_SEED_MODELS:
        assert {job.seed for job in jobs if job.model == model} == {2026}


def test_config_filters_models_and_seeds_from_the_plan():
    config = load_experiment_config(CONFIG_PATH)
    config = replace(
        config, models=("psrc", "dlinear"), seeds=(2026,)
    )

    jobs = config.jobs()

    assert {job.model for job in jobs} == {"psrc", "dlinear"}
    assert {job.seed for job in jobs} == {2026}
    assert len(jobs) == 9 * 2


def test_public_baseline_ignores_non_anchor_seed_overrides_in_the_plan():
    config = load_experiment_config(CONFIG_PATH)
    config = replace(config, models=("tcn",), seeds=(2027, 2028))

    jobs = config.jobs()

    assert {job.model for job in jobs} == {"tcn"}
    assert {job.seed for job in jobs} == {2026}
    assert len(jobs) == 9


H24_CONFIG_PATH = "configs/formal_experiment_h24.json"


def test_h24_archive_preserves_the_previous_campaign():
    config = load_experiment_config(H24_CONFIG_PATH)

    assert [dataset.name for dataset in config.datasets] == [
        "dkasc_site31", "dkasc_site9a", "pvod_station00", "pvod_station01"
    ]
    assert set(config.models) == EXPECTED_MODELS
    assert config.seeds == (2026, 2027, 2028)
    assert config.horizons == (24,)
    assert config.output_dir.name == "final"
    jobs = config.jobs()
    assert len(jobs) == 4 * 13 * 3 + 4 * len(SINGLE_SEED_MODELS)
    assert {job.seed for job in jobs} == EXPECTED_SEEDS
    assert {(job.seq_len, job.horizon) for job in jobs} == {(96, 24)}


def test_loader_rejects_unknown_horizon_and_seed_subsets(tmp_path):
    import pytest

    raw = json.loads(open(CONFIG_PATH, encoding="utf-8").read())
    raw["best_parameters"] = str(Path("configs/best_params.json").resolve())
    raw["horizons"] = [12]
    bad = tmp_path / "bad_horizon.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(bad)

    raw["horizons"] = [16]
    raw["seeds"] = [42]
    bad_seed = tmp_path / "bad_seed.json"
    bad_seed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(bad_seed)

    # The allow-lists are the only supported campaign dimensions.
    from formal.config import ALLOWED_HORIZONS, ALLOWED_SEEDS

    assert set(ALLOWED_HORIZONS) == {16, 24}
    assert set(ALLOWED_SEEDS) == EXPECTED_SEEDS
