import json
from pathlib import Path

from formal.config import load_experiment_config
from formal.runner import config_fingerprint, finalize_report, main, should_skip

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_plans_the_h16_matrix_without_training(
    tmp_path, monkeypatch
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run constructed or trained a model")

    monkeypatch.setattr("formal.runner.run_job", fail_if_called)

    summary = main(["--dry-run", "--output", str(tmp_path)])

    assert summary.planned == 9 * 18
    assert summary.executed == 0
    assert summary.skipped == 0


def test_dataset_filter_keeps_all_models_seeds_and_horizons(tmp_path):
    summary = main(
        [
            "--dry-run",
            "--datasets",
            "dkasc_site31",
            "pvod_station00",
            "--output",
            str(tmp_path),
        ]
    )

    assert summary.planned == 2 * 18
    assert summary.executed == 0


def test_resume_only_skips_matching_complete_result(tmp_path):
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    job = config.jobs()[0]
    fingerprint = config_fingerprint(job, config)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"status": "complete", "config_fingerprint": "old"}),
        encoding="utf-8",
    )
    assert should_skip(result_path, fingerprint) is False

    result_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "config_fingerprint": fingerprint,
                "headline_test_evaluation_passes": 1,
            }
        ),
        encoding="utf-8",
    )
    assert should_skip(result_path, fingerprint) is True


def test_fingerprint_changes_with_dataset():
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    first, second = config.jobs()[0], config.jobs()[4]

    assert config_fingerprint(first, config) != config_fingerprint(second, config)


def test_fingerprint_changes_when_dataset_changes(tmp_path):
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")
    job = config.jobs()[0]
    before = config_fingerprint(job, config)
    path = config.dataset(job.dataset).path
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        assert config_fingerprint(job, config) != before
    finally:
        path.write_bytes(original)


def test_incomplete_matrix_does_not_create_report(tmp_path, monkeypatch):
    config = load_experiment_config(ROOT / "configs" / "formal_experiment.json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("partial results must not create a workbook")

    monkeypatch.setattr("formal.runner.build_workbook", fail_if_called)

    assert finalize_report(tmp_path, config) is None
    assert not (tmp_path / "formal_comparison.xlsx").exists()


def test_fm_h16_preset_plans_the_locked_four_station_matrix(tmp_path):
    summary = main(
        ["--preset", "fm-h16", "--dry-run", "--output", str(tmp_path)]
    )

    assert summary.planned == 4 * 17
    assert summary.executed == 0


def test_fm_h16_preset_rejects_dimension_overrides(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="cannot be combined"):
        main(
            [
                "--preset", "fm-h16", "--datasets", "hkust",
                "--dry-run", "--output", str(tmp_path),
            ]
        )


def test_rebuild_fm_is_only_available_for_the_locked_preset(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="requires --preset"):
        main(["--rebuild-fm", "--dry-run", "--output", str(tmp_path)])
