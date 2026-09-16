"""Single command-line runner for the configured formal comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from formal.config import (
    DATASET_NAMES,
    MODEL_NAMES,
    SEEDS,
    ExperimentConfig,
    JobSpec,
    load_experiment_config,
)
from formal.engine import _parameter_record, run_job
from formal.fm_table import build_fm_comparison_workbook
from formal.report import build_workbook, write_report_payload

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "formal_experiment.json"
DRIFT_LIMIT = 0.005
FM_H16_DATASETS = ("dkasc_site31", "dkasc_site1a", "pvod_station02", "hkust")
FM_H16_MODELS = tuple(model for model in MODEL_NAMES if model != "smart_persistence")
FM_H16_REPORT = ROOT / "outputs" / "correction_tune_v1" / "test_confirmation_report.json"


@dataclass(frozen=True)
class ExecutionSummary:
    planned: int
    executed: int
    skipped: int


def config_fingerprint(job: JobSpec, config: ExperimentConfig) -> str:
    parameter_source, parameters = _parameter_record(job, config)
    payload = {
        "job": asdict(job),
        "dataset_sha256": hashlib.sha256(
            config.dataset(job.dataset).path.read_bytes()
        ).hexdigest(),
        "epochs": config.epochs,
        "patience": config.patience,
        "train_fraction": config.train_fraction,
        "validation_end_fraction": config.validation_end_fraction,
        "output_projection": config.output_projection,
        "train_windows_per_epoch": config.train_windows_per_epoch,
        "psrc": config.psrc,
        "parameter_source": parameter_source,
        "parameters": parameters,
    }
    if job.model == "smart_persistence":
        # Bumped when the deterministic clear-sky feature audit changes;
        # invalidates smart persistence results produced against a
        # mis-identified constant proxy column. Key is absent from every
        # other model's payload so their fingerprints stay unchanged.
        payload["solar_geometry_audit"] = "night_trajectory_v1"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifacts_complete(result_path: Path, result: dict[str, object]) -> bool:
    directory = result_path.parent
    prediction_name = result.get("predictions_artifact")
    if not prediction_name or not (directory / str(prediction_name)).is_file():
        return False
    # Deterministic baselines store no checkpoint; every trained model must.
    checkpoint_name = result.get("checkpoint_artifact")
    if checkpoint_name and not (directory / str(checkpoint_name)).is_file():
        return False
    return True


def should_skip(
    result_path: Path,
    fingerprint: str,
    *,
    require_artifacts: bool = False,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        result.get("status") == "complete"
        and result.get("config_fingerprint") == fingerprint
        and result.get("headline_test_evaluation_passes") == 1
    )
    if not valid:
        return False
    if require_artifacts and not _artifacts_complete(result_path, result):
        return False
    return True


def _backup_legacy_result(result_path: Path) -> None:
    """Preserve the pre-dump archive before the first overwrite of a task."""
    backup_path = result_path.with_name("result.predump_backup.json")
    if not backup_path.is_file():
        backup_path.write_text(
            result_path.read_text(encoding="utf-8"), encoding="utf-8"
        )


def _drift_record(
    job: JobSpec,
    old: dict[str, object],
    new: dict[str, object],
    fingerprint: str,
) -> dict[str, object]:
    old_fingerprint = old.get("config_fingerprint")
    record: dict[str, object] = {
        "key": job.key,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": fingerprint,
    }
    if old_fingerprint != fingerprint:
        record["status"] = "fingerprint_changed"
        return record
    old_rmse = float(old["test"]["rmse_physical"])
    new_rmse = float(new["test"]["rmse_physical"])
    relative = abs(new_rmse - old_rmse) / max(abs(old_rmse), 1e-12)
    record.update(
        {
            "status": "conflict" if relative > DRIFT_LIMIT else "reproduced",
            "old_rmse_physical": old_rmse,
            "new_rmse_physical": new_rmse,
            "relative_abs_diff": relative,
        }
    )
    return record


def finalize_report(
    output_dir: Path,
    config: ExperimentConfig,
    jobs: list[JobSpec] | None = None,
    fm_report: Path | None = None,
    report_name: str | None = None,
) -> Path | None:
    """Build the workbook only when every locked job has a valid result."""
    results = []
    for job in jobs or config.jobs():
        result_path = output_dir / job.key / "result.json"
        if not should_skip(result_path, config_fingerprint(job, config)):
            return None
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    if fm_report is not None:
        return build_fm_comparison_workbook(
            results,
            fm_report,
            output_dir / (report_name or "formal_comparison_subset_with_fm.xlsx"),
        )
    payload_path = write_report_payload(results, output_dir / "formal_report.json")
    workbook_path = output_dir / (report_name or "formal_comparison.xlsx")
    build_workbook(payload_path, workbook_path)
    return workbook_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preset",
        choices=("fm-h16",),
        help="locked experiment preset; fm-h16 reproduces the published four-station table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="override the output directory declared by the configuration",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=list(SEEDS))
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="resume also requires predictions.npz/checkpoint.pt to be present",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the report from complete locked result files without training",
    )
    parser.add_argument(
        "--fm-report",
        type=Path,
        help="locked Chronos test-confirmation JSON; builds the four-station FM table",
    )
    parser.add_argument(
        "--rebuild-fm",
        action="store_true",
        help="rerun the locked FM validation campaign and explicit test confirmation",
    )
    parser.add_argument(
        "--report-name",
        help="output workbook filename (not a path)",
    )
    return parser


def main(argv: list[str] | None = None) -> ExecutionSummary:
    args = _parser().parse_args(argv)
    config = load_experiment_config(args.config)
    output_dir = (args.output or config.output_dir).resolve()
    if args.preset == "fm-h16":
        if args.datasets or args.models or args.seeds:
            raise ValueError("--preset fm-h16 cannot be combined with dataset/model/seed filters")
        selected_datasets = set(FM_H16_DATASETS)
        selected_models = set(FM_H16_MODELS)
        selected_seeds = {2026}
        args.fm_report = args.fm_report or FM_H16_REPORT
        args.report_name = args.report_name or "formal_comparison_subset_with_fm.xlsx"
    else:
        selected_datasets = set(args.datasets or (item.name for item in config.datasets))
        selected_models = set(args.models or config.models)
        selected_seeds = set(args.seeds or config.seeds)
    jobs = [
        job
        for job in config.jobs()
        if job.dataset in selected_datasets
        and job.model in selected_models
        and job.seed in selected_seeds
    ]
    if args.preset == "fm-h16":
        order = {name: index for index, name in enumerate(FM_H16_DATASETS)}
        model_order = {name: index for index, name in enumerate(FM_H16_MODELS)}
        jobs.sort(key=lambda job: (order[job.dataset], model_order[job.model], job.seed))
    if args.report_name and Path(args.report_name).name != args.report_name:
        raise ValueError("--report-name must be a filename, not a path")
    if args.rebuild_fm and args.preset != "fm-h16":
        raise ValueError("--rebuild-fm requires --preset fm-h16")
    if args.rebuild_fm and args.report_only:
        raise ValueError("--rebuild-fm and --report-only are mutually exclusive")
    if args.dry_run:
        for job in jobs:
            print(job.key)
        summary = ExecutionSummary(len(jobs), 0, 0)
        print(f"planned={summary.planned} executed=0 skipped=0")
        return summary
    if args.report_only:
        workbook_path = finalize_report(
            output_dir,
            config,
            jobs=jobs,
            fm_report=args.fm_report.resolve() if args.fm_report else None,
            report_name=args.report_name,
        )
        if workbook_path is None:
            raise RuntimeError(
                "report-only requires complete results with matching config fingerprints"
            )
        print(f"report={workbook_path}", flush=True)
        return ExecutionSummary(len(jobs), 0, len(jobs))

    executed = 0
    skipped = 0
    drift_records = []
    for job in jobs:
        result_path = output_dir / job.key / "result.json"
        fingerprint = config_fingerprint(job, config)
        if args.resume and should_skip(
            result_path,
            fingerprint,
            require_artifacts=args.require_artifacts,
        ):
            skipped += 1
            print(f"[skip] {job.key}", flush=True)
            continue
        print(f"[run] {job.key}", flush=True)
        previous = None
        if result_path.is_file():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = None
            if previous is not None:
                _backup_legacy_result(result_path)
        result = run_job(job, config, artifacts_dir=result_path.parent)
        result["config_fingerprint"] = fingerprint
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(result_path)
        executed += 1
        if previous is not None:
            record = _drift_record(job, previous, result, fingerprint)
            drift_records.append(record)
            if record.get("status") == "conflict":
                print(
                    "[CONFLICT] {} RMSE {:.6f} -> {:.6f} (rel {:.4%})".format(
                        job.key,
                        record["old_rmse_physical"],
                        record["new_rmse_physical"],
                        record["relative_abs_diff"],
                    ),
                    flush=True,
                )
    summary = ExecutionSummary(len(jobs), executed, skipped)
    print(
        f"planned={summary.planned} executed={executed} skipped={skipped}",
        flush=True,
    )
    if drift_records:
        drift_path = output_dir / "rerun_drift_report.json"
        drift_path.write_text(
            json.dumps(drift_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        conflicts = [r for r in drift_records if r.get("status") == "conflict"]
        print(f"drift_report={drift_path} conflicts={len(conflicts)}", flush=True)
    if args.rebuild_fm:
        from run_correction_tune_h16 import main as run_fm_campaign

        fm_root = args.fm_report.resolve().parent
        run_fm_campaign(
            [
                "--config", str(args.config),
                "--output-root", str(fm_root),
                "--step0", "--round1", "--round2",
                "--shuffle-control", "--test-confirm",
            ]
        )
    workbook_path = finalize_report(
        output_dir,
        config,
        jobs=jobs,
        fm_report=args.fm_report.resolve() if args.fm_report else None,
        report_name=args.report_name,
    )
    if workbook_path is not None:
        print(f"report={workbook_path}", flush=True)
    return summary
