"""Reproduce every mechanism-analysis artifact from formal prediction dumps.

By default reads ``outputs/final`` (the audited H24 campaign) and writes
under ``outputs/analysis``.  Other campaigns pass ``--final-dir`` and
``--horizon`` (e.g. the five-station H16 single-seed rerun uses
``--final-dir outputs/final_h16 --horizon 16 --seeds 2026``).
Requires every archived run that is read to carry ``predictions.npz``
dumps (see ``formal/runner.py --require-artifacts``).
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from analysis import case_study, gate, horizon, ranking, significance
from analysis.artifacts import DATASETS, DEFAULT_ANALYSIS_DIR, DEFAULT_FINAL_DIR

STAGES = ("horizon", "gate", "bootstrap", "cases", "ranking")


def _station_list(raw: list[str] | None) -> tuple[str, ...]:
    if not raw:
        return tuple(DATASETS)
    unknown = [name for name in raw if name not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; known: {list(DATASETS)}")
    return tuple(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="analysis output directory",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help="formal run directory containing per-job folders with dumps",
    )
    parser.add_argument("--horizon", type=int, default=24, choices=(16, 24))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2026, 2027, 2028),
        help="PSRC seeds archived for this campaign",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="subset of stations to analyse (default: all)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="formal_report.json for the ranking stage (default: "
        "<final-dir>/formal_report.json)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=STAGES,
        default=list(STAGES),
        help="run a subset of stages",
    )
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path(args.final_dir)
    seeds = tuple(args.seeds)
    datasets = _station_list(args.datasets)
    report_path = Path(args.report) if args.report else final_dir / "formal_report.json"

    runners = {
        "horizon": lambda: horizon.run(
            output_dir,
            horizon=args.horizon,
            seeds=seeds,
            final_dir=final_dir,
            datasets=datasets,
        ),
        "gate": lambda: gate.run(
            output_dir,
            horizon=args.horizon,
            seeds=seeds,
            final_dir=final_dir,
            datasets=datasets,
        ),
        "bootstrap": lambda: significance.run(
            output_dir,
            horizon=args.horizon,
            seeds=seeds,
            final_dir=final_dir,
            datasets=datasets,
        ),
        "cases": lambda: case_study.run(
            output_dir,
            horizon=args.horizon,
            seed=seeds[0],
            final_dir=final_dir,
            datasets=datasets,
        ),
        "ranking": lambda: ranking.run(output_dir, report_path=report_path),
    }
    summary: dict[str, object] = {}
    for stage in args.only:
        print(f"[analysis] stage: {stage}")
        try:
            summary[stage] = runners[stage]()
        except Exception:  # noqa: BLE001 - keep other stages runnable
            summary[stage] = {"error": traceback.format_exc()}
            print(f"[analysis] stage {stage} failed:\n{summary[stage]['error']}")
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    failed = [stage for stage, value in summary.items() if "error" in value]
    print(f"[analysis] done; failed stages: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
