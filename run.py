"""Run the current Site1B/Site24 Ours H48 experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "experiments" / "run_recent_pv_benchmarks.py"
DATA_ROOT = ROOT / "datasets" / "australia" / "DKASC_paper_2021"

SITES = {
    "site1b": {
        "data": DATA_ROOT / "DKASC_site1B_2021MarMay_15min.csv",
        "batch": "64",
        "d_model": "32",
        "weight_decay": "0.0001",
        "loss": "mse_ramp_phys",
        "ramp": "0.1",
        "route": "0.02",
        "specialization": "0.01",
        "cma_lr": "0.2",
    },
    "site24": {
        "data": DATA_ROOT / "DKASC_site24_2021MarMay_15min.csv",
        "batch": "96",
        "d_model": "96",
        "weight_decay": "0.00003",
        "loss": "mse",
        "ramp": "0",
        "route": "0",
        "specialization": "0",
        "cma_lr": "0.1",
    },
}


def build_command(site: str, mode: str, output: Path) -> list[str]:
    """Translate a locked public profile to the shared experiment engine."""
    profile = SITES[site]
    smoke = mode == "smoke"
    options = {
        "--data": profile["data"],
        "--target-col": "Target",
        "--out": output,
        "--models": "ours",
        "--seq-len": 96,
        "--horizon": 48,
        "--cycle-len": 96,
        "--epochs": 1 if smoke else 100,
        "--patience": 1 if smoke else 10,
        "--max-windows": 128 if smoke else 1000,
        "--max-train-windows": 128 if smoke else 8000,
        "--max-val-windows": 64 if smoke else 0,
        "--max-test-windows": 64 if smoke else 0,
        "--split-train-fraction": 0.70,
        "--split-validation-end-fraction": 0.85,
        "--train-sampling-policy": "epoch_uniform",
        "--batch": profile["batch"],
        "--d-model": profile["d_model"],
        "--lr": 0.0005,
        "--weight-decay": profile["weight_decay"],
        "--optimizer": "adam",
        "--base-scheduler": "none",
        "--checkpoint-average-k": 3,
        "--observed-policy": "all",
        "--future-feature-policy": "none",
        "--prompt-future-solar-policy": "none",
        "--evaluation-projection": "nonnegative",
        "--seed": 2026,
        "--ours-revin-mode": "full",
        "--ours-gtr-period": 24,
        "--ours-dropout": 0.05,
        "--ours-corpatch-heads": 4,
        "--ours-cma-heads": 4,
        "--ours-cma-dropout": 0.1,
        "--ours-cma-training": "staged",
        "--ours-cma-prompt-features": "gpt2_numeric",
        "--ours-cma-epochs": 8,
        "--ours-cma-lr-multiplier": profile["cma_lr"],
        "--ours-semantic-strength-max": 2.0,
        "--ours-semantic-strength-steps": 41,
        "--loss-kind-override": profile["loss"],
        "--ramp-loss-weight": profile["ramp"],
        "--route-loss-weight": profile["route"],
        "--router-specialization-weight": profile["specialization"],
        "--semantic-mae-weight": 0.20,
        "--selection-metric": "all_mse",
    }
    command = [sys.executable, str(ENGINE)]
    for flag, value in options.items():
        command.extend((flag, str(value)))
    command.extend(("--ours-patch-hours", "1", "2", "4", "8"))
    if mode != "formal":
        command.append("--validation-only")
    return command


def local_path(path: Path) -> Path:
    """Reject outputs that could overwrite files outside this project."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"output must stay inside {ROOT}: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=SITES, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "validation", "formal"),
        default="validation",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = SITES[args.site]["data"]
    if not data.is_file():
        parser.error(f"dataset not found: {data}")
    output = args.out or (
        ROOT / "results" / "ours_h48" / args.mode / f"{args.site}_s2026.json"
    )
    try:
        output = local_path(output)
    except ValueError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(args.site, args.mode, output), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
