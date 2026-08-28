from pathlib import Path

import pytest

import run


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_site_profiles_keep_locked_h48_parameters(tmp_path: Path):
    site1b = run.build_command("site1b", "validation", tmp_path / "1b.json")
    site24 = run.build_command("site24", "validation", tmp_path / "24.json")

    assert Path(option(site1b, "--data")).name == "DKASC_site1B_2021MarMay_15min.csv"
    assert option(site1b, "--d-model") == "32"
    assert option(site1b, "--batch") == "64"
    assert option(site1b, "--loss-kind-override") == "mse_ramp_phys"
    assert option(site1b, "--ours-cma-lr-multiplier") == "0.2"

    assert Path(option(site24, "--data")).name == "DKASC_site24_2021MarMay_15min.csv"
    assert option(site24, "--d-model") == "96"
    assert option(site24, "--batch") == "96"
    assert option(site24, "--loss-kind-override") == "mse"
    assert option(site24, "--ours-cma-lr-multiplier") == "0.1"

    for command in (site1b, site24):
        assert option(command, "--models") == "ours"
        assert option(command, "--seq-len") == "96"
        assert option(command, "--horizon") == "48"
        assert option(command, "--seed") == "2026"
        assert "--validation-only" in command
        assert "--fsra-loss-weight" not in command
        assert "--ours-cma-scale-route" not in command
        assert "--ours-solar-anchor" not in command


def test_mode_controls_budget_and_test_access(tmp_path: Path):
    smoke = run.build_command("site1b", "smoke", tmp_path / "smoke.json")
    formal = run.build_command("site1b", "formal", tmp_path / "formal.json")

    assert option(smoke, "--epochs") == "1"
    assert option(smoke, "--max-train-windows") == "128"
    assert option(smoke, "--max-val-windows") == "64"
    assert "--validation-only" in smoke

    assert option(formal, "--epochs") == "100"
    assert option(formal, "--max-train-windows") == "8000"
    assert option(formal, "--max-test-windows") == "0"
    assert "--validation-only" not in formal


def test_output_must_remain_inside_project(tmp_path: Path):
    allowed = run.ROOT / "results" / "ours_h48" / "smoke.json"
    assert run.local_path(allowed) == allowed.resolve()

    with pytest.raises(ValueError, match="output must stay inside"):
        run.local_path(tmp_path / "outside.json")
