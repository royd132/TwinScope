import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_recent_pv_benchmarks as benchmark


physical_loss = benchmark.physical_loss


@pytest.fixture
def point_loss_case():
    prediction = torch.tensor([[0.0, 2.0]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction)
    stats = {
        "target_sd": 1.0,
        "target_mu": 0.0,
        "capacity": 1.0,
        "dt_hours": 0.25,
    }
    return prediction, target, mask, stats


def test_mae_mode_uses_absolute_point_error(point_loss_case):
    """Catch accidentally routing the MAE trial through the legacy MSE branch."""
    prediction, target, mask, stats = point_loss_case

    loss = physical_loss(prediction, target, "mae", stats, mask=mask)

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_mse_mae_mode_balances_squared_and_absolute_error(point_loss_case):
    """Catch either component being dropped from the balanced point loss."""
    prediction, target, mask, stats = point_loss_case

    loss = physical_loss(prediction, target, "mse_mae", stats, mask=mask)

    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_backbone_screen_disables_both_adapter_training_stages():
    """Catch a backbone trial silently fitting FSRA or Prompt/CMA adapters."""
    policy = getattr(benchmark, "adapter_stage_policy", None)

    assert policy is not None, "runner must expose the adapter-stage policy"
    assert policy(
        skip_adapter_stages=True,
        fsra_requested=True,
        prompt_requested=True,
    ) == (False, False)


def test_full_trial_keeps_requested_adapter_training_stages():
    """Catch the screening switch leaking into a later full CMA trial."""
    policy = getattr(benchmark, "adapter_stage_policy", None)

    assert policy is not None, "runner must expose the adapter-stage policy"
    assert policy(
        skip_adapter_stages=False,
        fsra_requested=True,
        prompt_requested=True,
    ) == (True, True)
