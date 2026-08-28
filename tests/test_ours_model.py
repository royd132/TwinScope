import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layers.semantic_alignment import AdaptiveMultiHeadCMAResidual
from models.factory import build_model as build_from_factory
from models.ours import Model
from models.ours_v73 import Model as LegacyModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_recent_pv_benchmarks as benchmark


def model_config(**overrides):
    values = {
        "task_name": "long_term_forecast",
        "features": "MS",
        "seq_len": 96,
        "pred_len": 48,
        "enc_in": 9,
        "d_model": 32,
        "cycle": 96,
        "sample_hours": 0.25,
        "physics_indices": [0, 1, 2, 3, 6, 7, 8],
        "semantic_prompt_dim": 104,
        "patch_hours": (1.0, 2.0, 4.0, 8.0),
        "gtr_period": 24,
        "revin_mode": "full",
        "dropout": 0.0,
        "corpatch_heads": 4,
        "cma_heads": 4,
        "cma_dropout": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def model_inputs():
    return (
        torch.randn(2, 96, 9),
        torch.tensor([0, 24]),
        None,
        torch.randn(2, 9, 104),
    )


def test_cma_aligns_prompt_without_a_trainable_prompt_encoder():
    module = AdaptiveMultiHeadCMAResidual(
        seq_len=96,
        d_model=32,
        prompt_dim=104,
        num_heads=4,
        dropout=0.0,
    )
    history = torch.randn(2, 96, 9)
    prompt = torch.randn(2, 9, 104)
    state = torch.randn(2, 32)

    residual, gate, strength, weights = module(history, prompt, state)
    residual.square().mean().backward()

    assert residual.shape == history.shape
    assert gate.shape == (2, 9, 1)
    assert weights.shape == (2, 9, 4)
    assert torch.isfinite(residual).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 9))
    assert 0.0 < float(strength.detach()) < 1.0
    assert not any(
        isinstance(child, nn.TransformerEncoder) for child in module.modules()
    )
    assert not hasattr(module, "prompt_encoder")
    for head in module.cma_heads:
        assert head.in_proj_weight.grad is not None
        assert torch.count_nonzero(head.in_proj_weight.grad) > 0


def test_cma_rejects_prompt_with_wrong_variable_axis():
    module = AdaptiveMultiHeadCMAResidual(96, 32, 104, num_heads=4, dropout=0.0)

    with pytest.raises(ValueError, match=r"\[batch, variables, prompt_dim\]"):
        module(torch.randn(2, 96, 9), torch.randn(2, 8, 104), torch.randn(2, 32))


def test_ours_forward_is_finite_and_backpropagates_through_numeric_and_cma_paths():
    model = Model(model_config())
    output = model(*model_inputs(), mask=None)

    assert output.shape == (2, 48, 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.count_nonzero(model.gtr.cycle_query.grad) > 0
    cma_grad = model.semantic_adapter.cma_heads[0].in_proj_weight.grad
    assert cma_grad is not None
    assert torch.count_nonzero(cma_grad) > 0


def test_zero_semantic_strength_and_missing_prompt_preserve_numerical_forecast():
    model = Model(model_config()).eval()
    history, cycle, decoder, prompt = model_inputs()

    model.semantic_enabled = True
    model.semantic_strength = 0.0
    with torch.no_grad():
        zero_strength = model(history, cycle, decoder, prompt)
    model.semantic_enabled = False
    with torch.no_grad():
        disabled = model(history, cycle, decoder, prompt)
    model.semantic_enabled = True
    model.semantic_strength = 1.0
    with torch.no_grad():
        missing_prompt = model(history, cycle, decoder, None)

    torch.testing.assert_close(zero_strength, disabled, rtol=0.0, atol=0.0)
    torch.testing.assert_close(missing_prompt, disabled, rtol=0.0, atol=0.0)


def test_corpatch_routes_normalized_scales_without_cross_scale_attention():
    model = Model(model_config()).eval()
    with torch.no_grad():
        model(*model_inputs())

    assert len(model.corpatch.temporal_mixers) == 4
    assert len(model.corpatch.variable_attentions) == 4
    assert not hasattr(model.corpatch, "cross_scale_attention")
    torch.testing.assert_close(
        model.last_scale_weights.sum(-1),
        torch.ones_like(model.last_scale_weights[..., 0]),
    )
    for retired in (
        "fsra_adapter",
        "scale_prompt_router",
        "anchor",
        "nwp_decoder",
        "variant",
    ):
        assert not hasattr(model, retired)


def test_corpatch_rejects_incompatible_attention_width():
    with pytest.raises(ValueError, match="divisible"):
        Model(model_config(d_model=30, corpatch_heads=4))


def test_factory_maps_public_ours_name_to_clean_model():
    assert isinstance(build_from_factory("ours", model_config()), Model)


def test_shared_engine_builds_clean_ours_with_physical_patch_scales():
    args = SimpleNamespace(
        seq_len=96,
        horizon=48,
        d_model=32,
        patch_len=16,
        stride=8,
        cycle_len=96,
        ours_patch_hours=(1.0, 2.0, 4.0, 8.0),
        ours_gtr_period=24,
        ours_revin_mode="full",
        ours_dropout=0.0,
        ours_corpatch_heads=4,
        ours_cma_heads=4,
        ours_cma_dropout=0.0,
    )
    stats = {
        "dt_hours": 0.25,
        "physics_indices": [0, 1, 2, 3, 6, 7, 8],
        "quality_indices": [],
        "_dynamic_prompt_dim": 104,
    }

    model, loss_kind = benchmark.build_model("ours", args, channels=9, stats=stats)

    assert isinstance(model, Model)
    assert model.patch.patch_lengths == [4, 8, 16, 32]
    assert model.prompt_dim == 104
    assert loss_kind == "mse_ramp_phys_night"


def test_historical_long_name_remains_isolated_from_public_ours_model():
    args = SimpleNamespace(
        seq_len=96,
        horizon=48,
        d_model=32,
        patch_len=16,
        stride=8,
        cycle_len=96,
        anchor_output_mode="direct",
        ours_solar_anchor=False,
        ours_patch_hours=(1.0, 2.0, 4.0, 8.0),
        ours_gtr_period=24,
        ours_revin_mode="full",
        ours_dropout=0.0,
        ours_corpatch_heads=4,
        ours_cma_heads=4,
        ours_cma_dropout=0.0,
        ours_cma_scale_route=False,
        ours_cma_prompt_features="gpt2_numeric",
    )
    stats = {
        "dt_hours": 0.25,
        "physics_indices": [0, 1, 2, 3, 6, 7, 8],
        "quality_indices": [],
        "spectral_indices": list(range(9)),
        "feature_names": [f"feature_{index}" for index in range(8)] + ["Target"],
        "_dynamic_prompt_dim": 104,
    }

    model, _ = benchmark.build_model(
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma",
        args,
        channels=9,
        stats=stats,
    )

    assert isinstance(model, LegacyModel)


def test_current_result_metadata_omits_retired_branches_only_for_ours():
    payload = {
        "model": "ours",
        "semantic_strength": 1.0,
        "fsra_strength": 0.0,
        "anchor_output_mode": "gated",
        "frequency_v9_mode": None,
        "reference_path_uses_future_solar_geometry": False,
        "prompt_alignment": "linear_projection_norm",
    }

    cleaned = benchmark.clean_current_ours_result(payload)

    assert cleaned == {
        "model": "ours",
        "semantic_strength": 1.0,
        "prompt_alignment": "linear_projection_norm",
    }
    legacy = dict(payload, model="legacy")
    assert benchmark.clean_current_ours_result(legacy) == legacy
