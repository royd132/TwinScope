"""Tests for the pre-registered PSRC component-ablation switches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ours import Model

BATCH, SEQ, HORIZON, CHANNELS, D_MODEL = 2, 96, 16, 4, 32

ALL_ARMS = (
    frozenset(),
    frozenset({"no_gtr"}),
    frozenset({"no_ldrive"}),
    frozenset({"equal_gl_fusion"}),
    frozenset({"no_spectral"}),
    frozenset({"equal_tf_fusion"}),
    frozenset({"numerical_backbone"}),
    frozenset({"numeric_residual"}),
    frozenset({"no_gate"}),
)


def _config(ablation, *, use_gate=True, max_gate=1.0):
    token_mean = np.zeros((6, 6), dtype=np.float32).tolist()
    token_scale = np.ones((6, 6), dtype=np.float32).tolist()
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="MS",
        seq_len=SEQ,
        pred_len=HORIZON,
        enc_in=CHANNELS,
        d_model=D_MODEL,
        cycle=96,
        dropout=0.0,
        corpatch_heads=4,
        semantic_heads=4,
        patch_hours=[1.0, 2.0, 4.0],
        gtr_period=24,
        sample_hours=0.25,
        semantic_residual_max_correction=0.5,
        semantic_residual_max_gate=max_gate,
        physical_semantic={
            "roles": {},
            "feature_mu": [0.0] * CHANNELS,
            "feature_sd": [1.0] * CHANNELS,
            "target_mu": 0.0,
            "target_sd": 1.0,
            "token_mean": token_mean,
            "token_scale": token_scale,
            "d_token": 32,
            "use_gate": use_gate,
        },
        ablation=tuple(sorted(ablation or ())),
    )


def _inputs():
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, CHANNELS)
    cycle = torch.tensor([0, 12], dtype=torch.long)
    return x, cycle


def _build(ablation, *, use_gate=True, max_gate=1.0):
    torch.manual_seed(0)
    model = Model(_config(ablation, use_gate=use_gate, max_gate=max_gate))
    model.eval()
    return model


def test_default_without_ablation_attribute_matches_full():
    x, cycle = _inputs()
    cfg_default = _config(None)
    del cfg_default.ablation
    torch.manual_seed(0)
    m_default = Model(cfg_default).eval()
    with torch.no_grad():
        y_default = m_default(x, cycle, None, None, None)
    m_full = _build(frozenset())
    with torch.no_grad():
        y_full = m_full(x, cycle, None, None, None)
    assert torch.equal(y_default, y_full)


@pytest.mark.parametrize("ablation", ALL_ARMS)
def test_every_arm_runs_with_valid_shape_and_finite_output(ablation):
    use_gate = "no_gate" not in ablation
    model = _build(ablation, use_gate=use_gate, max_gate=1.0)
    x, cycle = _inputs()
    with torch.no_grad():
        y = model(x, cycle, None, None, None)
    assert y.shape == (BATCH, HORIZON, 1)
    assert torch.isfinite(y).all()
    assert set(model.ablation) == set(ablation)


def test_numerical_backbone_emits_base_with_zero_delta_and_gate():
    model = _build(frozenset({"numerical_backbone"}))
    x, cycle = _inputs()
    with torch.no_grad():
        y = model(x, cycle, None, None, None).squeeze(-1)
    base, delta = model.last_semantic_decomposition
    assert torch.equal(y, base)
    assert torch.count_nonzero(delta) == 0
    assert torch.count_nonzero(model.last_psrc_gate) == 0


def test_no_spectral_bypasses_spectral_branch_and_tf_router():
    model = _build(frozenset({"no_spectral"}))
    calls = {"spectral": 0, "router": 0}
    model.spectral.register_forward_hook(lambda *a: calls.__setitem__("spectral", calls["spectral"] + 1))
    model.tf_router.register_forward_hook(lambda *a: calls.__setitem__("router", calls["router"] + 1))
    x, cycle = _inputs()
    with torch.no_grad():
        model(x, cycle, None, None, None)
    assert calls == {"spectral": 0, "router": 0}
    # GTR and L-Drive still feed the temporal path.
    assert calls["spectral"] == 0


def test_no_gate_forces_confidence_gate_to_exactly_one():
    model = _build(frozenset({"no_gate"}), use_gate=False, max_gate=1.0)
    assert model.residual_calibration.confidence_head is None
    x, cycle = _inputs()
    with torch.no_grad():
        model(x, cycle, None, None, None)
    assert torch.allclose(
        model.last_psrc_gate, torch.ones_like(model.last_psrc_gate)
    )


def test_numeric_residual_skips_physics_but_still_corrects():
    model = _build(frozenset({"numeric_residual"}))
    physical_calls = {"n": 0}
    model.physical_semantic_encoder.register_forward_hook(
        lambda *a: physical_calls.__setitem__("n", physical_calls["n"] + 1)
    )
    # Break the zero-init so the residual head emits a non-zero correction.
    torch.nn.init.normal_(model.residual_calibration.residual_head.weight, std=0.05)
    x, cycle = _inputs()
    with torch.no_grad():
        y = model(x, cycle, None, None, None).squeeze(-1)
    base, delta = model.last_semantic_decomposition
    assert physical_calls["n"] == 0          # no physics-semantic conditioning
    assert torch.count_nonzero(delta) > 0   # numeric residual still corrects
    assert not torch.allclose(y, base)
    assert torch.allclose(y, base + delta)


def test_full_uses_physical_semantic_encoder():
    model = _build(frozenset())
    calls = {"n": 0}
    model.physical_semantic_encoder.register_forward_hook(
        lambda *a: calls.__setitem__("n", calls["n"] + 1)
    )
    x, cycle = _inputs()
    with torch.no_grad():
        model(x, cycle, None, None, None)
    assert calls["n"] == 1
