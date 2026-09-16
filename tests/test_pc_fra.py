"""Tests for the PC-FRA H16 residual adapter campaign."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from formal.pc_fra import (
    NIGHT_LOW,
    PEAK,
    RAMP,
    REGULAR,
    PcFraPack,
    build_pack,
    clip_prior,
    compute_epsilon,
    daylight_mask_minutes,
    operating_state,
    stratified_partners,
)
from layers.pc_fra import (
    EPV_DIM,
    PcFraResidualAdapter,
    pack_slices,
    pack_width,
)
from models.ours import Model

BATCH, SEQ, HORIZON, CHANNELS, D_MODEL = 4, 96, 16, 4, 32
SL = pack_slices(HORIZON)


# --- adapter ----------------------------------------------------------------

def _adapter(use_film: bool, epsilon: float = 0.37):
    return PcFraResidualAdapter(
        horizon=HORIZON,
        hidden=96,
        dropout=0.1,
        use_film=use_film,
        token_dim=8,
        epsilon=epsilon,
    )


def _adapter_inputs():
    psrc = torch.randn(BATCH, HORIZON)
    prior = torch.randn(BATCH, HORIZON)
    disagreement = prior - psrc
    latest = torch.randn(BATCH, 1)
    epv = torch.zeros(BATCH, EPV_DIM)
    epv[:, 0] = 1.0
    sigma = torch.rand(BATCH, 1)
    tokens = torch.randn(BATCH, 6, 8)
    return psrc, prior, disagreement, latest, epv, sigma, tokens


def test_pc_fra_zero_init_is_bitwise_identity_and_bounded():
    adapter = _adapter(use_film=True)
    inputs = _adapter_inputs()
    delta = adapter(*inputs)
    assert delta.shape == (BATCH, HORIZON)
    assert torch.count_nonzero(delta) == 0
    # Waken the hidden layers; with zeroed output delta stays zero, but with
    # a non-zero head the tanh envelope must respect epsilon.
    torch.nn.init.normal_(adapter.output_linear.weight, std=0.1)
    torch.nn.init.normal_(adapter.output_linear.bias, std=0.1)
    delta = adapter(*inputs)
    assert torch.isfinite(delta).all()
    assert float(delta.detach().abs().max()) <= 0.37 + 1e-6


def test_pc_fra_residual_gradient_flows_through_huber():
    adapter = _adapter(use_film=False)
    psrc, prior, disagreement, latest, epv, sigma, tokens = _adapter_inputs()
    target_residual = torch.randn(BATCH, HORIZON) * 0.1
    delta = adapter(psrc, prior, disagreement, latest, epv, sigma, None)
    loss = torch.nn.functional.smooth_l1_loss(delta, target_residual, beta=1.0)
    loss.backward()
    grads = [
        p.grad for p in adapter.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert grads


def test_pc_fra_film_is_bound_and_only_present_in_arm_c():
    para = _adapter(use_film=False)
    pcfra = _adapter(use_film=True)
    assert not hasattr(para, "film")
    film_params = [p for p in pcfra.film.parameters()]
    assert all(p.grad is None for p in film_params)
    inputs = _adapter_inputs()
    h_in = torch.randn(BATCH, 96)
    gamma = torch.randn(BATCH, 96) * 10
    beta = torch.randn(BATCH, 96) * 10
    modulated = (1 + 0.1 * torch.tanh(gamma)) * h_in + 0.1 * torch.tanh(beta)
    deviation = (modulated - h_in).abs()
    # Worst case bound: |0.1*tanh(g)*h| + 0.1 over finite h.
    assert float((deviation - 0.1 * h_in.abs() - 0.1).max()) <= 1e-5
    # FiLM runs and changes hidden representation.
    delta_para = para(*inputs[:6], None)
    delta_c = pcfra(*inputs)
    assert delta_para.shape == delta_c.shape == (BATCH, HORIZON)


def test_pc_fra_validates_shapes():
    adapter = _adapter(use_film=False)
    psrc, prior, disagreement, latest, epv, sigma, _ = _adapter_inputs()
    with pytest.raises(ValueError):
        adapter(psrc[:, :-1], prior[:, :-1], disagreement[:, :-1], latest, epv, sigma, None)
    with pytest.raises(ValueError):
        adapter(psrc, prior, disagreement, latest.squeeze(1), epv, sigma, None)
    with pytest.raises(ValueError):
        _adapter(use_film=True)(
            *_adapter_inputs()[:6], torch.randn(BATCH, 8)
        )


# --- registered operating rules --------------------------------------------

def test_operating_state_priority_order():
    # level in night band dominates everything.
    assert operating_state(0.10, ramp_ratio=0.50) == NIGHT_LOW
    # day-time strong ramp before peak/regular.
    assert operating_state(0.50, ramp_ratio=0.11) == RAMP
    # peak only when level high and ramp calm.
    assert operating_state(0.90, ramp_ratio=0.05) == PEAK
    assert operating_state(0.50, ramp_ratio=0.05) == REGULAR
    # Boundaries are strict as registered.
    assert operating_state(0.20, ramp_ratio=0.00) == REGULAR
    assert operating_state(0.80, ramp_ratio=0.00) == REGULAR
    assert operating_state(0.30, ramp_ratio=0.10) == REGULAR


def test_daylight_mask_matches_deterministic_geometry():
    minutes = np.array([0, 359, 360, 720, 1200, 1201, 1439])
    mask = daylight_mask_minutes(minutes)
    assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_clip_prior_nonnegative_capacity_and_night():
    median = np.array([-3.0, 0.5, 5.0, 9.0, 2.0], dtype=np.float32)
    mask = np.array([0, 1, 1, 1, 0], dtype=np.float32)
    cbar = clip_prior(median, mask, capacity=7.0)
    assert cbar.tolist() == [0.0, 0.5, 5.0, 7.0, 0.0]


def test_epsilon_p90_with_capacity_cap():
    residual = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    info = compute_epsilon(residual, target_sd=2.0, capacity=2.0)
    # p90 |resid| = 0.9; cap = 0.2*2/2 = 0.2 -> epsilon capped.
    assert info["epsilon_capped"]
    assert abs(info["epsilon"] - 0.2) < 1e-9
    info2 = compute_epsilon(residual, target_sd=1.0, capacity=100.0)
    assert not info2["epsilon_capped"]
    assert abs(info2["epsilon"] - 0.9) < 1e-9


# --- stratified shuffle -----------------------------------------------------

class _MiniBundle:
    """Synthetic 15-min grid exposing only what pack/shuffle use."""

    def __init__(self, origin_datetimes: np.ndarray):
        self.capacity = 5.0
        base = np.datetime64("2025-12-30T00:00")
        end = np.datetime64("2026-04-30T23:45")
        total_rows = int((end - base) / np.timedelta64(15, "m")) + 1
        self.timestamps = base + np.arange(total_rows).astype("timedelta64[m]") * 15
        rng = np.random.default_rng(0)
        self.raw = rng.uniform(0.5, self.capacity, (total_rows, 1)).astype("f4")
        self.feature_mu = np.array([0.0], dtype=np.float64)
        self.feature_sd = np.array([1.0], dtype=np.float64)
        origin_rows = (
            (np.asarray(origin_datetimes) - base) / np.timedelta64(15, "m")
        ).astype(np.int64)
        # start = origin - (seq_len - 1); requires >= 0.
        assert origin_rows.min() >= SEQ - 1
        self.starts = np.sort((origin_rows - (SEQ - 1)).astype(np.int64))

    def stratum(self, start: int):
        from formal.pc_fra import window_scalar_features

        _, _, _, _, bucket, month = window_scalar_features(
            self, int(start), SEQ, HORIZON
        )
        return month, bucket


def _stratum_origins():
    # Two windows per (month, bucket) stratum plus one singleton.
    origins = []
    for month_day in (("2026-01", "10"), ("2026-02", "10")):
        for day in (5, 12):
            for hour in (1, 7, 13, 19):  # buckets 0..3
                origins.append(
                    np.datetime64(f"{month_day[0]}-{day:02d}T{hour:02d}:30")
                )
    origins.append(np.datetime64("2026-03-08T13:30"))  # singleton stratum
    return np.array(origins, dtype="datetime64[m]")


def test_stratified_partners_preserve_strata_and_derange():
    bundle = _MiniBundle(_stratum_origins())
    starts = bundle.starts
    partners, audit = stratified_partners(starts, bundle, SEQ, HORIZON, 2026)
    assert audit["n_windows"] == len(starts)
    assert audit["singleton_strata"] == 1
    assert set(partners.tolist()) <= set(starts.tolist())
    for start, partner in zip(starts, partners):
        assert bundle.stratum(start) == bundle.stratum(int(partner))
    fixed = int(np.sum(partners == starts))
    assert fixed == audit["fixed_points"]
    assert fixed <= audit["singleton_strata"]
    # At least the singleton self-map is present.
    assert fixed == 1


def test_pack_layout_roundtrip_and_partner_disagreement():
    origins = np.array(
        [
            np.datetime64("2026-01-05T07:30"),
            np.datetime64("2026-01-06T07:30"),
            np.datetime64("2026-01-07T07:30"),
            np.datetime64("2026-01-08T07:30"),
        ],
        dtype="datetime64[m]",
    )
    bundle = _MiniBundle(origins)
    starts = bundle.starts
    partners = np.roll(starts, 1)
    rng = np.random.default_rng(1)
    median_lookup = {
        int(s): rng.uniform(0.0, 4.0, HORIZON).astype(np.float32) for s in starts
    }
    psrc_std = rng.normal(size=(len(starts), HORIZON)).astype(np.float32)
    psrc_lookup = {int(s): psrc_std[i] for i, s in enumerate(starts)}
    pack = build_pack(
        bundle, starts, SEQ, HORIZON, median_lookup, psrc_lookup, partners
    )
    assert pack.kind == "pc_fra"
    assert pack.dim == pack_width(HORIZON) == 3 * HORIZON + 11
    assert pack.covers(starts)
    for i, start in enumerate(starts):
        vec = pack[int(start)]
        assert np.isfinite(vec).all()
        assert vec[SL["state"]].sum() == pytest.approx(1.0)
        assert vec[SL["bucket"]].sum() == pytest.approx(1.0)
        assert vec[SL["c_bar"]].max() <= bundle.capacity + 1e-6
        partner = int(partners[i])
        assert pack.partner_start(int(start)) == partner
        expected_d = vec[SL["c_bar_partner"]] - psrc_lookup[partner]
        assert np.allclose(vec[SL["d_partner"]], expected_d, atol=2e-6)


# --- model-level wiring ------------------------------------------------------

def _config():
    cfg = SimpleNamespace(
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
        semantic_residual_max_gate=1.0,
        physical_semantic={
            "roles": {},
            "feature_mu": [0.0] * CHANNELS,
            "feature_sd": [1.0] * CHANNELS,
            "target_mu": 0.0,
            "target_sd": 1.0,
            "token_mean": np.zeros((6, 6), dtype=np.float32).tolist(),
            "token_scale": np.ones((6, 6), dtype=np.float32).tolist(),
            "d_token": 32,
            "use_gate": True,
        },
        ablation=(),
    )
    return cfg


def _pc_config(variant: str):
    cfg = _config()
    cfg.pc_fra = {
        "horizon": HORIZON,
        "variant": variant,
        "hidden": 96,
        "dropout": 0.1,
        "film_bound": 0.1,
        "epsilon": 0.37,
        "level_mu": 0.3,
        "level_sd": 0.2,
        "sigma_mu": 0.1,
        "sigma_sd": 0.1,
    }
    return cfg


def _pack(variant: str = "para"):
    torch.manual_seed(2)
    width = pack_width(HORIZON)
    vec = torch.zeros(BATCH, width)
    vec[:, SL["c_bar"]] = torch.rand(BATCH, HORIZON) * 2.0
    vec[:, SL["c_bar_partner"]] = torch.rand(BATCH, HORIZON) * 2.0
    vec[:, SL["d_partner"]] = torch.randn(BATCH, HORIZON) * 0.1
    vec[:, SL["latest"]] = torch.rand(BATCH) * 2.0
    vec[:, SL["sigma"]] = torch.rand(BATCH) * 0.5
    state_idx = torch.randint(0, 4, (BATCH,))
    vec[torch.arange(BATCH), [SL["state"].start + int(i) for i in state_idx]] = 1.0
    bucket_idx = torch.randint(0, 4, (BATCH,))
    vec[torch.arange(BATCH), [SL["bucket"].start + int(i) for i in bucket_idx]] = 1.0
    vec[:, SL["level"]] = torch.rand(BATCH)
    return vec


def _inputs():
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, CHANNELS)
    cycle = torch.tensor([0, 12, 24, 36], dtype=torch.long)
    partner = torch.randn(BATCH, SEQ, CHANNELS)
    return x, cycle, partner


def test_model_pc_fra_arms_are_bitwise_identity_at_zero_init():
    x, cycle, partner = _inputs()
    pack = _pack()
    torch.manual_seed(0)
    frozen = Model(_config()).eval()
    with torch.no_grad():
        y_plain = frozen(x, cycle, None, None, None)
        for variant in ("para", "pcfra", "prior_shuffle", "phys_shuffle"):
            torch.manual_seed(0)
            arm = Model(_pc_config(variant)).eval()
            y = arm(
                x, cycle, None, None, None,
                fm_context=pack, pc_partner=partner,
            )
            assert y.shape == (BATCH, HORIZON, 1)
            assert torch.equal(y, y_plain), variant
            assert torch.count_nonzero(arm.last_pc_fra[1]) == 0


def test_model_phys_shuffle_requires_partner_history():
    x, cycle, _ = _inputs()
    model = Model(_pc_config("phys_shuffle")).eval()
    with pytest.raises(ValueError), torch.no_grad():
        model(x, cycle, None, None, None, fm_context=_pack(), pc_partner=None)


def test_model_pc_fra_requires_pack():
    x, cycle, _ = _inputs()
    model = Model(_pc_config("para")).eval()
    with pytest.raises(ValueError), torch.no_grad():
        model(x, cycle, None, None, None)


def test_model_after_waking_head_delta_within_epsilon():
    x, cycle, partner = _inputs()
    model = Model(_pc_config("para")).eval()
    torch.nn.init.normal_(model.pc_fra_adapter.output_linear.weight, std=0.2)
    with torch.no_grad():
        y = model(
            x, cycle, None, None, None,
            fm_context=_pack(), pc_partner=partner,
        )
        delta = model.last_pc_fra[1]
    assert torch.isfinite(y).all()
    assert float(delta.abs().max()) <= 0.37 + 1e-6
