"""Tests for the Chronos-only prior tuning module (no Chronos inference)."""

from __future__ import annotations

import numpy as np
import pytest

from formal.chronos_prior import (
    CHRONOS2_QUANTILES,
    HORIZON_SEGMENTS,
    L96_PHASE_A_READOUTS,
    READOUT_FULL_MEAN,
    READOUT_Q50,
    READOUT_TRIMMED,
    SCALE_GAMMAS,
    SINGLE_Q_READOUTS,
    STAGE_A_CONTEXTS,
    STAGE_A_READOUTS,
    contract_prior,
    per_horizon_corr,
    pool_diagnostics,
    prior_cache_path_v4,
    readout,
    readout_quantile,
    residual_diagnostics,
    scale_disagreement,
    scale_disagreement_segments,
    select_worst_site,
    skew_readout,
    skew_readout_name,
    select_stage_a,
    valid_starts,
)

H = 16
TAUS = np.asarray(CHRONOS2_QUANTILES)


def _grid_from_quantile_fn(fn, n=8):
    """q_grid [N, Q, H] where Q_tau = fn(tau) per horizon column."""
    vals = np.asarray([fn(float(t)) for t in TAUS], dtype=np.float32)
    return np.broadcast_to(vals[None, :, None], (n, len(TAUS), H)).copy()


# --- readouts ---------------------------------------------------------------

def test_median_readout_picks_native_q50():
    grid = _grid_from_quantile_fn(lambda t: 2.0 * t)
    out = readout(grid, TAUS, READOUT_Q50)
    assert out.shape == (8, H)
    assert np.allclose(out, 1.0)


def test_trimmed_and_full_means_match_trapezoid_integrals():
    grid = _grid_from_quantile_fn(lambda t: t**2)
    trimmed = readout(grid, TAUS, READOUT_TRIMMED)[0, 0]
    full = readout(grid, TAUS, READOUT_FULL_MEAN)[0, 0]
    # Expectations use the SAME discrete trapezoid rule over the native
    # grid points (the registered definition), written independently here.
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    qvals = TAUS**2
    m_trim = (TAUS >= 0.2 - 1e-12) & (TAUS <= 0.8 + 1e-12)
    expected_trim = trapz(qvals[m_trim], TAUS[m_trim]) / 0.6
    expected_full = trapz(qvals, TAUS) / (TAUS[-1] - TAUS[0])
    assert abs(trimmed - expected_trim) < 1e-7
    assert abs(full - expected_full) < 1e-7
    # Right-skewed distribution: full mean > trimmed mean > median.
    assert full > trimmed > readout(grid, TAUS, READOUT_Q50)[0, 0]
    # Both readouts stay inside the quantile envelope.
    assert TAUS[0] ** 2 < trimmed < TAUS[-1] ** 2


def test_native_grid_covers_every_registered_quantile():
    grid = _grid_from_quantile_fn(lambda t: t)
    for q in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90):
        out = readout_quantile(grid, TAUS, q)
        assert np.allclose(out, q)
    with pytest.raises(ValueError):
        readout_quantile(grid, TAUS, 0.37)
    with pytest.raises(ValueError):
        readout(grid, TAUS, "nonsense")


def test_single_quantile_names_are_formal_stage_a_candidates():
    # 9 single native quantiles + median + trimmed + full = 12 readouts.
    assert len(SINGLE_Q_READOUTS) == 9
    assert SINGLE_Q_READOUTS[0] == "q0.10" and SINGLE_Q_READOUTS[-1] == "q0.90"
    assert len(STAGE_A_READOUTS) == 12
    assert set(SINGLE_Q_READOUTS) <= set(STAGE_A_READOUTS)
    assert {READOUT_Q50, READOUT_TRIMMED, READOUT_FULL_MEAN} <= set(STAGE_A_READOUTS)
    # 4 contexts x 12 readouts = 48 formal configurations.
    assert len(STAGE_A_CONTEXTS) * len(STAGE_A_READOUTS) == 48
    # The named single-quantile readout goes through readout() itself.
    grid = _grid_from_quantile_fn(lambda t: 10.0 * t)
    out = readout(grid, TAUS, "q0.40")
    assert out.shape == (8, H) and np.allclose(out, 4.0)


def test_registered_contexts_and_segments():
    assert STAGE_A_CONTEXTS == (96, 192, 384, 672)
    covered = sum(b - a for a, b in HORIZON_SEGMENTS)
    assert covered == H


# --- long-context window validity ------------------------------------------

def test_valid_starts_enforces_full_context_origin_alignment():
    starts = np.array([0, 100, 576, 577, 1000])
    # Need start + 96 - L_C >= 0.
    assert valid_starts(starts, 96, 96).tolist() == starts.tolist()
    assert valid_starts(starts, 96, 384).tolist() == [576, 577, 1000]
    assert valid_starts(starts, 96, 672).tolist() == [576, 577, 1000]
    assert int(valid_starts(np.array([575]), 96, 672).size) == 0


# --- residual information metrics -------------------------------------------

def test_residual_diagnostics_exact_linear_relationship():
    rng = np.random.default_rng(0)
    target = rng.uniform(0, 5, (50, H))
    psrc = target - rng.uniform(0, 1, (50, H))  # PSRC under-forecasts
    scale = 2.0
    # Chronos prior explains the residual with coefficient 0.5.
    residual = target - psrc
    prior = psrc + 0.5 * residual
    diag = residual_diagnostics(prior, psrc, target, target_sd=scale)
    d = ((prior - psrc) / scale).reshape(-1)
    r = (residual / scale).reshape(-1)
    assert diag["residual_corr"] == pytest.approx(1.0, abs=1e-6)
    assert abs(diag["ols_beta"] - 2.0) < 1e-5  # beta*D=R -> beta=2
    assert diag["residual_sse_reduction"] == pytest.approx(1.0, abs=1e-6)
    assert diag["direction_accuracy"] == pytest.approx(1.0)
    assert diag["oracle_projected_residual_rmse_std"] == pytest.approx(0.0, abs=1e-7)
    assert diag["prior_mbe_physical"] == pytest.approx(
        float(np.mean(prior - target)), abs=1e-6
    )


def test_residual_diagnostics_zero_disagreement_is_safe():
    target = np.zeros((4, H))
    psrc = np.zeros((4, H))
    diag = residual_diagnostics(psrc, psrc, target, target_sd=1.0)
    assert np.isnan(diag["residual_corr"])
    assert diag["direction_n_rows"] == 0


def test_per_horizon_corr_length_and_anti_correlation():
    n = 20
    window_signal = np.linspace(-1.0, 2.0, n)[:, None] + np.arange(H)[None, :]
    target = np.broadcast_to(window_signal, (n, H)).copy()
    psrc = np.zeros_like(target)
    prior = -target  # disagreement perfectly anti-aligned with residual
    corr = per_horizon_corr(prior, psrc, target, target_sd=1.0)
    assert len(corr) == H
    assert all(c == pytest.approx(-1.0, abs=1e-6) for c in corr)


# --- pooled selection --------------------------------------------------------

def _arrays(corr_slope, n=40, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=n * H)
    d = corr_slope * r + rng.normal(scale=0.5, size=n * H)
    e = rng.normal(scale=1.3, size=n * H)
    usable = np.abs(d) > 1e-6
    return {
        "d": d, "r": r, "err_std": e,
        "dir_hits": int(np.sum(np.sign(d[usable]) == np.sign(r[usable]))),
        "dir_n": int(np.sum(usable)),
    }


def test_pool_and_selection_prefers_higher_corr_then_direction_then_rmse():
    per_station_configs = [
        # (ctx, readout, slope, err_scale)
        (96, READOUT_Q50, 0.2, 2.0),
        (384, READOUT_FULL_MEAN, 0.8, 1.0),
        (384, READOUT_Q50, 0.8, 1.5),
    ]
    rows = []
    for ctx, name, slope, err_scale in per_station_configs:
        per_station = {
            "s1": {"_arrays": _arrays(slope, seed=1)},
            "s2": {"_arrays": _arrays(slope, seed=2)},
        }
        # Make standalone RMSE differ deterministically.
        for st in per_station.values():
            st["_arrays"]["err_std"] *= err_scale
        pooled = pool_diagnostics(per_station, ("s1", "s2"))
        rows.append({"context_rows": ctx, "readout": name, **pooled})
    decision = select_stage_a(rows)
    assert decision["selected_context_rows"] == 384
    assert decision["selected_readout"] == READOUT_FULL_MEAN  # corr tie, lower RMSE
    assert len(decision["ranking"]) == 3


# --- physical contract --------------------------------------------------------

class _ClockBundle:
    capacity = 5.0

    def __init__(self, origins):
        base = np.datetime64("2026-01-01T00:00")
        end = np.datetime64("2026-01-10T00:00")
        rows = int((end - base) / np.timedelta64(15, "m")) + 1
        self.timestamps = base + np.arange(rows).astype("timedelta64[m]") * 15
        self.raw = np.ones((rows, 1), dtype=np.float32)
        self.feature_mu = np.array([0.0])
        self.feature_sd = np.array([1.0])
        self.starts = (
            ((np.asarray(origins) - base) / np.timedelta64(15, "m")).astype(int)
            - 95
        )


def test_contract_prior_is_horizon_wise_night_mask_and_capacity():
    # Origins 05:00 and 18:00: the 4h future crosses the 20:00/06:00 edges.
    bundle = _ClockBundle(
        [np.datetime64("2026-01-05T05:00"), np.datetime64("2026-01-05T18:00")]
    )
    prior = np.full((2, H), 9.0, dtype=np.float32)  # exceeds capacity
    out = contract_prior(prior, bundle, bundle.starts, 96, H)
    assert out.shape == (2, H)
    assert out.max() <= bundle.capacity + 1e-6
    assert (out >= 0).all()
    from formal.pc_fra import split_daylight_mask

    mask = split_daylight_mask(bundle, bundle.starts, 96, H)
    assert bool((out[mask == 0] == 0.0).all())
    assert bool((out[mask] > 0.0).all())
    # 05:00 origin -> future rows 05:15..; first three (05:15/30/45) are
    # masked night, 06:00 onward is day.
    assert mask[0, :3].sum() == 0 and mask[0, 3] == 1 and mask[0, -1] == 1
    # 18:00 origin -> daylight first half, masked after 20:00.
    assert mask[1, :8].all() and not mask[1, -1]


class _GeometricBundle:
    """Bundle with a SolarElevationMask-like column (index 1)."""

    capacity = 5.0
    daylight_index = 1

    def __init__(self, future_columns):
        base = np.datetime64("2026-01-01T00:00")
        end = np.datetime64("2026-01-10T00:00")
        rows = int((end - base) / np.timedelta64(15, "m")) + 1
        self.timestamps = base + np.arange(rows).astype("timedelta64[m]") * 15
        # col 0 = target, col 1 = geometric solar mask.
        self.raw = np.zeros((rows, 2), dtype=np.float32)
        self.feature_mu = np.array([0.0, 0.0])
        self.feature_sd = np.array([1.0, 1.0])
        origins = [np.datetime64("2026-01-05T05:00"),
                   np.datetime64("2026-01-05T12:00")]
        self.starts = (
            ((np.asarray(origins) - base) / np.timedelta64(15, "m")).astype(int)
            - 95
        )
        for i, col in enumerate(future_columns):
            target_start = int(self.starts[i]) + 96
            self.raw[target_start : target_start + H, 1] = col


def test_geometric_solar_mask_overrides_clock_heuristic():
    from formal.pc_fra import split_daylight_mask

    # Window 0: clock says night at 05:15-05:45, geometry says full daylight
    # (e.g. high-latitude summer dawn). Window 1: clock says noon-day,
    # geometry says night (e.g. polar/seasonal anomaly) - column wins.
    bundle = _GeometricBundle([np.ones(H, np.float32), np.zeros(H, np.float32)])
    mask = split_daylight_mask(bundle, bundle.starts, 96, H)
    assert mask.shape == (2, H)
    assert bool(mask[0].all())       # geometry keeps the early dawn window
    assert bool((~mask[1]).all())    # geometry zeroes the noon window
    prior = np.full((2, H), 3.0, dtype=np.float32)
    out = contract_prior(prior, bundle, bundle.starts, 96, H)
    assert np.allclose(out[0], 3.0) and np.allclose(out[1], 0.0)


def test_cache_path_tags_context_and_norm(tmp_path):
    dataset = tmp_path / "x.csv"
    dataset.write_text("timestamp,pv\n2026-01-01 00:00,1.0\n", encoding="utf-8")
    p96 = prior_cache_path_v4(dataset, H, 96, "raw")
    p384 = prior_cache_path_v4(dataset, H, 384, "raw")
    pcap = prior_cache_path_v4(dataset, H, 384, "capacity")
    assert p96.name != p384.name != pcap.name
    assert "ctx384" in p384.name and "capacity" in pcap.name and "v4" in p96.name
    with pytest.raises(ValueError):
        prior_cache_path_v4(dataset, H, 384, "weird")


# --- Strict L96 calibration: skew / scale / worst-site ---------------------

def test_phase_a_candidate_set_is_locked_l96():
    names = list(L96_PHASE_A_READOUTS) + [skew_readout_name(b) for b in (-0.5,-0.25,0.0,0.25,0.5)]
    assert len(names) == 11
    assert "q0.35" in names and "q0.45" in names
    assert skew_readout_name(0.0) == "skew_b+0.00"


def test_skew_readout_matches_median_at_beta_zero():
    grid = _grid_from_quantile_fn(lambda t: 2.0 + 3.0 * t)
    med = readout(grid, TAUS, READOUT_Q50)
    assert np.allclose(skew_readout(grid, TAUS, 0.0), med)


def test_skew_readout_zero_for_symmetric_grid_and_linear_in_beta():
    # Linear quantile function => q90+q10-2q50 == 0 exactly.
    grid = _grid_from_quantile_fn(lambda t: t)
    assert np.allclose(skew_readout(grid, TAUS, 0.5), 0.5)
    assert np.allclose(skew_readout(grid, TAUS, -0.5), 0.5)

    # Asymmetric convex grid: skew term s>0, output must be linear in beta.
    grid2 = _grid_from_quantile_fn(lambda t: t**2)
    q10 = readout_quantile(grid2, TAUS, 0.10)
    q50 = readout_quantile(grid2, TAUS, 0.50)
    q90 = readout_quantile(grid2, TAUS, 0.90)
    s = q90 + q10 - 2 * q50
    assert np.all(s > 0)
    for b in (-0.5, -0.25, 0.25, 0.5):
        assert np.allclose(skew_readout(grid2, TAUS, b), q50 + b * s)


def test_scale_disagreement_endpoints_and_convex_bounds():
    rng = np.random.default_rng(0)
    psrc = rng.uniform(0.0, 4.0, size=(6, H)).astype(np.float32)
    prior = np.clip(psrc + rng.normal(0, 1, (6, H)), 0, 5).astype(np.float32)
    assert np.allclose(scale_disagreement(prior, psrc, 1.0), prior)
    assert np.allclose(scale_disagreement(prior, psrc, 0.0), psrc)
    mix = scale_disagreement(prior, psrc, 0.25)
    assert np.allclose(mix, psrc + 0.25 * (prior - psrc))
    # gamma in [0,1] between two nonneg trajectories stays nonneg.
    assert (mix >= 0).all()


def test_segment_scaling_uses_four_groups():
    psrc = np.zeros((3, H), np.float32)
    prior = np.ones((3, H), np.float32)
    groups = (0.1, 0.2, 0.3, 0.4)
    out = scale_disagreement_segments(prior, psrc, groups)
    for (lo, hi), g in zip(HORIZON_SEGMENTS, groups):
        assert np.allclose(out[:, lo:hi], g)
    with pytest.raises(ValueError):
        scale_disagreement_segments(prior, psrc, (0.1, 0.2, 0.3))


def test_residual_diagnostics_reports_scale_ratio_kappa():
    # Varying R with D exactly 2x R in magnitude => kappa 2.
    rng = np.random.default_rng(1)
    target = rng.uniform(0.5, 3.0, size=(8, H)).astype(np.float32)
    psrc = np.zeros_like(target)
    prior = (2.0 * target).astype(np.float32)
    diag = residual_diagnostics(prior, psrc, target, target_sd=1.0)
    assert diag["scale_ratio_kappa"] == pytest.approx(2.0, abs=1e-6)


def test_worst_site_selection_ranks_by_weakest_station_not_pooled():
    def mk(name, rho_a, rho_b):
        return {"candidate": name, "per_station": {
            "s1": {"residual_corr": rho_a, "direction_accuracy": 0.7,
                   "scale_ratio_kappa": 1.0, "prior_rmse_physical": 1.0,
                   "prior_mbe_physical": 0.0},
            "s2": {"residual_corr": rho_b, "direction_accuracy": 0.7,
                   "scale_ratio_kappa": 1.0, "prior_rmse_physical": 1.0,
                   "prior_mbe_physical": 0.0},
        }}
    rows = [mk("high_pooled_but_negative", 0.90, -0.14),
            mk("balanced", 0.30, 0.28),
            mk("slightly_weaker_balanced", 0.29, 0.27)]
    out = select_worst_site(rows, ("s1", "s2"), top_k=2)
    assert out["selected"][0]["candidate"] == "balanced"
    assert out["selected"][1]["candidate"] == "slightly_weaker_balanced"
    # The high-pooled candidate with a negative station ranks last.
    assert out["ranking"][-1]["candidate"] == "high_pooled_but_negative"


# --- Correction shrinkage (run_correction_tune_h16) ------------------------

def test_correction_shrinkage_alpha_known_cases():
    from run_correction_tune_h16 import shrinkage_analysis

    rng = np.random.default_rng(7)
    psrc = np.zeros((20, H), np.float32)
    target = rng.normal(0, 1, (20, H)).astype(np.float32)

    def arr(pred):
        return {"psrc_std": psrc, "pred_b_std": pred.astype(np.float32),
                "target_std": target, "target_sd": np.float32(1.0),
                "target_mu": np.float32(0.0)}

    # Delta == residual -> alpha 1.
    assert shrinkage_analysis(arr(target))["alpha_star"] == pytest.approx(1.0)
    # Delta opposite to residual -> clipped to 0.
    assert shrinkage_analysis(arr(-target))["alpha_star"] == 0.0
    # Delta = 2 R -> alpha 0.5.
    assert shrinkage_analysis(arr(2.0 * target))["alpha_star"] == pytest.approx(0.5)
    # Degenerate zero correction -> 0, no crash.
    assert shrinkage_analysis(arr(psrc))["alpha_star"] == 0.0
