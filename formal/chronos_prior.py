"""Chronos-only prior generation for the H16 residual-correction study.

Nothing in this module touches PSRC, the residual adapter, its position,
the physical semantic tokens, the loss, or the optimizer. It only varies
HOW the frozen Chronos-2 forecast prior is generated and read out:

* context length          L_C in {96, 192, 384, 672} (PSRC stays L96);
* input contract          raw PV vs deterministic capacity normalization;
* distributional readout  median vs quantile-integrated (trimmed/full) mean.

Caches are recipe **v4** (independent files; v3 is never overwritten) and
store the native 21-quantile grid once, so every readout and the single-
quantile/segment diagnostics are offline numpy operations - Chronos runs
exactly once per (context, input_norm) pair.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from formal.fm_priors import model_revision
from formal.pc_fra import clip_prior, split_daylight_mask

# Native Chronos-2 quantile grid (pipeline.quantiles), ordered ascending.
CHRONOS2_QUANTILES: tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
)
RECIPE_V4 = "v4"
BACKEND = "chronos-2"
MODEL_NAME = "amazon/chronos-2"

# Registered Stage A search space: 4 contexts x 12 readouts = 48 configs.
STAGE_A_CONTEXTS: tuple[int, ...] = (96, 192, 384, 672)
READOUT_Q50 = "q50"
READOUT_TRIMMED = "trimmed_mean_q20_q80"
READOUT_FULL_MEAN = "mean_q_full"
TRIM_LOW, TRIM_HIGH = 0.20, 0.80

# Single native-grid quantiles q0.10..q0.90 ARE formal Stage A candidates
# (offline reads of the cached grid, no extra Chronos inference), named
# "q0.10" ... "q0.90" to distinguish them from the median alias "q50".
SINGLE_Q_GRID: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(1, 10))
SINGLE_Q_READOUTS: tuple[str, ...] = tuple(f"q{q:.2f}" for q in SINGLE_Q_GRID)
STAGE_A_READOUTS: tuple[str, ...] = (
    SINGLE_Q_READOUTS + (READOUT_Q50, READOUT_TRIMMED, READOUT_FULL_MEAN)
)

# Diagnostics-only grids (never Stage A selection inputs beyond the 9
# candidates above): four fixed horizon segments.
SEGMENT_Q_GRID: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)
HORIZON_SEGMENTS: tuple[tuple[int, int], ...] = ((0, 4), (4, 8), (8, 12), (12, 16))

INPUT_NORMS: tuple[str, ...] = ("raw", "capacity")

# Strict L96-H16 distribution-shape calibration (context never changes).
# P* = q50 + beta*(q90 + q10 - 2*q50); beta=0 is exactly the q50 readout.
SKEW_BETAS: tuple[float, ...] = (-0.50, -0.25, 0.00, 0.25, 0.50)
# Phase A readouts fixed by the strict-L96 protocol.
L96_PHASE_A_READOUTS: tuple[str, ...] = (
    "q0.35", "q0.40", "q0.45",
    READOUT_Q50, READOUT_TRIMMED, READOUT_FULL_MEAN,
)

# Disagreement-scale calibration: P' = y_psrc + gamma*(P* - y_psrc).
SCALE_GAMMAS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
# Two fixed monotonic 4-segment schedules (H1-4 ... H13-16); near horizons
# trust PSRC more, far horizons let more of the foundation prior through.
SEGMENT_GAMMA_SCHEDULES: dict[str, tuple[float, ...]] = {
    "seg_0.10_0.20_0.30_0.40": (0.10, 0.20, 0.30, 0.40),
    "seg_0.20_0.30_0.40_0.50": (0.20, 0.30, 0.40, 0.50),
}

DIRECTION_EPS = 1e-6


# --- Cache paths ------------------------------------------------------------

def prior_cache_path_v4(
    dataset_path: str | Path,
    horizon: int,
    context_rows: int,
    input_norm: str,
) -> Path:
    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm must be one of {INPUT_NORMS}")
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:8]
    return (
        Path("data/fm_prior_cache")
        / f"{digest}__{BACKEND}__h{horizon}__ctx{int(context_rows)}"
        f"__{input_norm}__prior_{RECIPE_V4}.npz"
    )


def valid_starts(starts: np.ndarray, seq_len: int, context_rows: int) -> np.ndarray:
    """Windows whose full Chronos context lies inside the dataset rows."""
    starts = np.asarray(starts, dtype=np.int64)
    return starts[starts + int(seq_len) - int(context_rows) >= 0]


# --- Chronos-2 grid cache generation ---------------------------------------

def build_chronos2_grid_cache(
    bundle,
    starts: np.ndarray,
    horizon: int,
    context_rows: int,
    input_norm: str,
    *,
    seq_len: int = 96,
    batch_size: int = 128,
    force: bool = False,
    out_path: str | Path | None = None,
) -> Path:
    """Materialise one v4 cache: native 21-quantile grid per window.

    Context ends at the same forecast origin as PSRC's L96 history
    (``power[s+96-L_C : s+96]``). The capacity contract divides the input
    by the deterministic training capacity only; Chronos' internal
    scaling is untouched and outputs are multiplied back exactly.

    ``out_path`` writes a sidecar cache (same format) instead of the
    registered digest-keyed location; used for one-off split caches so
    development caches are never overwritten.
    """
    import torch
    from chronos import Chronos2Pipeline

    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm must be one of {INPUT_NORMS}")
    context_rows = int(context_rows)
    horizon = int(horizon)
    path = prior_cache_path_v4(bundle.path, horizon, context_rows, input_norm)
    if out_path is not None:
        path = Path(out_path)
    if path.is_file() and not force:
        return path

    requested = np.asarray(starts, dtype=np.int64)
    kept = valid_starts(requested, seq_len, context_rows)
    dropped = int(len(requested) - len(kept))
    power = np.nan_to_num(np.asarray(bundle.raw[:, -1]), nan=0.0).astype(np.float32)
    power = np.clip(power, 0.0, None)
    if input_norm == "capacity":
        scale = float(bundle.capacity)
        feed_power = power / scale
    else:
        scale = 1.0
        feed_power = power

    def window(start: int) -> np.ndarray:
        lo = int(start) + seq_len - context_rows
        return feed_power[lo : int(start) + seq_len]

    contexts = torch.from_numpy(np.stack([window(s) for s in kept])).unsqueeze(1)
    total = contexts.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_NAME, device_map=device)
    model_quantiles = [float(q) for q in pipeline.quantiles]
    if tuple(model_quantiles) != CHRONOS2_QUANTILES:
        raise ValueError(
            f"unexpected Chronos-2 quantile grid: {model_quantiles}"
        )

    grid = np.empty((total, len(CHRONOS2_QUANTILES), horizon), dtype=np.float16)
    started = time.perf_counter()
    for offset in range(0, total, batch_size):
        chunk = contexts[offset : offset + batch_size]
        with torch.inference_mode():
            outputs = pipeline.predict(
                chunk,
                prediction_length=horizon,
                batch_size=int(chunk.shape[0]),
            )
        block = torch.stack([item[0] for item in outputs], dim=0)
        block = block.detach().float().cpu().numpy().astype(np.float32)
        # Independent heads can cross near zero: enforce a monotone grid
        # along the quantile axis before any readout.
        block.sort(axis=1)
        if input_norm == "capacity":
            # Exact inverse of the deterministic input division, then clip.
            block *= scale
        block = np.maximum(block, 0.0)
        width = block.shape[0]
        grid[offset : offset + width] = block.astype(np.float16)
        done = offset + width
        if (offset // batch_size) % 20 == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"[chronos2:v4] ctx={context_rows} norm={input_norm} "
                f"{done}/{total} ({done/total:.0%}) {rate:.0f} win/s "
                f"eta {eta/60:.1f} min",
                flush=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        starts=kept,
        q_grid=grid,
        taus=np.array(CHRONOS2_QUANTILES, dtype=np.float64),
        dim=np.int64(horizon),
        kind=np.array("chronos2_quantile_grid"),
        backend=np.array(BACKEND),
        recipe=np.array(RECIPE_V4),
        model=np.array(MODEL_NAME),
        model_revision=np.array(model_revision(MODEL_NAME) or ""),
        dataset_sha256=np.array(
            hashlib.sha256(Path(bundle.path).read_bytes()).hexdigest()
        ),
        input_contract=np.array("target_pv_history_only"),
        input_norm=np.array(input_norm),
        capacity_scale=np.float64(scale),
        context_rows=np.int64(context_rows),
        seq_rows=np.int64(seq_len),
        n_requested_windows=np.int64(len(requested)),
        n_short_context_dropped=np.int64(dropped),
        num_samples=np.int64(0),
        sampling_seed=np.int64(-1),
        clip_negative_to_zero=np.array(True),
        quantile_ordering=np.array("sort21_monotone"),
    )
    return path


def load_grid_cache(path: str | Path):
    """Read a v4 cache; returns (starts, q_grid[N,Q,H] float32, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing v4 chronos grid cache: {path}")
    with np.load(path) as archive:
        if str(archive["recipe"]) != RECIPE_V4:
            raise ValueError(f"{path} is not a v4 cache")
        taus = np.asarray(archive["taus"], dtype=np.float64)
        if tuple(float(t) for t in taus) != CHRONOS2_QUANTILES:
            raise ValueError(f"{path} quantile grid mismatch")
        starts = np.asarray(archive["starts"], dtype=np.int64)
        q_grid = np.asarray(archive["q_grid"], dtype=np.float32)
        meta = {
            "cache_path": str(path),
            "context_rows": int(archive["context_rows"]),
            "input_norm": str(archive["input_norm"]),
            "capacity_scale": float(archive["capacity_scale"]),
            "model_revision": str(archive["model_revision"]),
            "quantile_ordering": str(archive["quantile_ordering"]),
            "n_windows": int(len(starts)),
            "n_short_context_dropped": int(archive["n_short_context_dropped"]),
        }
    return starts, q_grid, meta


# --- Distributional readouts (pure numpy) -----------------------------------

def _trapz_mean(values: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Trapezoid integral of Q(tau) divided by the covered tau width."""
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trapz(values, taus, axis=1) / float(taus[-1] - taus[0])


def readout(q_grid: np.ndarray, taus: np.ndarray, name: str) -> np.ndarray:
    """Return one physical-unit prior trajectory [N, H]."""
    if name == READOUT_Q50:
        idx = tuple(float(t) for t in taus).index(0.50)
        return q_grid[:, idx, :].copy()
    if name == READOUT_FULL_MEAN:
        # Mean over the full native grid [0.01, 0.99] (width normalised).
        return _trapz_mean(q_grid, taus)
    if name == READOUT_TRIMMED:
        keep = (taus >= TRIM_LOW - 1e-12) & (taus <= TRIM_HIGH + 1e-12)
        sub_q = q_grid[:, keep, :]
        sub_t = taus[keep]
        return _trapz_mean(sub_q, sub_t)
    # Any "q0.35"-style name resolves to that native-grid quantile; the grid
    # membership check inside readout_quantile rejects off-grid levels.
    if name.startswith("q"):
        return readout_quantile(q_grid, taus, float(name[1:]))
    raise ValueError(f"unknown readout {name!r}")


def readout_quantile(q_grid: np.ndarray, taus: np.ndarray, q: float) -> np.ndarray:
    levels = [float(t) for t in taus]
    if float(q) not in levels:
        raise ValueError(f"quantile {q} is not on the native grid")
    return q_grid[:, levels.index(float(q)), :].copy()


def skew_readout_name(beta: float) -> str:
    return f"skew_b{beta:+.2f}"


def skew_readout(q_grid: np.ndarray, taus: np.ndarray, beta: float) -> np.ndarray:
    """Distribution-shape trajectory  P* = q50 + beta*(q90 + q10 - 2*q50).

    ``beta`` re-weights the asymmetry of Chronos' predictive distribution
    without learning any parameter from the target series; beta=0 returns
    the native median exactly. Output stays a single [N, H] physical-unit
    trajectory, so the downstream adapter contract is unchanged.
    """
    q10 = readout_quantile(q_grid, taus, 0.10)
    q50 = readout_quantile(q_grid, taus, 0.50)
    q90 = readout_quantile(q_grid, taus, 0.90)
    return (q50 + float(beta) * (q90 + q10 - 2.0 * q50)).astype(np.float32)


def scale_disagreement(
    prior_phys: np.ndarray, psrc_phys: np.ndarray, gamma: float
) -> np.ndarray:
    """P' = y_psrc + gamma*(P* - y_psrc), one scalar strength (L96 only)."""
    return (psrc_phys + float(gamma) * (prior_phys - psrc_phys)).astype(np.float32)


def scale_disagreement_segments(
    prior_phys: np.ndarray,
    psrc_phys: np.ndarray,
    gamma_groups: tuple[float, ...],
) -> np.ndarray:
    """Same calibration with one gamma per fixed HORIZON_SEGMENTS group."""
    if len(gamma_groups) != len(HORIZON_SEGMENTS):
        raise ValueError("one gamma per horizon segment required")
    out = np.array(psrc_phys, dtype=np.float32, copy=True)
    for gamma, (lo, hi) in zip(gamma_groups, HORIZON_SEGMENTS):
        out[:, lo:hi] = psrc_phys[:, lo:hi] + float(gamma) * (
            prior_phys[:, lo:hi] - psrc_phys[:, lo:hi]
        )
    return out


def contract_prior(
    prior_phys: np.ndarray,
    bundle,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
) -> np.ndarray:
    """Apply the registered PV contract: horizon-wise mask and clipping."""
    mask = split_daylight_mask(bundle, np.asarray(starts), seq_len, horizon)
    out = np.empty_like(prior_phys, dtype=np.float32)
    for i in range(len(starts)):
        out[i] = clip_prior(prior_phys[i], mask[i].astype(np.float32), bundle.capacity)
    return out


# --- Residual information diagnostics ---------------------------------------

def residual_diagnostics(
    prior_phys: np.ndarray,
    psrc_phys: np.ndarray,
    target_phys: np.ndarray,
    target_sd: float,
) -> dict[str, object]:
    """Correlation / OLS coefficient / direction accuracy against residual.

    D = (P_chronos - y_psrc)/sd and R = (y - y_psrc)/sd are the standardized
    disagreement and the true remaining PSRC residual. All information
    metrics live in that standardized residual space.
    """
    d = ((prior_phys - psrc_phys) / float(target_sd)).reshape(-1)
    r = ((target_phys - psrc_phys) / float(target_sd)).reshape(-1)
    d_mean, r_mean = d.mean(), r.mean()
    dc, rc = d - d_mean, r - r_mean
    denom = float(np.sqrt(np.sum(dc**2) * np.sum(rc**2)))
    corr = float(np.sum(dc * rc) / denom) if denom > 0 else float("nan")
    dd = float(np.sum(d * d))
    beta = float(np.sum(d * r) / dd) if dd > 0 else float("nan")
    # R_new = R - beta*D; fraction of residual SSE removed = corr^2.
    r_sse = float(np.sum(r**2))
    sse_reduction = float(np.clip(corr**2, 0.0, 1.0))
    rmse0 = float(np.sqrt(np.mean(r**2)))
    rmse_after = float(np.sqrt(max(r_sse - dd * beta**2, 0.0) / r.size))
    usable = np.abs(d) > DIRECTION_EPS
    if np.any(usable):
        direction = float(np.mean(np.sign(d[usable]) == np.sign(r[usable])))
        n_dir = int(np.sum(usable))
    else:
        direction, n_dir = float("nan"), 0
    err = prior_phys.reshape(-1) - target_phys.reshape(-1)
    scale_ratio = float(np.std(d) / np.std(r)) if np.std(r) > 0 else float("nan")
    return {
        "residual_corr": corr,
        "ols_beta": beta,
        "residual_sse_reduction": sse_reduction,
        "psrc_residual_rmse_std": rmse0,
        "oracle_projected_residual_rmse_std": rmse_after,
        "direction_accuracy": direction,
        "direction_n_rows": n_dir,
        "scale_ratio_kappa": scale_ratio,
        "prior_rmse_physical": float(np.sqrt(np.mean(err**2))),
        "prior_mae_physical": float(np.mean(np.abs(err))),
        "prior_mbe_physical": float(np.mean(err)),
        "disagreement_mean_std": float(d_mean),
        "residual_mean_std": float(r_mean),
    }


def per_horizon_corr(
    prior_phys: np.ndarray,
    psrc_phys: np.ndarray,
    target_phys: np.ndarray,
    target_sd: float,
) -> list[float]:
    h = prior_phys.shape[1]
    out = []
    for j in range(h):
        d = (prior_phys[:, j] - psrc_phys[:, j]) / target_sd
        r = (target_phys[:, j] - psrc_phys[:, j]) / target_sd
        dc, rc = d - d.mean(), r - r.mean()
        denom = float(np.sqrt(np.sum(dc**2) * np.sum(rc**2)))
        out.append(float(np.sum(dc * rc) / denom) if denom > 0 else float("nan"))
    return out


def single_quantile_table(
    q_grid: np.ndarray,
    taus: np.ndarray,
    contract_fn,
    psrc_phys: np.ndarray,
    target_phys: np.ndarray,
    target_sd: float,
) -> dict[str, dict[str, float]]:
    """Diagnostic-only: residual corr for q in {0.1,...,0.9}."""
    table: dict[str, dict[str, float]] = {}
    for q in SINGLE_Q_GRID:
        prior = contract_fn(readout_quantile(q_grid, taus, q))
        diag = residual_diagnostics(prior, psrc_phys, target_phys, target_sd)
        table[f"q{int(round(q * 100)):02d}"] = {
            "residual_corr": diag["residual_corr"],
            "ols_beta": diag["ols_beta"],
            "direction_accuracy": diag["direction_accuracy"],
            "prior_rmse_physical": diag["prior_rmse_physical"],
        }
    return table


def segment_table(
    q_grid: np.ndarray,
    taus: np.ndarray,
    contract_fn,
    psrc_phys: np.ndarray,
    target_phys: np.ndarray,
    target_sd: float,
) -> list[dict[str, object]]:
    """Diagnostic-only: best-q search per four fixed horizon segments."""
    rows = []
    for k, (lo, hi) in enumerate(HORIZON_SEGMENTS):
        best = None
        per_q = {}
        for q in SEGMENT_Q_GRID:
            prior = contract_fn(readout_quantile(q_grid, taus, q))
            d = ((prior - psrc_phys) / target_sd)[:, lo:hi].reshape(-1)
            r = ((target_phys - psrc_phys) / target_sd)[:, lo:hi].reshape(-1)
            dc, rc = d - d.mean(), r - r.mean()
            denom = float(np.sqrt(np.sum(dc**2) * np.sum(rc**2)))
            corr = float(np.sum(dc * rc) / denom) if denom > 0 else float("nan")
            per_q[f"q{int(round(q * 100)):02d}"] = corr
            if best is None or corr > best[1]:
                best = (q, corr)
        rows.append(
            {
                "segment": f"H{lo + 1}-H{hi}",
                "best_q": float(best[0]),
                "best_residual_corr": float(best[1]),
                "corr_per_q": per_q,
            }
        )
    return rows


def pool_diagnostics(
    per_station: dict[str, dict[str, object]],
    stations: tuple[str, ...],
) -> dict[str, object]:
    """Pool residual information across both dev stations.

    Stations differ in capacity (4.3 vs 15.2 kW), so every pooled error is
    standardized by the station's target SD before pooling; per-station
    physical metrics are reported separately by the caller.
    """
    ds, rs, es = [], [], []
    dir_hits, dir_n = 0, 0
    for station in stations:
        a = per_station[station]["_arrays"]
        ds.append(a["d"])
        rs.append(a["r"])
        es.append(a["err_std"])
        dir_hits += a["dir_hits"]
        dir_n += a["dir_n"]
    d = np.concatenate(ds)
    r = np.concatenate(rs)
    e = np.concatenate(es)
    dc, rc = d - d.mean(), r - r.mean()
    denom = float(np.sqrt(np.sum(dc**2) * np.sum(rc**2)))
    corr = float(np.sum(dc * rc) / denom) if denom > 0 else float("nan")
    dd = float(np.sum(d * d))
    beta = float(np.sum(d * r) / dd) if dd > 0 else float("nan")
    r_sse = float(np.sum(r**2))
    return {
        "pooled_residual_corr": corr,
        "pooled_ols_beta": beta,
        "pooled_residual_sse_reduction": float(np.clip(corr**2, 0, 1)),
        "pooled_oracle_residual_rmse_std": float(
            np.sqrt(max(r_sse - dd * beta**2, 0.0) / r.size)
        ),
        "pooled_direction_accuracy": float(dir_hits / max(dir_n, 1)),
        "pooled_prior_rmse_std": float(np.sqrt(np.mean(e**2))),
        "pooled_prior_mae_std": float(np.mean(np.abs(e))),
        "pooled_prior_mbe_std": float(np.mean(e)),
    }


def select_stage_a(rows: list[dict[str, object]]) -> dict[str, object]:
    """PRIMARY pooled residual corr; SECONDARY direction; TIEBREAK RMSE."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["pooled_residual_corr"]),
            -float(row["pooled_direction_accuracy"]),
            float(row["pooled_prior_rmse_std"]),
        ),
    )
    best = ordered[0]
    return {
        "selection_rule": (
            "PRIMARY=max pooled residual corr; SECONDARY=max direction "
            "accuracy; TIEBREAK=min standalone prior RMSE"
        ),
        "selected_context_rows": int(best["context_rows"]),
        "selected_readout": str(best["readout"]),
        "selected_metrics": {
            k: v for k, v in best.items()
            if k in (
                "pooled_residual_corr", "pooled_direction_accuracy",
                "pooled_prior_rmse_std", "pooled_ols_beta",
            )
        },
        "ranking": [
            {
                "context_rows": int(row["context_rows"]),
                "readout": str(row["readout"]),
                "pooled_residual_corr": float(row["pooled_residual_corr"]),
                "pooled_direction_accuracy": float(
                    row["pooled_direction_accuracy"]
                ),
                "pooled_prior_rmse_std": float(
                    row["pooled_prior_rmse_std"]
                ),
            }
            for row in ordered
        ],
    }


def worst_site_table(
    rows: list[dict[str, object]],
    stations: tuple[str, ...],
) -> list[dict[str, object]]:
    """Rank rows by the WORST station (max-min), never by pooled score.

    Each row must carry ``per_station`` -> station -> residual metrics.
    Key: max over stations of -rho, then -direction, then |kappa-1|.
    Ties after that resolve on mean (then min) standalone prior RMSE.
    """
    enriched = []
    for row in rows:
        per = row["per_station"]
        corrs = {s: float(per[s]["residual_corr"]) for s in stations}
        dirs = {s: float(per[s]["direction_accuracy"]) for s in stations}
        kappas = {s: float(per[s]["scale_ratio_kappa"]) for s in stations}
        rmses = {s: float(per[s]["prior_rmse_physical"]) for s in stations}
        enriched.append({
            **row,
            "_worst_corr": min(corrs.values()),
            "_worst_direction": min(dirs.values()),
            "_worst_kappa_gap": max(abs(k - 1.0) for k in kappas.values()),
            "_mean_rmse": float(np.mean(list(rmses.values()))),
        })
    enriched.sort(
        key=lambda r: (
            -r["_worst_corr"],
            -r["_worst_direction"],
            r["_worst_kappa_gap"],
            r["_mean_rmse"],
        )
    )
    return enriched


def select_worst_site(
    rows: list[dict[str, object]],
    stations: tuple[str, ...],
    top_k: int = 2,
) -> dict[str, object]:
    """Pick top-k candidates whose weakest station is strongest."""
    ordered = worst_site_table(rows, stations)
    keep = []
    for row in ordered[:top_k]:
        keep.append({
            "candidate": str(row["candidate"]),
            "worst_station_residual_corr": row["_worst_corr"],
            "worst_station_direction_accuracy": row["_worst_direction"],
            "worst_station_kappa_gap": row["_worst_kappa_gap"],
            "per_station": {
                s: {
                    "residual_corr": float(
                        row["per_station"][s]["residual_corr"]
                    ),
                    "direction_accuracy": float(
                        row["per_station"][s]["direction_accuracy"]
                    ),
                    "scale_ratio_kappa": float(
                        row["per_station"][s]["scale_ratio_kappa"]
                    ),
                    "prior_mbe_physical": float(
                        row["per_station"][s]["prior_mbe_physical"]
                    ),
                }
                for s in stations
            },
        })
    return {
        "selection_rule": (
            "PRIMARY=max over candidates of min over stations corr(D,R); "
            "SECONDARY=max min-station direction accuracy; "
            "TIEBREAK=min max-station |kappa-1|"
        ),
        "stations": list(stations),
        "selected": keep,
        "ranking": [
            {
                "candidate": str(r["candidate"]),
                "worst_station_residual_corr": r["_worst_corr"],
                "worst_station_direction_accuracy": r["_worst_direction"],
                "worst_station_kappa_gap": r["_worst_kappa_gap"],
            }
            for r in ordered
        ],
    }
