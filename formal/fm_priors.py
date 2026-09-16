"""Offline frozen time-series foundation-model *forecast priors*.

A frozen Chronos model produces, once, the median future-H trajectory for
every legal forecast window from the **target PV history alone**::

    P_{t+1:t+H} = quantile_0.5(F_Chronos(X_{t-L+1:t}^{PV}))

Trajectories are cached per window in physical units. The registered
Chronos calibration converts these caches into the compact PC-FRA packs;
the training process never loads Chronos.

Causality is strict: row range ``[start, start + seq_len)`` only - the
forecast horizon is never read. Sampling is seeded per batch so the cache
is reproducible. The only physical constraint applied in this first round
is the PV lower bound (negative generations are clipped to zero).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

RECIPE_VERSION = "v1"
# v2 caches add lower/upper sample quantiles for uncertainty-aware analysis.
RECIPE_VERSION_QUANTILES = "v2"
# v3 caches come from Chronos-2, a native deterministic quantile model
# (no sampling); q010/q050/q090 are emitted directly by the forecaster.
RECIPE_VERSION_CHRONOS2 = "v3"
CACHE_DIR = Path("data/fm_prior_cache")

# model_key -> backend configuration. ``native_quantiles`` marks forecasters
# that emit quantiles directly (Chronos-2) instead of Monte Carlo samples.
PRIOR_BACKENDS: dict[str, dict[str, object]] = {
    "chronos-t5-small": {
        "model": "amazon/chronos-t5-small",
        "num_samples": 21,
        "native_quantiles": False,
    },
    "chronos-2": {
        "model": "amazon/chronos-2",
        "native_quantiles": True,
        "recipe": RECIPE_VERSION_CHRONOS2,
        # Quantiles consumed by the late residual adapter; Chronos-2 emits
        # a denser grid natively, these are the three cached levels.
        "quantiles": (0.1, 0.5, 0.9),
    },
}
DEFAULT_PRIOR_BACKEND = "chronos-t5-small"

# Chronos-2 is deterministic: there is no sampling procedure. Record the
# sentinels explicitly so the cache contract is unambiguous.
CHRONOS2_NUM_SAMPLES = 0
CHRONOS2_SAMPLING_SEED = -1


def prior_backend_config(backend: str) -> dict[str, object]:
    try:
        return dict(PRIOR_BACKENDS[backend])
    except KeyError as exc:
        raise ValueError(
            f"unknown prior backend {backend!r}; choose from {sorted(PRIOR_BACKENDS)}"
        ) from exc


def all_window_starts(bundle) -> np.ndarray:
    """All legal train/val/test windows in ascending start-row order."""
    return np.unique(
        np.concatenate(
            [bundle.train.starts, bundle.validation.starts, bundle.test.starts]
        )
    )


def prior_cache_path(
    dataset_path: str | Path, backend: str, horizon: int,
    recipe: str = RECIPE_VERSION,
) -> Path:
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:8]
    return CACHE_DIR / f"{digest}__{backend}__h{horizon}__prior_{recipe}.npz"


def model_revision(model_name: str) -> str | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def build_prior_cache(
    bundle,
    *,
    backend: str = DEFAULT_PRIOR_BACKEND,
    horizon: int | None = None,
    batch_size: int = 64,
    num_samples: int | None = None,
    quantiles: tuple[float, ...] = (0.5,),
    seed: int = 2026,
    force: bool = False,
) -> Path:
    """Freeze-encode prior trajectories for every window (offline).

    ``quantiles=(0.5,)`` produces the v1 median cache. Adding more sample
    quantiles (e.g. ``(0.1, 0.5, 0.9)``) produces a v2 cache that also
    stores ``q010`` / ``q090`` for uncertainty-aware analysis; the median
    is always stored as ``prior`` so the training-side loader is unchanged.
    """
    import torch
    from chronos import ChronosPipeline

    cfg = prior_backend_config(backend)
    if cfg.get("native_quantiles", False):
        # Native quantile models emit a fixed grid; the cached levels are
        # fixed per recipe (v3 = q10/q50/q90), ignoring CLI --quantiles.
        return _build_native_quantile_cache(
            bundle,
            backend=backend,
            horizon=horizon or int(bundle.train.horizon),
            batch_size=batch_size,
            force=force,
        )
    model_name = str(cfg["model"])
    if num_samples is None:
        num_samples = int(cfg["num_samples"])
    horizon = int(bundle.train.horizon if horizon is None else horizon)
    qs = tuple(float(q) for q in quantiles)
    if 0.5 not in qs:
        raise ValueError("quantiles must include the median 0.5")
    recipe = RECIPE_VERSION if qs == (0.5,) else RECIPE_VERSION_QUANTILES
    path = prior_cache_path(bundle.path, backend, horizon, recipe=recipe)
    if path.is_file() and not force:
        return path

    starts = all_window_starts(bundle).astype(np.int64)
    # Target column only, physical units; NaNs and negatives are not valid
    # PV context (strictly history rows).
    power = np.nan_to_num(np.asarray(bundle.raw[:, -1]), nan=0.0).astype(np.float32)
    contexts = torch.from_numpy(
        np.stack([np.clip(power[s : s + seq_len], 0.0, None) for s in starts])
    )
    total = contexts.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(model_name, device_map=device)

    grids = {q: np.empty((total, horizon), dtype=np.float16) for q in qs}
    started = time.perf_counter()
    for index, offset in enumerate(range(0, total, batch_size)):
        chunk = contexts[offset : offset + batch_size]
        # Deterministic per-batch sampling seed: the cache reproduces.
        torch.manual_seed(seed + index)
        with torch.inference_mode():
            samples = pipeline.predict(
                chunk,
                prediction_length=horizon,
                num_samples=num_samples,
            )
        for q in qs:
            values = samples.quantile(q, dim=1).float().cpu().numpy()
            grids[q][offset : offset + chunk.shape[0]] = np.maximum(
                values, 0.0
            ).astype(np.float16)
        done = min(offset + batch_size, total)
        if index % 50 == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"[prior:{backend}] {done}/{total} ({done/total:.0%}) "
                f"{rate:.1f} win/s eta {eta/60:.1f} min",
                flush=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        starts=starts,
        prior=grids[0.5],
        dim=np.int64(horizon),
        kind=np.array("forecast_prior"),
        backend=np.array(backend),
        recipe=np.array(recipe),
        quantiles=np.array(qs, dtype=np.float64),
        model=np.array(model_name),
        num_samples=np.int64(num_samples),
        model_revision=np.array(model_revision(model_name) or ""),
        dataset_sha256=np.array(
            hashlib.sha256(Path(bundle.path).read_bytes()).hexdigest()
        ),
    )
    for q in qs:
        if q == 0.5:
            continue
        # percentile-named fields: 0.1 -> q010, 0.9 -> q090
        payload[f"q{int(round(q * 100)):03d}"] = grids[q]
    np.savez_compressed(path, **payload)
    return path


def _build_native_quantile_cache(
    bundle,
    *,
    backend: str,
    horizon: int,
    batch_size: int,
    force: bool,
) -> Path:
    """Freeze Chronos-2 native q10/q50/q90 trajectories (recipe v3).

    Chronos-2 is a deterministic quantile forecaster: there is no sampling
    loop or sampling seed. The input contract is identical to the v1/v2
    caches - target PV history rows ``[start, start + seq_len)`` only,
    physical units, negatives clipped to zero.
    """
    import torch
    from chronos import Chronos2Pipeline

    cfg = prior_backend_config(backend)
    model_name = str(cfg["model"])
    qs = tuple(float(q) for q in cfg["quantiles"])
    if qs != (0.1, 0.5, 0.9):
        raise ValueError("v3 chronos-2 cache requires quantiles (0.1, 0.5, 0.9)")
    horizon = int(horizon)
    seq_len = int(bundle.train.seq_len)
    recipe = str(cfg["recipe"])
    path = prior_cache_path(bundle.path, backend, horizon, recipe=recipe)
    if path.is_file() and not force:
        return path

    starts = all_window_starts(bundle).astype(np.int64)
    # Target column only, physical units, strictly historical rows.
    power = np.nan_to_num(np.asarray(bundle.raw[:, -1]), nan=0.0).astype(np.float32)
    contexts = torch.from_numpy(
        np.stack([np.clip(power[s : s + seq_len], 0.0, None) for s in starts])
    ).unsqueeze(1)  # [N, 1, L]
    total = contexts.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)
    model_quantiles = [float(q) for q in pipeline.quantiles]
    missing = [q for q in qs if q not in model_quantiles]
    if missing:
        raise ValueError(f"{model_name} does not emit quantiles {missing}")
    q_index = {q: model_quantiles.index(q) for q in qs}

    grids = {q: np.empty((total, horizon), dtype=np.float16) for q in qs}
    started = time.perf_counter()
    for offset in range(0, total, batch_size):
        chunk = contexts[offset : offset + batch_size]
        with torch.inference_mode():
            outputs = pipeline.predict(
                chunk,
                prediction_length=horizon,
                batch_size=int(chunk.shape[0]),
            )
        # Each output is [1, n_quantiles, horizon]; stack to [bs, n_q, H].
        quantiled = torch.stack([item[0] for item in outputs], dim=0)
        block = quantiled.detach().float().cpu().numpy()
        done_width = block.shape[0]
        # Chronos-2 emits independent quantile heads, which can cross at
        # near-zero generation. Enforce q10 <= q50 <= q90 by sorting the
        # three levels (minimum change: only crossing cells are touched).
        selected = block[:, [q_index[q] for q in qs], :]
        selected.sort(axis=1)
        for column, q in enumerate(qs):
            values = selected[:, column, :]
            grids[q][offset : offset + done_width] = np.maximum(values, 0.0).astype(
                np.float16
            )
        done = offset + done_width
        if (offset // batch_size) % 20 == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"[prior:{backend}] {done}/{total} ({done/total:.0%}) "
                f"{rate:.1f} win/s eta {eta/60:.1f} min",
                flush=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        starts=starts,
        # ``prior`` stays the median so generic loaders keep working.
        prior=grids[0.5],
        q050=grids[0.5],
        q010=grids[0.1],
        q090=grids[0.9],
        dim=np.int64(horizon),
        kind=np.array("forecast_prior"),
        backend=np.array(backend),
        recipe=np.array(recipe),
        quantiles=np.array(qs, dtype=np.float64),
        model_quantiles=np.array(model_quantiles, dtype=np.float64),
        model=np.array(model_name),
        # Native quantile head: no Monte Carlo samples and no RNG.
        num_samples=np.int64(CHRONOS2_NUM_SAMPLES),
        sampling_seed=np.int64(CHRONOS2_SAMPLING_SEED),
        output_type=np.array("native_quantiles"),
        model_revision=np.array(model_revision(model_name) or ""),
        dataset_sha256=np.array(
            hashlib.sha256(Path(bundle.path).read_bytes()).hexdigest()
        ),
        # Explicit causality contract.
        input_contract=np.array("target_pv_history_only"),
        context_rows=np.int64(seq_len),
        clip_negative_to_zero=np.array(True),
        # Independent quantile heads cross near zero; the three cached
        # levels are order-enforced by per-cell sorting before clipping.
        quantile_ordering=np.array("sort3_monotone"),
    )
    np.savez_compressed(path, **payload)
    return path
