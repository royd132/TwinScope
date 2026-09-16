"""Offline feature contract for the PC-FRA H16 campaign.

Everything in this module is built from information available at the
forecast origin (plus the frozen Chronos-2 q0.5 cache and the frozen
PSRC predictions). No future target, future weather observation, or
quality flag is ever used. Continuous scalars are normalised with
training-split statistics only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from layers.pc_fra import pack_slices, pack_width

# Registered, non-tuned operating rules.
RECENT_STEPS = 4  # 15 min x 4 = last hour of observed PV
NIGHT_LOW_RATIO = 0.20
PEAK_RATIO = 0.80
RAMP_RATIO = 0.10
# Rule application order (index into the state one-hot).
STATE_NAMES = ("night_low", "ramp", "peak", "regular")
NIGHT_LOW, RAMP, PEAK, REGULAR = 0, 1, 2, 3
# epsilon guard: 90th-percentile training residual, capped at 20% of the
# station capacity after physical conversion.
EPSILON_QUANTILE = 0.90
EPSILON_CAPACITY_FRACTION = 0.20
# Negative diagnostic: shuffling is judged "content-independent" when the
# shuffled arm is no more than this much worse than arm C on validation.
DIAGNOSTIC_LIKE_TOLERANCE = 0.01


def slices(horizon: int) -> dict[str, slice | int]:
    return pack_slices(horizon)


def operating_state(level_ratio: float, ramp_ratio: float) -> int:
    """Registered order: night/low, then ramp, then peak, else regular."""
    if level_ratio < NIGHT_LOW_RATIO:
        return NIGHT_LOW
    if ramp_ratio > RAMP_RATIO:
        return RAMP
    if level_ratio > PEAK_RATIO:
        return PEAK
    return REGULAR


def daylight_mask_minutes(minutes: np.ndarray) -> np.ndarray:
    """Fixed-clock fallback mask, identical to data.DaylightHeuristic.

    Used only when the bundle carries no geometric daylight column
    (``SolarElevationMask``). Real datasets ship that deterministic,
    season/latitude-aware column and it is preferred on future rows.
    """
    return ((minutes >= 6 * 60) & (minutes <= 20 * 60)).astype(np.float32)


def _origin_index(start: int, seq_len: int) -> int:
    return int(start) + int(seq_len) - 1


def _future_daylight_mask(bundle, target_start: int, horizon: int,
                          future_minutes: np.ndarray) -> np.ndarray:
    """Horizon-wise deterministic mask for the H future rows.

    Prefers the bundle's registered daylight feature (geometric
    ``SolarElevationMask`` when present, else the synthetic clock
    heuristic) read directly from the future rows; falls back to the
    fixed 06:00-20:00 clock mask if no column is registered.
    """
    index = getattr(bundle, "daylight_index", None)
    if index is not None:
        column = np.asarray(
            bundle.raw[int(target_start) : int(target_start) + int(horizon), int(index)],
            dtype=np.float32,
        )
        if column.shape[0] == int(horizon):
            return (column >= 0.5).astype(np.float32)
    return daylight_mask_minutes(future_minutes)


def _window_parts(bundle, start: int, seq_len: int, horizon: int):
    start = int(start)
    origin = start + seq_len - 1
    target_start = start + seq_len
    recent = bundle.raw[origin - RECENT_STEPS + 1 : origin + 1, -1]
    future_stamps = bundle.timestamps[target_start : target_start + horizon]
    future_minutes = (
        future_stamps.astype("datetime64[m]").astype(np.int64) % (24 * 60)
    )
    mask = _future_daylight_mask(bundle, target_start, horizon, future_minutes)
    origin_stamp = bundle.timestamps[origin].astype("datetime64[m]")
    origin_minute = int(origin_stamp.astype(np.int64) % (24 * 60))
    month = int(origin_stamp.astype("datetime64[M]").astype(np.int64)) % 12 + 1
    return recent, mask, origin_minute, month


def window_scalar_features(bundle, start: int, seq_len: int, horizon: int):
    """Return (level_ratio, sigma_phys, ramp_ratio, state, bucket, month)."""
    recent, _, origin_minute, month = _window_parts(
        bundle, start, seq_len, horizon
    )
    capacity = float(bundle.capacity)
    level = float(np.mean(recent))
    sigma = float(np.std(recent, ddof=0))
    ramp = float(np.max(np.abs(np.diff(recent)))) if len(recent) > 1 else 0.0
    level_ratio = level / capacity
    ramp_ratio = ramp / capacity
    state = operating_state(level_ratio, ramp_ratio)
    bucket = min(3, origin_minute // (6 * 60))
    return level_ratio, sigma, ramp_ratio, state, bucket, month


def clip_prior(median: np.ndarray, mask: np.ndarray, capacity: float) -> np.ndarray:
    """C_bar = clip(C, 0, capacity) * daylight_mask (physical units)."""
    return (np.clip(np.asarray(median, dtype=np.float64), 0.0, capacity) * mask).astype(
        np.float32
    )


def stratified_partners(
    starts: np.ndarray,
    bundle,
    seq_len: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Permute windows within (month, intra-day bucket) strata.

    Returns an aligned array of partner window starts and an audit record.
    The permutation is a derangement wherever the stratum has more than
    one member; singleton strata map to themselves and are counted.
    """
    starts = np.asarray(starts, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    keys = np.empty(len(starts), dtype=[("month", "i4"), ("bucket", "i4")])
    for i, start in enumerate(starts):
        _, _, _, state, bucket, month = window_scalar_features(
            bundle, start, seq_len, horizon
        )
        del state
        keys[i] = (month, bucket)
    partners = np.empty_like(starts)
    singleton_strata = 0
    unique_keys, counts = np.unique(keys, return_counts=True)
    for key in unique_keys:
        members = np.nonzero(
            (keys["month"] == key[0]) & (keys["bucket"] == key[1])
        )[0]
        size = len(members)
        if size == 1:
            partners[members[0]] = starts[members[0]]
            singleton_strata += 1
            continue
        perm = rng.permutation(size)
        for _ in range(10):
            if not np.any(perm == np.arange(size)):
                break
            perm = rng.permutation(size)
        else:
            # Deterministic fallback: cyclic shift by one is always fixed-free.
            perm = (np.arange(size) + 1) % size
        partners[members] = starts[members[perm]]
    audit = {
        "seed": int(seed),
        "n_windows": int(len(starts)),
        "n_strata": int(len(unique_keys)),
        "singleton_strata": int(singleton_strata),
        "fixed_points": int(np.sum(partners == starts)),
        "min_stratum_size": int(min(counts)),
        "max_stratum_size": int(max(counts)),
    }
    return partners, audit


class PcFraPack:
    """Row-indexed per-window feature pack for the PC-FRA adapter.

    Vector layout follows :func:`layers.pc_fra.pack_slices`. The pack
    also carries the stratified-shuffle partner start of every window so
    the dataset can fetch the partner history (physical-token shuffle
    diagnostic recomputes the six tokens of the partner window).
    """

    kind = "pc_fra"

    def __init__(
        self,
        starts: np.ndarray,
        vectors: np.ndarray,
        partner_starts: np.ndarray,
        horizon: int,
    ) -> None:
        starts = np.asarray(starts, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        partner_starts = np.asarray(partner_starts, dtype=np.int64)
        width = pack_width(horizon)
        if vectors.ndim != 2 or vectors.shape != (len(starts), width):
            raise ValueError(f"pc-fra pack vectors must be [N, {width}]")
        if partner_starts.shape != starts.shape:
            raise ValueError("partner_starts must align with starts")
        if np.any(np.diff(starts) <= 0):
            raise ValueError("pack starts must be strictly ascending")
        self.starts = starts
        self.embeddings = vectors
        self.partner_starts = partner_starts
        self.dim = width
        self.horizon = int(horizon)

    def _position(self, start: int) -> int:
        position = int(np.searchsorted(self.starts, int(start)))
        if position >= len(self.starts) or int(self.starts[position]) != int(start):
            raise KeyError(f"no pc-fra pack row for window start {start}")
        return position

    def __getitem__(self, start: int) -> np.ndarray:
        return self.embeddings[self._position(start)]

    def partner_start(self, start: int) -> int:
        return int(self.partner_starts[self._position(start)])

    def covers(self, starts: np.ndarray) -> bool:
        index = np.searchsorted(self.starts, np.asarray(starts, dtype=np.int64))
        return bool(
            np.all(index < len(self.starts))
            and np.all(
                self.starts[np.clip(index, 0, len(self.starts) - 1)] == starts
            )
        )

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            starts=self.starts,
            vectors=self.embeddings,
            partner_starts=self.partner_starts,
            horizon=np.int64(self.horizon),
            kind=np.array("pc_fra"),
        )
        for key, value in (metadata or {}).items():
            if isinstance(value, (int, float)):
                payload[key] = np.float64(value)
            elif isinstance(value, str):
                payload[key] = np.array(value)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> "PcFraPack":
        with np.load(path) as archive:
            if str(archive["kind"]) != "pc_fra":
                raise ValueError(f"{path} is not a pc_fra pack")
            return cls(
                archive["starts"],
                archive["vectors"],
                archive["partner_starts"],
                horizon=int(archive["horizon"]),
            )


def load_chronos_median(cache_path: Path, dataset_path: Path):
    """Read the frozen Chronos-2 q0.5 cache with a full metadata audit."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing Chronos-2 v3 cache: {cache_path}")
    with np.load(cache_path) as archive:
        required = ("starts", "q050", "dataset_sha256", "context_rows")
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"{cache_path} missing fields {missing}")
        dataset_sha = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
        if str(archive["dataset_sha256"]) != dataset_sha:
            raise ValueError("Chronos cache dataset SHA256 does not match dataset")
        if int(archive["context_rows"]) != 96:
            raise ValueError("Chronos cache context_rows must be 96 (L96)")
        metadata = {
            "cache_path": str(cache_path),
            "model": str(archive["model"]) if "model" in archive else None,
            "model_revision": (
                str(archive["model_revision"])
                if "model_revision" in archive
                else None
            ),
            "recipe": str(archive["recipe"]) if "recipe" in archive else None,
            "num_samples": (
                int(archive["num_samples"]) if "num_samples" in archive else None
            ),
            "sampling_seed": (
                int(archive["sampling_seed"]) if "sampling_seed" in archive else None
            ),
            "input_contract": (
                str(archive["input_contract"]) if "input_contract" in archive else None
            ),
            "quantile_ordering": (
                str(archive["quantile_ordering"])
                if "quantile_ordering" in archive
                else None
            ),
            "dataset_sha256": dataset_sha,
            "n_windows": int(len(archive["starts"])),
        }
        starts = np.asarray(archive["starts"], dtype=np.int64)
        median = np.asarray(archive["q050"], dtype=np.float32)
    return starts, median, metadata


def compute_training_statistics(
    bundle, train_starts: np.ndarray, seq_len: int, horizon: int
) -> dict[str, float]:
    """Training-only z-score statistics for level ratio and sigma."""
    levels = np.empty(len(train_starts), dtype=np.float64)
    sigmas = np.empty(len(train_starts), dtype=np.float64)
    for i, start in enumerate(train_starts):
        level_ratio, sigma, _, _, _, _ = window_scalar_features(
            bundle, start, seq_len, horizon
        )
        levels[i] = level_ratio
        sigmas[i] = sigma
    return {
        "level_mu": float(levels.mean()),
        "level_sd": float(levels.std() + 1e-8),
        "sigma_mu": float(sigmas.mean()),
        "sigma_sd": float(sigmas.std() + 1e-8),
    }


def compute_epsilon(
    residual_std: np.ndarray, target_sd: float, capacity: float
) -> dict[str, float]:
    """p90 of |y - y_psrc| over training windows, capped at 0.2*capacity."""
    residual_std = np.asarray(residual_std, dtype=np.float64)
    raw = float(np.quantile(np.abs(residual_std), EPSILON_QUANTILE))
    cap_standardized = EPSILON_CAPACITY_FRACTION * float(capacity) / float(target_sd)
    return {
        "epsilon": float(min(raw, cap_standardized)),
        "epsilon_p90_raw": raw,
        "epsilon_cap_standardized": float(cap_standardized),
        "epsilon_capped": bool(raw > cap_standardized),
    }


def build_pack(
    bundle,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
    median_lookup: dict[int, np.ndarray],
    psrc_lookup: dict[int, np.ndarray],
    partner_starts: np.ndarray,
) -> PcFraPack:
    """Assemble the per-window pack from frozen prior + frozen PSRC outputs.

    ``median_lookup`` / ``psrc_lookup`` map window start rows to the q0.5
    trajectory (physical) and the frozen PSRC prediction (standardized).
    """
    starts = np.asarray(starts, dtype=np.int64)
    width = pack_width(horizon)
    sl = slices(horizon)
    vectors = np.empty((len(starts), width), dtype=np.float32)
    target_mu = float(bundle.feature_mu[-1])
    target_sd = float(bundle.feature_sd[-1])
    for i, start in enumerate(starts):
        start = int(start)
        _, mask, _, _ = _window_parts(bundle, start, seq_len, horizon)
        level_ratio, sigma_phys, _, state, bucket, _ = window_scalar_features(
            bundle, start, seq_len, horizon
        )
        c_bar = clip_prior(median_lookup[start], mask, bundle.capacity)
        partner = int(partner_starts[i])
        _, partner_mask, _, _ = _window_parts(bundle, partner, seq_len, horizon)
        c_bar_partner = clip_prior(
            median_lookup[partner], partner_mask, bundle.capacity
        )
        d_partner = (c_bar_partner.astype(np.float64) - target_mu) / target_sd - (
            psrc_lookup[partner]
        )
        vectors[i, sl["c_bar"]] = c_bar
        vectors[i, sl["c_bar_partner"]] = c_bar_partner
        vectors[i, sl["d_partner"]] = d_partner.astype(np.float32)
        vectors[i, sl["latest"]] = float(
            bundle.raw[start + seq_len - 1, -1]
        )
        vectors[i, sl["sigma"]] = sigma_phys
        vectors[i, sl["state"]] = 0.0
        vectors[i, sl["state"].start + state] = 1.0
        vectors[i, sl["level"]] = level_ratio
        vectors[i, sl["bucket"]] = 0.0
        vectors[i, sl["bucket"].start + bucket] = 1.0
    return PcFraPack(starts, vectors, partner_starts, horizon)


# --- Reporting helpers ------------------------------------------------------

def split_daylight_mask(bundle, starts: np.ndarray, seq_len: int, horizon: int):
    """[N, H] deterministic daylight mask over the future rows."""
    mask = np.empty((len(starts), horizon), dtype=np.float32)
    for i, start in enumerate(starts):
        _, m, _, _ = _window_parts(bundle, int(start), seq_len, horizon)
        mask[i] = m
    return mask.astype(bool)


def daylight_metrics(
    pred_physical: np.ndarray,
    target_physical: np.ndarray,
    daylight: np.ndarray,
    capacity: float,
) -> dict[str, float]:
    p = pred_physical[daylight]
    y = target_physical[daylight]
    error = p - y
    return {
        "rmse_daylight": float(np.sqrt(np.mean(error**2))),
        "mae_daylight": float(np.mean(np.abs(error))),
        "mbe_daylight": float(np.mean(error)),
        "nrmse_daylight": float(np.sqrt(np.mean(error**2)) / capacity),
        "nmae_daylight": float(np.mean(np.abs(error)) / capacity),
        "n_rows": int(len(y)),
    }


def paired_block_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    window_start_ns: np.ndarray,
    horizon_minutes: np.ndarray,
    capacity: float,
    *,
    block_days: int = 7,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, object]:
    """Paired 7-day block bootstrap CI of RMSE/MAE differences (B - A).

    Rows are grouped into contiguous calendar blocks starting at the
    first observed timestamp; the same block resample is used for both
    arms, preserving pairing.
    """
    rng = np.random.default_rng(int(seed))
    stamps = (
        np.asarray(window_start_ns, dtype="datetime64[ns]")
        + np.asarray(horizon_minutes, dtype="timedelta64[m]")
    )
    dates = stamps.astype("datetime64[D]").astype(np.int64)
    origin = dates.min()
    block_ids = (dates - origin) // int(block_days)
    unique_blocks = np.unique(block_ids)
    err_a = pred_a - target
    err_b = pred_b - target
    sq_a = err_a**2
    sq_b = err_b**2
    ab_a = np.abs(err_a)
    ab_b = np.abs(err_b)
    # Pre-aggregate per block for fast, deterministic resampling.
    blocks = {
        int(block): tuple(
            float(np.sum(vals))
            for vals in (sq_a[block_ids == block], sq_b[block_ids == block],
                         ab_a[block_ids == block], ab_b[block_ids == block])
        )
        + (int(np.sum(block_ids == block)),)
        for block in unique_blocks
    }
    base_sq_a = sum(v[0] for v in blocks.values())
    base_sq_b = sum(v[1] for v in blocks.values())
    base_ab_a = sum(v[2] for v in blocks.values())
    base_ab_b = sum(v[3] for v in blocks.values())
    n_total = sum(v[4] for v in blocks.values())

    def metrics_from(sums):
        sqa, sqb, aba, abb, n = sums
        rmse_a = np.sqrt(sqa / n)
        rmse_b = np.sqrt(sqb / n)
        return rmse_b - rmse_a, abb / n - aba / n

    boot_rmse = np.empty(n_boot)
    boot_mae = np.empty(n_boot)
    ids = list(blocks)
    for draw in range(n_boot):
        sampled = rng.integers(0, len(ids), size=len(ids))
        sums = [0.0, 0.0, 0.0, 0.0, 0]
        for idx in sampled:
            vals = blocks[ids[idx]]
            for k in range(5):
                sums[k] += vals[k]
        boot_rmse[draw], boot_mae[draw] = metrics_from(tuple(sums))
    observed_rmse, observed_mae = metrics_from(
        (base_sq_a, base_sq_b, base_ab_a, base_ab_b, n_total)
    )
    return {
        "n_blocks": int(len(ids)),
        "n_rows": int(n_total),
        "rmse_diff_observed": float(observed_rmse),
        "mae_diff_observed": float(observed_mae),
        "rmse_diff_ci95": [
            float(np.quantile(boot_rmse, 0.025)),
            float(np.quantile(boot_rmse, 0.975)),
        ],
        "mae_diff_ci95": [
            float(np.quantile(boot_mae, 0.025)),
            float(np.quantile(boot_mae, 0.975)),
        ],
        "capacity": float(capacity),
        "seed": int(seed),
    }
