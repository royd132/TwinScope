"""Leakage-safe solar geometry and physical-anchor helpers for v73."""
from __future__ import annotations

from typing import Optional

import torch


def build_solar_anchor_config(stats: dict) -> Optional[dict]:
    """Convert fitted training statistics into the model's solar-anchor config."""
    names = list(stats.get("feature_names") or [])
    future = list(stats.get("solar_future_features") or stats.get("future_features") or [])
    clear = stats.get("clear_sky_feature")
    if not clear or clear not in names or clear not in future:
        return None
    ci = names.index(clear)
    cfg = {
        "clear_hist_idx": ci,
        "clear_future_pos": future.index(clear),
        "target_mu": float(stats["target_mu"]),
        "target_sd": float(stats["target_sd"]),
        "clear_mu": float(stats["feature_mu"][ci]),
        "clear_sd": float(stats["feature_sd"][ci]),
        "clear_max": max(float(stats.get("clear_sky_training_max", 1.0)), 1e-6),
        "capacity": max(float(stats.get("capacity", 1.0)), 1e-6),
    }
    daylight = stats.get("daylight_feature")
    if daylight in names and daylight in future:
        di = names.index(daylight)
        cfg.update(daylight_future_pos=future.index(daylight),
                   daylight_mu=float(stats["feature_mu"][di]),
                   daylight_sd=float(stats["feature_sd"][di]))
    return cfg


def clear_sky_target(x: torch.Tensor, future_x: torch.Tensor, stats: dict):
    """Return a standardized clear-sky PV trajectory for deterministic anchoring."""
    cfg = build_solar_anchor_config(stats)
    if cfg is None or future_x is None:
        return None
    clear_hist = (x[..., cfg["clear_hist_idx"]] * cfg["clear_sd"] + cfg["clear_mu"]).clamp_min(0)
    clear_future = (future_x[..., cfg["clear_future_pos"]] * cfg["clear_sd"] + cfg["clear_mu"]).clamp_min(0)
    valid = clear_hist > 0.02 * cfg["clear_max"]
    latest = valid.flip(1).float().argmax(1)
    latest = (x.size(1) - 1 - latest).clamp_min(0)
    power = x[..., -1] * cfg["target_sd"] + cfg["target_mu"]
    k = (power / cfg["capacity"]) / (clear_hist / cfg["clear_max"]).clamp_min(0.02)
    k = k.gather(1, latest[:, None]).squeeze(1).clamp(0, 1.5)
    out = cfg["capacity"] * k[:, None] * clear_future / cfg["clear_max"]
    if "daylight_future_pos" in cfg:
        daylight = future_x[..., cfg["daylight_future_pos"]] * cfg["daylight_sd"] + cfg["daylight_mu"]
        out = out * (daylight > 0.5).to(out.dtype)
    return (out - cfg["target_mu"]) / max(cfg["target_sd"], 1e-6)
