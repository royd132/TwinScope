"""Training objectives used by the v73 diagram."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def v73_loss(pred, target, mask=None, kind="mse_ramp_phys_night", stats=None,
             future_x=None, model=None, ramp_weight=0.05, physics_weight=0.02):
    mask = torch.ones_like(target) if mask is None else mask.to(pred.dtype)
    point = ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
    loss = point
    if "ramp" in kind and pred.size(1) > 1:
        dp, dy = pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1]
        ramp_mask = mask[:, 1:] * mask[:, :-1]
        ramp = ((dp - dy).square() * ramp_mask).sum() / ramp_mask.sum().clamp_min(1.0)
        loss = loss + ramp_weight * ramp
    if "phys" in kind and stats is not None:
        lo = float(stats.get("target_min", 0.0)); hi = float(stats.get("target_max", 1.0))
        physical = F.relu(lo - pred).square() + F.relu(pred - hi).square()
        loss = loss + physics_weight * physical.mean()
    if "night" in kind and stats is not None and future_x is not None:
        night = stats.get("night_mask")
        if night is not None:
            loss = loss + 0.02 * (pred[night].square().mean() if night.any() else pred.new_zeros(()))
    if model is not None:
        alignment = getattr(model, "semantic_alignment_loss", None)
        if alignment is not None:
            loss = loss + 0.01 * alignment
        fsra = getattr(model, "fsra_alignment_loss", None)
        if fsra is not None:
            loss = loss + 0.01 * fsra
    return loss
