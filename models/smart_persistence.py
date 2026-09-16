"""Smart persistence baseline using deterministic future clear-sky geometry.

The forecast carries the last valid daylight clear-sky index KPV forward and
modulates it with the future clear-sky trajectory, which is a deterministic
function of time only (no observations or NWP in the future window).
"""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        geometry = getattr(configs, "solar_geometry", None)
        if geometry is None or geometry.get("clear_sky_index") is None:
            raise ValueError(
                "smart persistence requires a deterministic clear-sky "
                "geometry feature (e.g. ClearSkyGHIProxy)"
            )
        self.clear_hist_idx = int(geometry["clear_sky_index"])
        self.clear_future_pos = int(geometry["clear_future_pos"])
        self.daylight_future_pos = geometry.get("daylight_future_pos")
        self.target_mu = float(geometry["target_mu"])
        self.target_sd = float(geometry["target_sd"])
        self.clear_mu = float(geometry["clear_mu"])
        self.clear_sd = float(geometry["clear_sd"])
        self.clear_max = max(float(geometry["clear_sky_training_max"]), 1e-6)
        self.capacity = max(float(geometry["capacity"]), 1e-6)
        if self.daylight_future_pos is not None:
            self.daylight_mu = float(geometry["daylight_mu"])
            self.daylight_sd = float(geometry["daylight_sd"])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_mark_dec, mask
        if x_dec is None:
            raise ValueError(
                "smart persistence requires known future solar geometry"
            )
        power = x_enc[..., -1] * self.target_sd + self.target_mu
        clear_hist = (
            x_enc[..., self.clear_hist_idx] * self.clear_sd + self.clear_mu
        ).clamp_min(0.0)
        clear_future = (
            x_dec[..., self.clear_future_pos] * self.clear_sd + self.clear_mu
        ).clamp_min(0.0)
        clear_norm = clear_hist / self.clear_max
        valid = clear_norm > 0.02
        positions = torch.arange(x_enc.shape[1], device=x_enc.device).view(1, -1)
        latest = torch.where(
            valid, positions, positions.new_full(positions.shape, -1)
        ).max(dim=1).values
        safe_latest = latest.clamp_min(0)
        kpv_series = (power / self.capacity) / clear_norm.clamp_min(0.02)
        kpv = kpv_series.gather(1, safe_latest[:, None]).squeeze(1)
        kpv = torch.where(latest >= 0, kpv, torch.zeros_like(kpv)).clamp(0.0, 1.5)
        base_phys = self.capacity * kpv[:, None] * (clear_future / self.clear_max)
        if self.daylight_future_pos is not None:
            daylight = (
                x_dec[..., self.daylight_future_pos] * self.daylight_sd
                + self.daylight_mu
            )
            base_phys = base_phys * (daylight > 0.5).to(base_phys.dtype)
        out = (base_phys - self.target_mu) / self.target_sd
        return out.unsqueeze(-1)
