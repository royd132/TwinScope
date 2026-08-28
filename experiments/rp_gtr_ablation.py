"""Small local ablation harness for parallel GTR + MKAN + L-Context.

The implementation mirrors the local GTR/M_KAN code but keeps the experiment
self-contained so that the legacy files do not need to be re-encoded.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualDropPath(nn.Module):
    """Per-sample stochastic depth for a residual feature increment."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
        if not 0.0 <= self.drop_prob < 1.0:
            raise ValueError("drop path probability must be in [0, 1)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * keep / keep_prob


class GTR(nn.Module):
    def __init__(self, seq_len: int, channels: int, period_len: int = 24,
                 dropout: float = 0.10):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.linear = nn.Linear(seq_len, seq_len)
        k = 1 + 2 * (period_len // 2)
        self.conv = nn.Conv2d(1, 1, kernel_size=(2, k), padding=(0, k // 2), bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x, q: [B, C, L]
        b, c, l = x.shape
        g = self.linear(q)
        stacked = torch.stack([x, g], dim=2).reshape(-1, 1, 2, l)
        y = self.conv(stacked).reshape(b, c, l)
        return self.dropout(y)


class ChebyKANLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, degree: int = 3):
        super().__init__()
        self.out_dim = out_dim
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1) * 0.02)
        self.register_buffer("orders", torch.arange(degree + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        z = torch.tanh(x).unsqueeze(-1).expand(-1, -1, self.degree + 1)
        theta = torch.acos(z.clamp(-1 + 1e-5, 1 - 1e-5))
        basis = torch.cos(theta * self.orders)
        return torch.einsum("nid,iod->no", basis, self.coeff)


class MKAN(nn.Module):
    def __init__(self, channels: int, degree: int = 3):
        super().__init__()
        self.kan = ChebyKANLinear(channels, channels, degree)
        self.local = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        b, l, c = x.shape
        h1 = self.kan(x.reshape(b * l, c)).reshape(b, l, c)
        h2 = self.local(x.transpose(1, 2)).transpose(1, 2)
        return h1 + h2


class SpectralMKAN(nn.Module):
    """MKAN operating on the real/imaginary parts of the rFFT spectrum.

    The original parallel harness applied MKAN directly in the time domain,
    where its physical role overlapped with GTR and L-Drive.  Here every
    frequency bin is treated as one step and its real/imaginary coefficients
    are modeled jointly.  The output is transformed back to time and injected
    through a small learnable residual, so this branch starts close to identity.
    """
    MODES = {"legacy", "residual_off", "residual_full"}

    def __init__(self, channels: int, degree: int = 3, enhance_ratio: float = 0.1,
                 state_dim: Optional[int] = None, sample_hours: Optional[float] = None,
                 physical_bands: bool = False, state_conditioned_bands: bool = True,
                 router_activation: str = "gelu", mode: str = "legacy"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown SpectralMKAN mode: {mode}")
        self.mode = mode
        spectral_channels = 2 * channels
        self.channels = channels
        self.mkan = MKAN(spectral_channels, degree)
        self.gate = nn.Sequential(nn.Linear(2 * spectral_channels, spectral_channels), nn.Sigmoid())
        self.mix = nn.Linear(spectral_channels, spectral_channels)
        self.delta_norm = nn.LayerNorm(channels)
        self.enhance_weight = nn.Parameter(torch.tensor([enhance_ratio], dtype=torch.float32))
        # Non-persistent so legacy checkpoints retain their exact state-dict
        # contract.  The paired residual_off/residual_full controls instantiate
        # identical trainable parameters and differ only at the final mask.
        self.register_buffer(
            "residual_active",
            torch.tensor(0.0 if mode == "residual_off" else 1.0),
            persistent=False,
        )
        self.state_query = nn.Linear(state_dim, spectral_channels) if state_dim else None
        self.frequency_key = nn.Linear(spectral_channels, spectral_channels) if state_dim else None
        self.sample_hours = sample_hours
        self.physical_bands = physical_bands
        self.state_conditioned_bands = state_conditioned_bands
        if router_activation not in {"gelu", "relu"}:
            raise ValueError("router activation must be gelu or relu")
        router_act = nn.GELU if router_activation == "gelu" else nn.ReLU
        if physical_bands:
            if sample_hours is None:
                raise ValueError("physical-band routing requires sample_hours")
            if state_conditioned_bands:
                if state_dim is None:
                    raise ValueError("state-conditioned physical bands require state_dim")
                hidden = max(8, state_dim)
                self.band_router = nn.Sequential(
                    nn.Linear(state_dim + 3, hidden), router_act(), nn.Linear(hidden, 3),
                )
                nn.init.zeros_(self.band_router[-1].weight)
                nn.init.zeros_(self.band_router[-1].bias)
            else:
                self.band_router = None
        else:
            self.band_router = None
        if self.physical_bands:
            # The physical-band path uses ``band_router`` and never reaches
            # the alternative per-bin query/key attention. Keep the tensors
            # for checkpoint compatibility but exclude them from trainable
            # parameter and optimizer accounting.
            for module in (self.state_query, self.frequency_key):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        self.last_band_attention = None
        self.last_residual_gain = None
        self.last_raw_contribution_ratio = None
        self.last_contribution_ratio = None
        self.force_band_index = None
        self.band_state_intervention = None

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, C] -> spectrum: [B, F, C]
        b, l, c = x.shape
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        ri = torch.view_as_real(spectrum).reshape(b, spectrum.shape[1], 2 * c)
        # Frequency-wise normalization prevents the large DC component from
        # suppressing weaker but informative ramp/cloud bands.
        scale = torch.sqrt(ri.square().mean(dim=1, keepdim=True) + 1e-5).detach()
        z = ri / scale
        h = self.mkan(z)
        gate = self.gate(torch.cat([z, h], dim=-1))
        candidate = gate * self.mix(h) + (1.0 - gate) * z
        if self.physical_bands:
            if self.state_conditioned_bands and state is None:
                raise ValueError("physical-band spectrum requires a physical operating state")
            # Bands are defined in cycles/hour and therefore retain their
            # meaning across 5-min and 15-min sampling.  DC is excluded.
            frequency = torch.fft.rfftfreq(l, d=float(self.sample_hours)).to(x.device)
            band_masks = torch.stack([
                (frequency > 0.0) & (frequency <= 0.25),       # 4--24 h and slower resolvable evolution
                (frequency > 0.25) & (frequency <= 1.0),      # 1--4 h cloud-field evolution
                frequency > 1.0,                              # sub-hour fluctuation
            ], dim=0)
            spectral_energy = spectrum.abs().square().mean(dim=-1)
            energies = []
            available = []
            for mask in band_masks:
                available.append(bool(mask.any()))
                energies.append(spectral_energy[:, mask].mean(dim=1)
                                if mask.any() else spectral_energy.new_zeros(b))
            energy = torch.stack(energies, dim=-1)
            energy_prior = torch.log(energy + 1e-6)
            available_tensor = torch.tensor(available, device=x.device, dtype=torch.bool)
            available_count = available_tensor.sum().clamp_min(1)
            centered = energy_prior - (
                energy_prior.masked_fill(~available_tensor[None], 0.0).sum(-1, keepdim=True)
                / available_count
            )
            routed_state = state
            if self.band_state_intervention == "zero":
                routed_state = torch.zeros_like(state)
            elif self.band_state_intervention == "shuffle":
                routed_state = torch.roll(state, shifts=1, dims=0)
            logits = ((self.band_router(torch.cat([routed_state, centered], dim=-1)) + centered)
                      if self.band_router is not None else centered)
            logits = logits.masked_fill(~available_tensor[None], -1e4)
            if self.mode == "legacy":
                band_attention = torch.softmax(logits, dim=-1)
            else:
                # Independent gates preserve the original three physical
                # bands but allow several, or none, to contribute.  This fixes
                # the legacy softmax requirement that every window allocate
                # all probability mass to some spectral band.
                band_attention = torch.sigmoid(logits)
                band_attention = band_attention.masked_fill(
                    ~available_tensor[None], 0.0,
                )
            if self.force_band_index is not None:
                band_attention = F.one_hot(
                    torch.full((b,), int(self.force_band_index), device=x.device),
                    num_classes=3,
                ).to(x.dtype)
            frequency_gain = torch.einsum("bs,sf->bf", band_attention, band_masks.to(x.dtype))
            z_hat = z + frequency_gain[..., None] * (candidate - z)
            self.last_band_attention = band_attention.detach()
        elif self.state_query is not None:
            if state is None:
                raise ValueError("state-conditioned spectrum requires a physical operating state")
            # Each frequency bin is selected by an interaction between its
            # complex coefficient content and the current operating state.
            # Sigmoid (rather than softmax) allows several bands, or none, to
            # be active simultaneously.
            query = self.state_query(state)[:, None, :]
            key = self.frequency_key(z)
            logits = (query * key).sum(dim=-1) / math.sqrt(key.shape[-1])
            band_attention = torch.sigmoid(logits)
            z_hat = z + band_attention[..., None] * (candidate - z)
            self.last_band_attention = band_attention.detach()
        else:
            z_hat = candidate
            self.last_band_attention = None
        if self.mode == "legacy":
            ri_hat = (z_hat * scale).reshape(
                b, spectrum.shape[1], c, 2,
            ).contiguous()
            reconstructed = torch.fft.irfft(
                torch.view_as_complex(ri_hat), n=l, dim=1, norm="ortho",
            )
            return x + self.enhance_weight * self.delta_norm(reconstructed - x)

        # Identity-centred complex residual: retain the exact original
        # rFFT -> MKAN -> gate/mix -> physical-band route -> irFFT skeleton,
        # but reconstruct only the coefficient correction.  Crucially, do not
        # LayerNorm the time-domain delta; its learned spectral magnitude must
        # remain meaningful and may legitimately approach zero.
        delta_z = z_hat - z
        delta_ri = (delta_z * scale).reshape(
            b, spectrum.shape[1], c, 2,
        ).contiguous()
        raw_delta = torch.fft.irfft(
            torch.view_as_complex(delta_ri), n=l, dim=1, norm="ortho",
        )
        gain = torch.tanh(self.enhance_weight)
        delta = self.residual_active.to(x.dtype) * gain * raw_delta
        input_norm = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        self.last_residual_gain = gain.detach()
        self.last_raw_contribution_ratio = (
            raw_delta.flatten(1).norm(dim=1) / input_norm
        ).detach()
        self.last_contribution_ratio = (
            delta.flatten(1).norm(dim=1) / input_norm
        ).detach()
        return x + delta


class AbstentionGatedSpectralMKAN(nn.Module):
    """Conservative complex-spectrum expert with an explicit abstention route.

    This is an internal replacement for :class:`SpectralMKAN`; its input,
    output and position in the parallel time/frequency graph are unchanged.
    Unlike the legacy expert, it may assign probability to a null band, gates
    each channel independently, and caps the reconstructed residual relative
    to the input energy instead of LayerNorm-amplifying every correction to a
    nearly fixed scale.  ``off`` and ``adaptive`` instantiate identical
    parameters; a fixed output mask is their sole difference.
    """

    MODES = {"off": 0.0, "adaptive": 1.0}

    def __init__(self, channels: int, state_dim: int, sample_hours: float,
                 horizon: int, mode: str, degree: int = 3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown frequency-v5 mode: {mode}")
        if sample_hours <= 0.0:
            raise ValueError("frequency-v5 requires a positive sample interval")
        self.channels = int(channels)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode
        spectral_channels = 2 * self.channels

        self.mkan = MKAN(spectral_channels, degree)
        self.coefficient_gate = nn.Sequential(
            nn.Linear(2 * spectral_channels, spectral_channels), nn.Sigmoid(),
        )
        self.complex_mix = nn.Linear(spectral_channels, spectral_channels)
        hidden = max(8, int(state_dim))
        # Three physical bands plus an explicit null/abstention route.
        self.band_router = nn.Sequential(
            nn.Linear(int(state_dim) + 6, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.band_router[-1].weight)
        with torch.no_grad():
            self.band_router[-1].bias.copy_(
                torch.tensor([-0.3, -0.3, -0.3, 0.8])
            )
        # Per-channel reliability is based only on historical spectral shape.
        self.channel_gate = nn.Sequential(
            nn.Linear(3, max(4, hidden // 2)), nn.GELU(),
            nn.Linear(max(4, hidden // 2), 1),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.constant_(self.channel_gate[-1].bias, -1.0986123)  # 0.25
        # A small bounded residual avoids overwhelming the temporal branch.
        self.residual_logit = nn.Parameter(torch.tensor(-2.1972246))  # 0.10
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        self.last_band_attention = None
        self.last_null_probability = None
        self.last_channel_gate = None
        self.last_residual_scale = None
        self.last_contribution_ratio = None

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, length, channels = x.shape
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        ri = torch.view_as_real(spectrum).reshape(
            b, spectrum.shape[1], 2 * channels,
        )
        coefficient_scale = torch.sqrt(
            ri.square().mean(dim=1, keepdim=True) + 1e-5
        ).detach()
        z = ri / coefficient_scale
        hidden = self.mkan(z)
        coefficient_gate = self.coefficient_gate(torch.cat([z, hidden], dim=-1))
        candidate = coefficient_gate * self.complex_mix(hidden) + (1.0 - coefficient_gate) * z

        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        band_masks = torch.stack([
            (frequency > 0.0) & (frequency <= 0.25),
            (frequency > 0.25) & (frequency <= 1.0),
            frequency > 1.0,
        ], dim=0)
        energy_per_bin = spectrum.abs().square().mean(dim=-1)
        band_energy = []
        available = []
        for mask in band_masks:
            available.append(bool(mask.any()))
            band_energy.append(
                energy_per_bin[:, mask].mean(dim=1)
                if mask.any() else energy_per_bin.new_zeros(b)
            )
        energy = torch.stack(band_energy, dim=-1)
        log_energy = torch.log1p(energy)
        energy_fraction = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        spectral_entropy = -(
            energy_fraction * torch.log(energy_fraction.clamp_min(1e-6))
        ).sum(dim=-1, keepdim=True) / math.log(3.0)
        ramp = torch.diff(x, dim=1).abs().mean(dim=(1, 2), keepdim=False)[:, None]
        horizon_code = x.new_full(
            (b, 1), min(self.horizon_hours / 24.0, 2.0),
        )
        router_features = torch.cat([
            log_energy, spectral_entropy, ramp, horizon_code,
        ], dim=-1)
        logits = self.band_router(torch.cat([state, router_features], dim=-1))
        available_tensor = torch.tensor(
            available + [True], device=x.device, dtype=torch.bool,
        )
        logits = logits.masked_fill(~available_tensor[None], -1e4)
        route = torch.softmax(logits, dim=-1)
        frequency_gain = torch.einsum(
            "bs,sf->bf", route[:, :3], band_masks.to(x.dtype),
        )
        z_hat = z + frequency_gain[..., None] * (candidate - z)
        ri_hat = (z_hat * coefficient_scale).reshape(
            b, spectrum.shape[1], channels, 2,
        ).contiguous()
        reconstructed = torch.fft.irfft(
            torch.view_as_complex(ri_hat), n=length, dim=1, norm="ortho",
        )
        raw_delta = reconstructed - x

        input_rms = x.square().mean(dim=1).sqrt()
        delta_rms = raw_delta.square().mean(dim=1).sqrt()
        high_fraction = (
            spectrum[:, frequency > 1.0].abs().square().mean(dim=1)
            if bool((frequency > 1.0).any()) else torch.zeros_like(input_rms)
        ) / spectrum.abs().square().mean(dim=1).clamp_min(1e-6)
        channel_features = torch.stack([
            torch.log1p(input_rms), torch.log1p(delta_rms), high_fraction,
        ], dim=-1)
        channel_reliability = torch.sigmoid(
            self.channel_gate(channel_features)
        ).squeeze(-1)
        energy_normalized_delta = (
            raw_delta / delta_rms[:, None, :].clamp_min(1e-5)
            * input_rms[:, None, :]
        )
        residual_scale = torch.sigmoid(self.residual_logit) * self.active_mask.to(x.dtype)
        delta = (
            residual_scale * channel_reliability[:, None, :]
            * energy_normalized_delta
        )
        denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        self.last_band_attention = route.detach()
        self.last_null_probability = route[:, 3].detach()
        self.last_channel_gate = channel_reliability.detach()
        self.last_residual_scale = residual_scale.detach()
        self.last_contribution_ratio = (
            delta.flatten(1).norm(dim=1) / denominator
        ).detach()
        return x + delta


class EndogenousHierarchicalCalibratedSpectralMKAN(nn.Module):
    """Target-only hierarchical spectral correction in the original TF slot.

    The external graph is deliberately unchanged: normalized history enters
    this module in parallel with GTR/L-Drive and the returned residual is fused
    by the existing time--frequency router before semantic/Patch/CorPatch
    processing.  Internally, only historical PV Target coefficients are
    transformed.  Historical irradiance/weather channels remain in the time
    domain and provide contextual gate features.

    Three disjoint physical bands are independently sigmoid-gated, so several
    bands -- or none -- may be active.  Coefficients are energy-calibrated per
    band before a shared complex MKAN predicts bounded magnitude and phase
    adjustments.  The original spectrum remains the carrier and the final
    target residual is RMS-normalized and capped at three percent.  ``off``
    and ``full`` instantiate identical parameters; only ``active_mask`` differs.
    """

    MODES = {"off": 0.0, "full": 1.0}

    def __init__(self, channels: int, target_position: int, state_dim: int,
                 sample_hours: float, horizon: int, mode: str,
                 degree: int = 3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown frequency-v6 mode: {mode}")
        if not 0 <= int(target_position) < int(channels):
            raise ValueError("frequency-v6 target must be inside selected channels")
        if sample_hours <= 0.0:
            raise ValueError("frequency-v6 requires a positive sample interval")
        self.channels = int(channels)
        self.target_position = int(target_position)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode

        hidden = max(8, int(state_dim) // 2)
        self.complex_input = nn.Linear(2, hidden)
        self.complex_mkan = MKAN(hidden, degree)
        self.complex_output = nn.Linear(hidden, 2)
        # Context: encoded physical state + 3 band energies + entropy +
        # target ramp + four time-domain exogenous summaries + horizon.
        self.band_router = nn.Sequential(
            nn.Linear(int(state_dim) + 10, max(8, int(state_dim))), nn.GELU(),
            nn.Linear(max(8, int(state_dim)), 3),
        )
        nn.init.zeros_(self.band_router[-1].weight)
        nn.init.constant_(self.band_router[-1].bias, -2.1972246)  # 0.10
        # Maximum correction is 3%; sigmoid(-0.693) gives a 1% start.
        self.residual_logit = nn.Parameter(torch.tensor(-0.6931472))
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        channel_mask = torch.zeros(self.channels, dtype=torch.float32)
        channel_mask[self.target_position] = 1.0
        self.register_buffer("target_channel_mask", channel_mask, persistent=False)
        self.last_band_gate = None
        self.last_band_energy_fraction = None
        self.last_residual_scale = None
        self.last_contribution_ratio = None
        self.last_magnitude_adjustment = None
        self.last_phase_adjustment = None

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        target = x[:, :, self.target_position]
        spectrum = torch.fft.rfft(target, dim=1, norm="ortho")
        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        band_masks = torch.stack([
            (frequency > 0.0) & (frequency <= 0.25),
            (frequency > 0.25) & (frequency <= 1.0),
            frequency > 1.0,
        ], dim=0)

        power = spectrum.abs().square()
        energies = []
        calibrated = torch.zeros_like(spectrum)
        available = []
        for mask in band_masks:
            present = bool(mask.any())
            available.append(present)
            if present:
                energy = power[:, mask].mean(dim=1)
                scale = energy.sqrt().clamp_min(1e-5)
                calibrated[:, mask] = spectrum[:, mask] / scale[:, None]
            else:
                energy = power.new_zeros(b)
            energies.append(energy)
        energy = torch.stack(energies, dim=-1)
        energy_fraction = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(
            energy_fraction * torch.log(energy_fraction.clamp_min(1e-6))
        ).sum(dim=-1, keepdim=True) / math.log(3.0)

        if self.channels > 1:
            exogenous = torch.cat([
                x[:, :, :self.target_position],
                x[:, :, self.target_position + 1:],
            ], dim=-1)
            ex_mean = exogenous.mean(dim=(1, 2))
            ex_std = exogenous.std(dim=(1, 2), unbiased=False)
            ex_latest = exogenous[:, -1].mean(dim=1)
            ex_ramp = torch.diff(exogenous, dim=1).abs().mean(dim=(1, 2))
        else:
            ex_mean = ex_std = ex_latest = ex_ramp = target.new_zeros(b)
        target_ramp = torch.diff(target, dim=1).abs().mean(dim=1)
        horizon_code = target.new_full(
            (b,), min(self.horizon_hours / 24.0, 2.0),
        )
        context = torch.cat([
            torch.log1p(energy), entropy,
            target_ramp[:, None], ex_mean[:, None], ex_std[:, None],
            ex_latest[:, None], ex_ramp[:, None], horizon_code[:, None],
        ], dim=-1)
        gate_logits = self.band_router(torch.cat([state, context], dim=-1))
        available_tensor = torch.tensor(
            available, device=x.device, dtype=torch.bool,
        )
        gate_logits = gate_logits.masked_fill(~available_tensor[None], -20.0)
        band_gate = torch.sigmoid(gate_logits)
        frequency_gate = torch.einsum(
            "bs,sf->bf", band_gate, band_masks.to(x.dtype),
        )

        ri = torch.view_as_real(calibrated)
        latent = self.complex_mkan(F.gelu(self.complex_input(ri)))
        adjustment = self.complex_output(latent)
        # A bounded Cartesian complex gain is stable on the exact zero-valued
        # night spectrum.  Polar amplitude/phase backpropagation is ill posed
        # at zero coefficients and produced NaNs on real DKASC windows.
        real_adjustment = 0.25 * torch.tanh(adjustment[..., 0])
        imag_adjustment = 0.15 * torch.tanh(adjustment[..., 1])
        multiplier = torch.complex(
            1.0 + frequency_gate * real_adjustment,
            frequency_gate * imag_adjustment,
        )
        delta_spectrum = spectrum * (multiplier - 1.0)
        raw_delta = torch.fft.irfft(
            delta_spectrum, n=length, dim=1, norm="ortho",
        )
        # Epsilon belongs inside sqrt: clamping only after sqrt leaves an
        # infinite derivative at exact-zero night windows (sqrt'(0)), which
        # can turn the learned scale and gates into NaN after one optimizer
        # step even though the forward output itself is finite.
        target_rms = torch.sqrt(target.square().mean(dim=1) + 1e-6)
        delta_rms = torch.sqrt(raw_delta.square().mean(dim=1) + 1e-6)
        normalized_delta = (
            raw_delta / delta_rms[:, None].clamp_min(1e-5)
            * target_rms[:, None]
        )
        residual_scale = (
            0.03 * torch.sigmoid(self.residual_logit)
            * self.active_mask.to(x.dtype)
        )
        target_delta = residual_scale * normalized_delta
        output = target_delta[:, :, None] * self.target_channel_mask.to(x.dtype)[None, None, :]
        denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        self.last_band_gate = band_gate.detach()
        self.last_band_energy_fraction = energy_fraction.detach()
        self.last_residual_scale = residual_scale.detach()
        self.last_contribution_ratio = (
            output.flatten(1).norm(dim=1) / denominator
        ).detach()
        self.last_magnitude_adjustment = (
            real_adjustment.abs() * frequency_gate
        ).mean(dim=1).detach()
        self.last_phase_adjustment = (
            imag_adjustment.abs() * frequency_gate
        ).mean(dim=1).detach()
        return output


class PredictabilityGatedLocalSpectralResidual(nn.Module):
    """Target-only local spectral residual in the original parallel TF slot.

    Low-frequency solar-envelope information is context only.  The output is
    reconstructed exclusively from detrended mid/high-frequency target frames.
    Historical irradiance channels estimate coherence, while temporal spectral
    stability and a horizon-dependent prior let the expert abstain when local
    fluctuations are unlikely to persist.  Unlike frequency-v6, a small raw
    correction is never normalized upward: only an upper RMS cap is applied.

    ``off`` and ``full`` have the same modules and fusion graph.  The sole
    fixed difference is the final residual mask, and the outer TF softmax is
    deliberately not renormalized for ``off``.
    """

    MODES = {"off": 0.0, "full": 1.0}

    def __init__(self, channels: int, target_position: int, state_dim: int,
                 sample_hours: float, horizon: int, mode: str,
                 degree: int = 3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown frequency-v7 mode: {mode}")
        if not 0 <= int(target_position) < int(channels):
            raise ValueError("frequency-v7 target must be inside selected channels")
        if sample_hours <= 0.0:
            raise ValueError("frequency-v7 requires a positive sample interval")
        self.channels = int(channels)
        self.target_position = int(target_position)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode

        hidden = max(8, int(state_dim) // 2)
        self.complex_input = nn.Linear(2, hidden)
        self.complex_mkan = MKAN(hidden, degree)
        self.complex_output = nn.Linear(hidden, 2)
        # Per-band inputs: coherence, stability and local energy fraction.
        # Shared inputs: ramp persistence, low-envelope energy and horizon.
        self.confidence_router = nn.Sequential(
            nn.Linear(int(state_dim) + 9, max(8, int(state_dim))), nn.GELU(),
            nn.Linear(max(8, int(state_dim)), 2),
        )
        nn.init.zeros_(self.confidence_router[-1].weight)
        nn.init.constant_(self.confidence_router[-1].bias, -1.3862944)  # 0.20
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        channel_mask = torch.zeros(self.channels, dtype=torch.float32)
        channel_mask[self.target_position] = 1.0
        self.register_buffer("target_channel_mask", channel_mask, persistent=False)
        self.last_band_gate = None
        self.last_coherence = None
        self.last_stability = None
        self.last_horizon_prior = None
        self.last_clip_factor = None
        self.last_contribution_ratio = None
        self.last_magnitude_adjustment = None
        self.last_phase_adjustment = None

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor,
                       dim: int) -> torch.Tensor:
        shape = [1] * value.ndim
        shape[dim] = weight.numel()
        expanded = weight.reshape(shape).to(value.dtype)
        return (value * expanded).sum(dim=dim) / expanded.sum().clamp_min(1e-6)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        target = x[:, :, self.target_position]
        frame = min(length, max(4, int(round(4.0 / self.sample_hours))))
        hop = max(1, int(round(1.0 / self.sample_hours)))
        if frame == length:
            hop = frame
        window = torch.hann_window(
            frame, periodic=False, device=x.device, dtype=x.dtype,
        ).clamp_min(1e-3)

        # A four-hour moving mean supplies only the slow solar-envelope
        # context.  Its residual is the sole signal eligible for correction.
        smooth_kernel = frame if frame % 2 == 1 else min(length - 1, frame + 1)
        smooth_kernel = max(3, smooth_kernel)
        pad = smooth_kernel // 2
        padded = F.pad(target[:, None, :], (pad, pad), mode="replicate")
        envelope = F.avg_pool1d(padded, smooth_kernel, stride=1).squeeze(1)
        local_target = target - envelope
        target_frames = local_target.unfold(1, frame, hop)
        target_spectrum = torch.fft.rfft(
            target_frames * window[None, None, :], dim=-1, norm="ortho",
        )
        n_frames = target_spectrum.shape[1]
        frequency = torch.fft.rfftfreq(
            frame, d=self.sample_hours, device=x.device,
        )
        band_masks = torch.stack([
            (frequency > 0.25) & (frequency <= 1.0),
            frequency > 1.0,
        ], dim=0)
        available = band_masks.any(dim=1)

        target_power = target_spectrum.abs().square()
        band_energy = torch.stack([
            target_power[:, :, mask].mean(dim=(1, 2))
            if bool(mask.any()) else target_power.new_zeros(b)
            for mask in band_masks
        ], dim=-1)
        energy_fraction = band_energy / band_energy.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)

        recent_count = min(4, n_frames)
        recent_target = target_spectrum[:, -recent_count:, :]
        recent_power = recent_target.abs().square()
        if self.channels > 1:
            exogenous = torch.cat([
                x[:, :, :self.target_position],
                x[:, :, self.target_position + 1:],
            ], dim=-1).transpose(1, 2)
            ex_pad = F.pad(exogenous, (pad, pad), mode="replicate")
            ex_envelope = F.avg_pool1d(ex_pad, smooth_kernel, stride=1)
            ex_local = exogenous - ex_envelope
            ex_frames = ex_local.unfold(-1, frame, hop)
            ex_spectrum = torch.fft.rfft(
                ex_frames * window[None, None, None, :],
                dim=-1, norm="ortho",
            )[:, :, -recent_count:, :]
            cross = (
                recent_target[:, None, :, :] * ex_spectrum.conj()
            ).mean(dim=2)
            coherence_bins = (
                cross.abs().square()
                / (
                    recent_power.mean(dim=1)[:, None, :]
                    * ex_spectrum.abs().square().mean(dim=2)
                ).clamp_min(1e-6)
            ).clamp(0.0, 1.0).amax(dim=1)
        else:
            coherence_bins = target.new_zeros(b, frequency.numel())
        coherence = torch.stack([
            coherence_bins[:, mask].mean(dim=1)
            if bool(mask.any()) else target.new_zeros(b)
            for mask in band_masks
        ], dim=-1)

        split = max(1, recent_count // 2)
        early = recent_power[:, :split, :].mean(dim=1)
        late = recent_power[:, split:, :].mean(dim=1) if split < recent_count else early
        stability_bins = torch.exp(-torch.abs(torch.log(
            (late + 1e-6) / (early + 1e-6)
        )))
        stability = torch.stack([
            stability_bins[:, mask].mean(dim=1)
            if bool(mask.any()) else target.new_zeros(b)
            for mask in band_masks
        ], dim=-1)

        recent_diffs = torch.diff(target[:, -min(length, frame):], dim=1)
        if recent_diffs.shape[1] > 1:
            sign_agreement = (
                recent_diffs[:, 1:] * recent_diffs[:, :-1] > 0.0
            ).to(x.dtype).mean(dim=1)
        else:
            sign_agreement = target.new_zeros(b)
        ramp_strength = recent_diffs.abs().mean(dim=1) if recent_diffs.numel() else target.new_zeros(b)
        ramp_persistence = sign_agreement * torch.tanh(ramp_strength)

        global_spectrum = torch.fft.rfft(target, dim=1, norm="ortho")
        global_frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        low_mask = (global_frequency > 0.0) & (global_frequency <= 0.25)
        low_energy = (
            global_spectrum[:, low_mask].abs().square().mean(dim=1)
            if bool(low_mask.any()) else target.new_zeros(b)
        )
        horizon_code = target.new_full(
            (b,), min(self.horizon_hours / 24.0, 2.0),
        )
        context = torch.cat([
            coherence, stability, energy_fraction,
            ramp_persistence[:, None], torch.log1p(low_energy)[:, None],
            horizon_code[:, None],
        ], dim=-1)
        learned_gate = torch.sigmoid(
            self.confidence_router(torch.cat([state, context], dim=-1))
        )
        # Cloud/ramp bands have shorter persistence at longer horizons.  This
        # fixed physical prior prevents H48 from borrowing short-lived detail.
        tau = target.new_tensor([8.0, 2.0])
        horizon_prior = torch.exp(-self.horizon_hours / tau)
        predictability = torch.sqrt((coherence * stability).clamp_min(0.0))
        band_gate = learned_gate * predictability * horizon_prior[None, :]
        band_gate = band_gate * available.to(x.dtype)[None, :]
        frequency_gate = torch.einsum(
            "bs,sf->bf", band_gate, band_masks.to(x.dtype),
        )

        coefficient_scale = target_spectrum.abs().mean(
            dim=(1, 2), keepdim=True,
        ).clamp_min(1e-5)
        normalized = target_spectrum / coefficient_scale
        ri = torch.view_as_real(normalized)
        n_frequency = ri.shape[2]
        flattened_ri = ri.reshape(b, n_frames * n_frequency, 2)
        latent = self.complex_mkan(
            F.gelu(self.complex_input(flattened_ri))
        )
        adjustment = self.complex_output(latent).reshape(
            b, n_frames, n_frequency, 2,
        )
        real_adjustment = 0.20 * torch.tanh(adjustment[..., 0])
        imag_adjustment = 0.10 * torch.tanh(adjustment[..., 1])
        complex_adjustment = torch.complex(
            frequency_gate[:, None, :] * real_adjustment,
            frequency_gate[:, None, :] * imag_adjustment,
        )
        delta_spectrum = target_spectrum * complex_adjustment
        frame_delta = torch.fft.irfft(
            delta_spectrum, n=frame, dim=-1, norm="ortho",
        ) * window[None, None, :]
        folded = F.fold(
            frame_delta.transpose(1, 2), output_size=(1, length),
            kernel_size=(1, frame), stride=(1, hop),
        ).squeeze(1).squeeze(1)
        normalization = F.fold(
            window.square()[None, :, None].expand(b, -1, n_frames),
            output_size=(1, length), kernel_size=(1, frame),
            stride=(1, hop),
        ).squeeze(1).squeeze(1).clamp_min(1e-4)
        raw_delta = folded / normalization

        # Upper clipping only: useful small corrections remain small, and an
        # unconfident expert can produce an exact/near-exact zero residual.
        target_rms = torch.sqrt(target.square().mean(dim=1) + 1e-6)
        raw_rms = torch.sqrt(raw_delta.square().mean(dim=1) + 1e-6)
        maximum_rms = 0.03 * target_rms
        clip_factor = torch.minimum(
            torch.ones_like(raw_rms), maximum_rms / raw_rms.clamp_min(1e-6),
        )
        target_delta = (
            self.active_mask.to(x.dtype) * clip_factor[:, None] * raw_delta
        )
        output = (
            target_delta[:, :, None]
            * self.target_channel_mask.to(x.dtype)[None, None, :]
        )
        denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        self.last_band_gate = band_gate.detach()
        self.last_coherence = coherence.detach()
        self.last_stability = stability.detach()
        self.last_horizon_prior = horizon_prior.detach()
        self.last_clip_factor = clip_factor.detach()
        self.last_contribution_ratio = (
            output.flatten(1).norm(dim=1) / denominator
        ).detach()
        self.last_magnitude_adjustment = (
            real_adjustment.abs() * frequency_gate[:, None, :]
        ).mean(dim=(1, 2)).detach()
        self.last_phase_adjustment = (
            imag_adjustment.abs() * frequency_gate[:, None, :]
        ).mean(dim=(1, 2)).detach()
        return output


class RobustGlobalMultivariateSpectralMKAN(nn.Module):
    """Robust original-style multivariate spectrum in the unchanged TF slot.

    This expert deliberately restores the mechanisms that made the original
    spectral branch plausible: a full-window global rFFT, joint complex MKAN
    mixing across physically coherent PV/irradiance channels, and direct
    low/mid/high-band residuals.  The low band uses robust level histories;
    mid/high bands use robust first differences.  A channel-specific band gate
    replaces the single shared router, and an upper-only per-channel RMS cap
    replaces cross-channel LayerNorm so coherent solar responses are retained.

    ``off`` and ``full`` instantiate the same parameters.  Only the final
    residual mask differs; the outer temporal/frequency/identity router and
    every downstream module are unchanged.
    """

    MODES = {"off": 0.0, "full": 1.0}

    def __init__(self, channels: int, state_dim: int, sample_hours: float,
                 horizon: int, mode: str, degree: int = 3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown frequency-v8 mode: {mode}")
        if channels < 2:
            raise ValueError("frequency-v8 requires multivariate PV/irradiance input")
        if sample_hours <= 0.0:
            raise ValueError("frequency-v8 requires a positive sample interval")
        self.channels = int(channels)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode
        spectral_channels = 2 * self.channels

        self.mkan = MKAN(spectral_channels, degree)
        self.coefficient_gate = nn.Sequential(
            nn.Linear(2 * spectral_channels, spectral_channels), nn.Sigmoid(),
        )
        self.complex_mix = nn.Linear(spectral_channels, spectral_channels)
        hidden = max(8, int(state_dim))
        # State + per-channel low/mid/high energies + horizon -> a distinct
        # gate for every band/channel pair.  No softmax competition is used.
        self.band_channel_router = nn.Sequential(
            nn.Linear(int(state_dim) + 3 * self.channels + 1, hidden),
            nn.GELU(), nn.Linear(hidden, 3 * self.channels),
        )
        nn.init.zeros_(self.band_channel_router[-1].weight)
        nn.init.zeros_(self.band_channel_router[-1].bias)  # sigmoid = 0.5
        # A maximum 8% internal residual with a 4% initial cap.  This cap never
        # amplifies a small raw correction; the outer TF gate scales it again.
        self.residual_logits = nn.Parameter(torch.zeros(self.channels))
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        self.last_band_channel_gate = None
        self.last_source_energy_fraction = None
        self.last_robust_clip_fraction = None
        self.last_residual_cap = None
        self.last_residual_clip_fraction = None
        self.last_raw_contribution_ratio = None
        self.last_contribution_ratio = None
        self.last_channel_contribution_ratio = None

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, length, channels = x.shape
        # Protect the FFT from isolated sensor spikes without changing the
        # dataset or ordinary in-range observations.  The minimum +/-4 RevIN
        # units keeps normal daylight trajectories untouched.
        center = x.median(dim=1, keepdim=True).values
        mad = (x - center).abs().median(dim=1, keepdim=True).values
        robust_limit = (8.0 * 1.4826 * mad).clamp_min(4.0)
        lower, upper = center - robust_limit, center + robust_limit
        robust_x = torch.maximum(torch.minimum(x, upper), lower)
        robust_clipped = (robust_x != x).to(x.dtype).mean(dim=(1, 2))

        differences = torch.cat([
            torch.zeros_like(robust_x[:, :1]),
            robust_x[:, 1:] - robust_x[:, :-1],
        ], dim=1)
        level_spectrum = torch.fft.rfft(robust_x, dim=1, norm="ortho")
        difference_spectrum = torch.fft.rfft(
            differences, dim=1, norm="ortho",
        )
        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        band_masks = torch.stack([
            (frequency > 0.0) & (frequency <= 0.25),
            (frequency > 0.25) & (frequency <= 1.0),
            frequency > 1.0,
        ], dim=0)
        source_spectrum = torch.where(
            band_masks[0][None, :, None], level_spectrum,
            difference_spectrum,
        )
        # DC is intentionally excluded from every residual band.
        source_spectrum = source_spectrum * band_masks.any(dim=0)[None, :, None]

        source_ri = torch.view_as_real(source_spectrum)
        band_ri_scale = torch.stack([
            torch.sqrt(source_ri[:, mask].square().mean(dim=1) + 1e-5)
            if bool(mask.any()) else source_ri.new_ones(b, channels, 2)
            for mask in band_masks
        ], dim=1).detach()
        frequency_ri_scale = torch.einsum(
            "bscd,sf->bfcd", band_ri_scale, band_masks.to(x.dtype),
        ).clamp_min(1e-5)
        ri = source_ri.reshape(b, source_spectrum.shape[1], 2 * channels)
        coefficient_scale = frequency_ri_scale.reshape(
            b, source_spectrum.shape[1], 2 * channels,
        )
        z = ri / coefficient_scale
        hidden = self.mkan(z)
        coefficient_gate = self.coefficient_gate(torch.cat([z, hidden], dim=-1))
        candidate = (
            coefficient_gate * self.complex_mix(hidden)
            + (1.0 - coefficient_gate) * z
        )

        power = source_spectrum.abs().square()
        band_energy = torch.stack([
            power[:, mask, :].mean(dim=1)
            if bool(mask.any()) else power.new_zeros(b, channels)
            for mask in band_masks
        ], dim=1)
        energy_fraction = band_energy / band_energy.sum(
            dim=1, keepdim=True,
        ).clamp_min(1e-6)
        horizon_code = x.new_full(
            (b, 1), min(self.horizon_hours / 24.0, 2.0),
        )
        router_input = torch.cat([
            state, torch.log1p(band_energy).flatten(1), horizon_code,
        ], dim=-1)
        band_channel_gate = torch.sigmoid(
            self.band_channel_router(router_input)
        ).reshape(b, 3, channels)
        frequency_channel_gate = torch.einsum(
            "bsc,sf->bfc", band_channel_gate, band_masks.to(x.dtype),
        )

        delta_z = (
            frequency_channel_gate.repeat_interleave(2, dim=-1)
            * (candidate - z)
        )
        delta_ri = (delta_z * coefficient_scale).reshape(
            b, source_spectrum.shape[1], channels, 2,
        ).contiguous()
        delta_spectrum = torch.view_as_complex(delta_ri)
        raw_delta = torch.fft.irfft(
            delta_spectrum, n=length, dim=1, norm="ortho",
        )

        input_rms = torch.sqrt(robust_x.square().mean(dim=1) + 1e-6)
        raw_rms = torch.sqrt(raw_delta.square().mean(dim=1) + 1e-6)
        residual_cap = 0.06 * torch.sigmoid(self.residual_logits)
        maximum_rms = residual_cap[None, :] * input_rms
        clip_factor = torch.minimum(
            torch.ones_like(raw_rms),
            maximum_rms / raw_rms.clamp_min(1e-6),
        )
        delta = (
            self.active_mask.to(x.dtype)
            * clip_factor[:, None, :] * raw_delta
        )
        input_norm = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        channel_norm = x.norm(dim=1).clamp_min(1e-6)
        self.last_band_channel_gate = band_channel_gate.detach()
        self.last_source_energy_fraction = energy_fraction.detach()
        self.last_robust_clip_fraction = robust_clipped.detach()
        self.last_residual_cap = residual_cap.detach()
        self.last_residual_clip_fraction = (
            clip_factor < 0.999999
        ).to(x.dtype).mean(dim=1).detach()
        self.last_raw_contribution_ratio = (
            raw_delta.flatten(1).norm(dim=1) / input_norm
        ).detach()
        self.last_contribution_ratio = (
            delta.flatten(1).norm(dim=1) / input_norm
        ).detach()
        self.last_channel_contribution_ratio = (
            delta.norm(dim=1) / channel_norm
        ).detach()
        return delta


class EndogenousTrendSpectralAdapter(nn.Module):
    """Target-only global spectral adapter in the unchanged parallel slot.

    Frequency-v9 keeps the original outer graph intact, but removes the two
    mechanisms that made v8 untrainable in practice.  First, the endogenous PV
    target is separated from exogenous irradiance histories: only the target is
    transformed by the rFFT, while irradiance summaries condition the spectral
    gates.  Second, the spectral direction and its physical amplitude are
    learned separately, so a hard post-hoc RMS cap cannot cancel the gradients
    of the coefficient and band gates.

    Internally, one-hour and four-hour moving averages form an adaptive trend.
    Three independent complex MKAN experts model low, middle and high bands of
    that trend.  Their residual is reconstructed as a unit-RMS direction and a
    state-conditioned bounded amplitude is applied directly.  ``off`` and
    ``full`` instantiate identical parameters; only the final residual mask
    differs.
    """

    MODES = {"off": 0.0, "full": 1.0}

    def __init__(self, channels: int, target_position: int, state_dim: int,
                 sample_hours: float, horizon: int, mode: str,
                 degree: int = 3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown frequency-v9 mode: {mode}")
        if channels < 2:
            raise ValueError("frequency-v9 requires Target plus irradiance context")
        if not 0 <= int(target_position) < int(channels):
            raise ValueError("frequency-v9 target position is invalid")
        if sample_hours <= 0.0:
            raise ValueError("frequency-v9 requires a positive sample interval")
        self.channels = int(channels)
        self.target_position = int(target_position)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.fast_kernel = max(1, int(round(1.0 / self.sample_hours)))
        self.slow_kernel = max(
            self.fast_kernel, int(round(4.0 / self.sample_hours)),
        )
        self.mode = mode
        hidden = max(8, int(state_dim))

        # Independent pathways prevent the dominant slow PV envelope from
        # sharing all nonlinear parameters with cloud/ramp frequencies.
        self.band_mkan = nn.ModuleList([MKAN(2, degree) for _ in range(3)])
        self.band_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()) for _ in range(3)
        ])
        self.band_mix = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])

        exogenous_channels = self.channels - 1
        # State + three band energies + trend/residual energy + two summaries
        # per exogenous channel + horizon code.
        router_dim = int(state_dim) + 3 + 2 + 2 * exogenous_channels + 1
        self.trend_router = nn.Sequential(
            nn.Linear(int(state_dim) + 5, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.band_router = nn.Sequential(
            nn.Linear(router_dim, hidden), nn.GELU(), nn.Linear(hidden, 3),
        )
        self.amplitude_router = nn.Sequential(
            nn.Linear(router_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        for router in (self.trend_router, self.band_router,
                       self.amplitude_router):
            nn.init.zeros_(router[-1].weight)
            nn.init.zeros_(router[-1].bias)
        # 4.8% initial target-RMS correction, learnable in [0, 12%].  Unlike
        # v8 this is the actual amplitude, not an upper bound hit by every row.
        self.residual_logit = nn.Parameter(torch.tensor(-0.4054651))
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        target_mask = torch.zeros(self.channels, dtype=torch.float32)
        target_mask[self.target_position] = 1.0
        self.register_buffer("target_mask", target_mask, persistent=False)

        self.last_trend_mix = None
        self.last_band_gate = None
        self.last_residual_scale = None
        self.last_trend_energy_fraction = None
        self.last_raw_contribution_ratio = None
        self.last_contribution_ratio = None
        self.last_target_contribution_ratio = None
        self.last_robust_clip_fraction = None

    @staticmethod
    def _moving_average(value: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return value
        left = (kernel - 1) // 2
        right = kernel // 2
        padded = F.pad(
            value[:, None, :], (left, right), mode="replicate",
        )
        return F.avg_pool1d(padded, kernel_size=kernel, stride=1).squeeze(1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, length, channels = x.shape
        center = x.median(dim=1, keepdim=True).values
        mad = (x - center).abs().median(dim=1, keepdim=True).values
        robust_limit = (8.0 * 1.4826 * mad).clamp_min(4.0)
        robust_x = torch.maximum(
            torch.minimum(x, center + robust_limit), center - robust_limit,
        )
        robust_clipped = (robust_x != x).to(x.dtype).mean(dim=(1, 2))

        target = robust_x[:, :, self.target_position]
        target_rms = torch.sqrt(target.square().mean(dim=1) + 1e-6)
        fast_trend = self._moving_average(target, self.fast_kernel)
        slow_trend = self._moving_average(target, min(self.slow_kernel, length))
        fast_residual = target - fast_trend
        slow_residual = target - slow_trend
        fast_fraction = (
            fast_trend.square().mean(dim=1)
            / target.square().mean(dim=1).clamp_min(1e-6)
        ).clamp(0.0, 4.0)
        slow_fraction = (
            slow_trend.square().mean(dim=1)
            / target.square().mean(dim=1).clamp_min(1e-6)
        ).clamp(0.0, 4.0)
        fast_residual_fraction = (
            fast_residual.square().mean(dim=1)
            / target.square().mean(dim=1).clamp_min(1e-6)
        ).clamp(0.0, 4.0)
        slow_residual_fraction = (
            slow_residual.square().mean(dim=1)
            / target.square().mean(dim=1).clamp_min(1e-6)
        ).clamp(0.0, 4.0)
        horizon_code = x.new_full(
            (b, 1), min(self.horizon_hours / 24.0, 2.0),
        )
        trend_features = torch.stack([
            torch.log1p(fast_fraction), torch.log1p(slow_fraction),
            torch.log1p(fast_residual_fraction),
            torch.log1p(slow_residual_fraction),
        ], dim=-1)
        trend_mix = torch.sigmoid(self.trend_router(torch.cat([
            state, trend_features, horizon_code,
        ], dim=-1)))
        trend = trend_mix * fast_trend + (1.0 - trend_mix) * slow_trend
        trend = trend - trend.mean(dim=1, keepdim=True)

        spectrum = torch.fft.rfft(trend, dim=1, norm="ortho")
        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        band_masks = torch.stack([
            (frequency > 0.0) & (frequency <= 0.25),
            (frequency > 0.25) & (frequency <= 1.0),
            frequency > 1.0,
        ], dim=0)
        source_ri = torch.view_as_real(spectrum)
        band_energy = torch.stack([
            spectrum[:, mask].abs().square().mean(dim=1)
            if bool(mask.any()) else spectrum.real.new_zeros(b)
            for mask in band_masks
        ], dim=-1)
        band_energy_fraction = band_energy / band_energy.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)

        target_variance = target.square().mean(dim=1).clamp_min(1e-6)
        trend_fraction = (
            trend.square().mean(dim=1) / target_variance
        ).clamp(0.0, 4.0)
        residual_fraction = (
            (target - trend).square().mean(dim=1) / target_variance
        ).clamp(0.0, 4.0)
        exogenous_mask = torch.ones(channels, dtype=torch.bool, device=x.device)
        exogenous_mask[self.target_position] = False
        exogenous = robust_x[:, :, exogenous_mask]
        exogenous_level = (
            exogenous.abs().mean(dim=1) / target_rms[:, None].clamp_min(1e-4)
        ).clamp(0.0, 10.0)
        exogenous_ramp = (
            torch.diff(exogenous, dim=1).abs().mean(dim=1)
            / target_rms[:, None].clamp_min(1e-4)
        ).clamp(0.0, 10.0)
        router_input = torch.cat([
            state, torch.log1p(band_energy),
            torch.log1p(torch.stack([trend_fraction, residual_fraction], dim=-1)),
            torch.log1p(exogenous_level), torch.log1p(exogenous_ramp),
            horizon_code,
        ], dim=-1)
        band_gate = torch.sigmoid(self.band_router(router_input))

        delta_spectrum = torch.zeros_like(spectrum)
        for band_index, mask in enumerate(band_masks):
            if not bool(mask.any()):
                continue
            ri = source_ri[:, mask, :]
            coefficient_scale = torch.sqrt(
                ri.square().mean(dim=1, keepdim=True) + 1e-5,
            ).detach()
            z = ri / coefficient_scale
            hidden = self.band_mkan[band_index](z)
            coefficient_gate = self.band_gate[band_index](
                torch.cat([z, hidden], dim=-1),
            )
            # PGN-style multiplicative gating produces a residual directly;
            # it does not interpolate back toward an arbitrary reconstruction.
            delta_z = (
                torch.tanh(self.band_mix[band_index](hidden))
                * coefficient_gate
                * band_gate[:, band_index, None, None]
            )
            delta_ri = delta_z * coefficient_scale
            delta_spectrum[:, mask] = torch.view_as_complex(
                delta_ri.contiguous(),
            )

        raw_delta = torch.fft.irfft(
            delta_spectrum, n=length, dim=1, norm="ortho",
        )
        raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        raw_rms = torch.sqrt(raw_delta.square().mean(dim=1) + 1e-6)
        direction = raw_delta / raw_rms[:, None]
        residual_scale = 0.12 * torch.sigmoid(
            self.residual_logit + self.amplitude_router(router_input).squeeze(-1)
        )
        target_delta = (
            self.active_mask.to(x.dtype) * residual_scale[:, None]
            * target_rms[:, None] * direction
        )
        output = target_delta[:, :, None] * self.target_mask[None, None, :]

        input_norm = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        target_norm = target.norm(dim=1).clamp_min(1e-6)
        self.last_trend_mix = trend_mix.detach()
        self.last_band_gate = band_gate.detach()
        self.last_residual_scale = residual_scale.detach()
        self.last_trend_energy_fraction = torch.stack([
            trend_fraction, residual_fraction,
        ], dim=-1).detach()
        self.last_raw_contribution_ratio = (
            raw_delta.norm(dim=1) / target_norm
        ).detach()
        self.last_contribution_ratio = (
            output.flatten(1).norm(dim=1) / input_norm
        ).detach()
        self.last_target_contribution_ratio = (
            target_delta.norm(dim=1) / target_norm
        ).detach()
        self.last_robust_clip_fraction = robust_clipped.detach()
        return output


class SelectiveDualBandSpectralResidual(nn.Module):
    """Zero-initialized additive spectrum for physically coupled variables.

    The raw sequence supplies the slow solar/weather envelope, while a
    four-hour moving-average residual supplies cloud/ramp fluctuations.  Both
    experts are instantiated for every control so temporal-only, raw-low,
    detrended-mid-high and dual-frequency runs have identical parameters and
    initialization.  A fixed branch mask is the only difference.
    """

    MODES = {
        "temporal": (0.0, 0.0),
        "raw_low": (1.0, 0.0),
        "detrended_mid_high": (0.0, 1.0),
        "dual": (1.0, 1.0),
    }

    def __init__(self, channels: int, state_dim: int, sample_hours: float,
                 horizon: int, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown selective spectrum mode: {mode}")
        if sample_hours <= 0.0:
            raise ValueError("selective spectrum requires a positive sample interval")
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode
        self.low_expert = SpectralMKAN(channels, physical_bands=False)
        self.mid_high_expert = SpectralMKAN(channels, physical_bands=False)
        hidden = max(8, int(state_dim))
        self.branch_router = nn.Sequential(
            nn.Linear(int(state_dim) + 4, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Equal initial routing keeps paired controls deterministic.  Actual
        # contribution starts at exactly zero and is learned only when the
        # validation objective provides a useful gradient.
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)
        self.branch_alpha = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "branch_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        self.last_branch_gate = None
        self.last_branch_scale = None
        self.last_contribution_ratio = None
        self.last_source_energy = None

    def _band_component(self, x: torch.Tensor, lower: float,
                        upper: Optional[float]) -> torch.Tensor:
        length = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=x.device,
        )
        mask = frequency > float(lower)
        if upper is not None:
            mask = mask & (frequency <= float(upper))
        filtered = spectrum * mask[None, :, None].to(spectrum.dtype)
        return torch.fft.irfft(filtered, n=length, dim=1, norm="ortho")

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Slow component: raw PV/irradiance periods of four hours and longer.
        low_source = self._band_component(x, lower=0.0, upper=0.25)

        # Disturbance component: remove a four-hour envelope, then retain the
        # resolvable 0.5--4 h cloud/ramp band (0.25 cycles/hour to Nyquist).
        kernel = max(3, int(round(4.0 / self.sample_hours)))
        if kernel % 2 == 0:
            kernel += 1
        xt = x.transpose(1, 2)
        smooth = F.avg_pool1d(
            F.pad(xt, (kernel // 2, kernel // 2), mode="replicate"),
            kernel_size=kernel, stride=1,
        ).transpose(1, 2)
        detrended = x - smooth
        mid_high_source = self._band_component(
            detrended, lower=0.25, upper=None,
        )

        low_delta = self.low_expert(low_source) - low_source
        mid_high_delta = (
            self.mid_high_expert(mid_high_source) - mid_high_source
        )
        low_energy = low_source.square().mean(dim=(1, 2)).sqrt()
        mid_high_energy = mid_high_source.square().mean(dim=(1, 2)).sqrt()
        ramp = (
            torch.diff(x, dim=1).abs().mean(dim=(1, 2))
            if x.shape[1] > 1 else torch.zeros_like(low_energy)
        )
        horizon_code = x.new_full(
            (x.shape[0],), min(self.horizon_hours / 24.0, 2.0),
        )
        gate_features = torch.stack([
            torch.log1p(low_energy), torch.log1p(mid_high_energy),
            ramp, horizon_code,
        ], dim=-1)
        gate = torch.softmax(
            self.branch_router(torch.cat([state, gate_features], dim=-1)),
            dim=-1,
        )
        scale = torch.tanh(self.branch_alpha) * self.branch_mask.to(x.dtype)
        low_contribution = gate[:, 0, None, None] * scale[0] * low_delta
        mid_high_contribution = (
            gate[:, 1, None, None] * scale[1] * mid_high_delta
        )
        denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        contribution_ratio = torch.stack([
            low_contribution.flatten(1).norm(dim=1) / denominator,
            mid_high_contribution.flatten(1).norm(dim=1) / denominator,
        ], dim=-1)
        self.last_branch_gate = gate.detach()
        self.last_branch_scale = scale.detach()
        self.last_contribution_ratio = contribution_ratio.detach()
        self.last_source_energy = torch.stack([
            low_energy, mid_high_energy,
        ], dim=-1).detach()
        return low_contribution + mid_high_contribution


class OriginalPositionDualScaleSpectralResidual(nn.Module):
    """Target-led global/local expert inside the original parallel spectral branch.

    The module reads the same RevIN-normalized input as the temporal branch; it
    never reads the temporal branch output.  It returns an independent residual
    with the same ``[B,L,C]`` shape consumed by the established time/frequency
    router before semantic correction and Patch/CorPatch encoding.
    A global rFFT expert models the slow PV envelope; a local STFT expert
    models time-localized cloud transitions.  Historical irradiance spectra
    never become an independent prediction path and only gate the local target
    spectrum.  Fixed masks yield parameter-matched internal ablations.
    """

    MODES = {
        "off": (0.0, 0.0),
        "global_low": (1.0, 0.0),
        "local_cloud": (0.0, 1.0),
        "dual": (1.0, 1.0),
    }

    def __init__(self, channels: int, target_position: int, state_dim: int,
                 sample_hours: float, horizon: int, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown original-position frequency-v4 mode: {mode}")
        if not 0 <= int(target_position) < int(channels):
            raise ValueError("frequency-v4 target must be inside selected channels")
        self.channels = int(channels)
        self.target_position = int(target_position)
        self.sample_hours = float(sample_hours)
        self.horizon_hours = float(horizon) * self.sample_hours
        self.mode = mode

        self.global_expert = SpectralMKAN(1, physical_bands=False)
        frame = max(8, int(round(4.0 / self.sample_hours)))
        if frame % 2:
            frame += 1
        self.n_fft = frame
        self.hop_length = max(1, frame // 4)
        self.register_buffer(
            "stft_window", torch.hann_window(frame, periodic=True),
            persistent=False,
        )
        hidden = max(8, int(state_dim) // 2)
        self.local_complex = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.exogenous_gate = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.branch_router = nn.Sequential(
            nn.Linear(int(state_dim) + 4, max(8, int(state_dim))), nn.GELU(),
            nn.Linear(max(8, int(state_dim)), 2),
        )
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)
        self.branch_alpha = nn.Parameter(torch.full((2,), 0.20273255))
        self.register_buffer(
            "branch_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        channel_mask = torch.zeros(self.channels, dtype=torch.float32)
        channel_mask[self.target_position] = 1.0
        self.register_buffer("target_channel_mask", channel_mask, persistent=False)
        self.last_branch_gate = None
        self.last_branch_scale = None
        self.last_contribution_ratio = None
        self.last_source_energy = None
        self.last_local_spectral_gate = None

    def _global_low(self, target: torch.Tensor) -> torch.Tensor:
        length = target.shape[1]
        spectrum = torch.fft.rfft(target, dim=1, norm="ortho")
        frequency = torch.fft.rfftfreq(
            length, d=self.sample_hours, device=target.device,
        )
        mask = (frequency > 0.0) & (frequency <= 0.25)
        return torch.fft.irfft(
            spectrum * mask[None, :, None].to(spectrum.dtype),
            n=length, dim=1, norm="ortho",
        )

    def _stft(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            signal, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window.to(device=signal.device, dtype=signal.dtype),
            center=True, normalized=True, return_complex=True,
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        target = x[:, :, self.target_position:self.target_position + 1]
        low_source = self._global_low(target)
        global_delta = self.global_expert(low_source) - low_source

        target_spectrum = self._stft(target.squeeze(-1))
        if self.channels > 1:
            exogenous = torch.cat([
                x[:, :, :self.target_position],
                x[:, :, self.target_position + 1:],
            ], dim=-1)
            b, l, c = exogenous.shape
            exogenous_spectrum = self._stft(
                exogenous.permute(0, 2, 1).reshape(b * c, l)
            ).reshape(b, c, target_spectrum.shape[1], target_spectrum.shape[2])
            exogenous_amplitude = exogenous_spectrum.abs().mean(dim=1)
        else:
            exogenous_amplitude = torch.zeros_like(target_spectrum.abs())
        amplitude_features = torch.stack([
            torch.log1p(target_spectrum.abs()),
            torch.log1p(exogenous_amplitude),
        ], dim=-1)
        local_gate = torch.sigmoid(self.exogenous_gate(amplitude_features)).squeeze(-1)
        real_imag = torch.view_as_real(target_spectrum)
        coefficient_scale = torch.sqrt(
            real_imag.square().mean(dim=2, keepdim=True) + 1e-5
        ).detach()
        normalized = real_imag / coefficient_scale
        candidate = self.local_complex(normalized)
        local_delta_spectrum = torch.view_as_complex(
            ((candidate - normalized) * local_gate[..., None]
             * coefficient_scale).contiguous()
        )
        local_delta = torch.istft(
            local_delta_spectrum, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft,
            window=self.stft_window.to(device=x.device, dtype=x.dtype),
            center=True, normalized=True, length=x.shape[1],
        ).unsqueeze(-1)

        low_energy = low_source.square().mean(dim=(1, 2)).sqrt()
        local_energy = local_delta.square().mean(dim=(1, 2)).sqrt()
        ramp = torch.diff(target, dim=1).abs().mean(dim=(1, 2))
        horizon_code = x.new_full(
            (x.shape[0],), min(self.horizon_hours / 24.0, 2.0),
        )
        router_features = torch.stack([
            torch.log1p(low_energy), torch.log1p(local_energy),
            ramp, horizon_code,
        ], dim=-1)
        branch_gate = torch.sigmoid(
            self.branch_router(torch.cat([state, router_features], dim=-1))
        )
        scale = torch.tanh(self.branch_alpha) * self.branch_mask.to(x.dtype)
        global_contribution = (
            branch_gate[:, 0, None, None] * scale[0] * global_delta
        )
        local_contribution = (
            branch_gate[:, 1, None, None] * scale[1] * local_delta
        )
        target_contribution = global_contribution + local_contribution
        output = (
            target_contribution
            * self.target_channel_mask.to(x.dtype)[None, None, :]
        )
        denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
        contribution_ratio = torch.stack([
            global_contribution.flatten(1).norm(dim=1) / denominator,
            local_contribution.flatten(1).norm(dim=1) / denominator,
        ], dim=-1)
        self.last_branch_gate = branch_gate.detach()
        self.last_branch_scale = scale.detach()
        self.last_contribution_ratio = contribution_ratio.detach()
        self.last_source_energy = torch.stack([
            low_energy, local_energy,
        ], dim=-1).detach()
        self.last_local_spectral_gate = local_gate.detach()
        return output


class EndogenousSpectralTargetGate(nn.Module):
    """Output-level spectral modulation for the endogenous PV target only.

    The historical target is projected to a latent sequence, transformed by a
    learnable complex linear map in the Fourier domain, and returned to the
    time domain before being aligned with the target-channel CorPatch memory.
    Following time--frequency collaborative gating, the spectral memory gates
    a bounded temporal proposal instead of competing with the temporal route
    for softmax probability mass.  Every validation control instantiates the
    same parameters; ``mode`` changes only a fixed on/off buffer.
    """

    MODES = {"off": 0.0, "gate": 1.0}

    def __init__(self, d_model: int, n_tokens: int, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown endogenous spectral gate mode: {mode}")
        self.mode = mode
        self.n_tokens = int(n_tokens)
        self.input_projection = nn.Linear(1, d_model, bias=False)
        # W = W_r + j W_i.  Sharing the two real-valued maps across the real
        # and imaginary equations implements an actual complex linear layer.
        self.complex_real = nn.Linear(d_model, d_model, bias=False)
        self.complex_imag = nn.Linear(d_model, d_model, bias=False)
        self.frequency_norm = nn.LayerNorm(d_model)
        self.time_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.time_projection = nn.Linear(d_model, d_model)
        self.frequency_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        # v2 started at exactly zero and learned too slowly.  A 0.10 initial
        # residual is conservative but gives the spectral path an immediate,
        # testable gradient.  The fixed mask keeps paired controls exact.
        self.residual_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.register_buffer(
            "active_mask", torch.tensor(self.MODES[mode], dtype=torch.float32),
            persistent=True,
        )
        self.last_gate = None
        self.last_scale = None
        self.last_contribution_ratio = None
        self.last_spectral_energy = None

    def forward(self, target_history: torch.Tensor,
                target_memory: torch.Tensor) -> torch.Tensor:
        # target_history: [B,L], target_memory: [B,N,D]
        embedded = self.input_projection(target_history.unsqueeze(-1))
        spectrum = torch.fft.rfft(embedded, dim=1, norm="ortho")
        real, imag = spectrum.real, spectrum.imag
        transformed_real = (
            self.complex_real(real) - self.complex_imag(imag)
        )
        transformed_imag = (
            self.complex_real(imag) + self.complex_imag(real)
        )
        transformed = torch.complex(
            F.gelu(transformed_real), F.gelu(transformed_imag),
        )
        reconstructed = torch.fft.irfft(
            transformed, n=target_history.shape[1], dim=1, norm="ortho",
        )
        frequency_memory = F.adaptive_avg_pool1d(
            reconstructed.transpose(1, 2), self.n_tokens,
        ).transpose(1, 2)
        frequency_memory = self.frequency_norm(frequency_memory)

        temporal_proposal = torch.tanh(
            self.time_projection(self.time_norm(target_memory))
        )
        spectral_gate = torch.sigmoid(
            self.frequency_projection(frequency_memory)
        )
        interaction = self.output_projection(
            temporal_proposal * spectral_gate
        )
        scale = torch.sigmoid(self.residual_logit)
        contribution = (
            self.active_mask.to(target_memory.dtype) * scale * interaction
        )
        denominator = target_memory.flatten(1).norm(dim=1).clamp_min(1e-6)
        contribution_ratio = (
            contribution.flatten(1).norm(dim=1) / denominator
        )
        self.last_gate = spectral_gate.detach()
        self.last_scale = scale.detach()
        self.last_contribution_ratio = contribution_ratio.detach()
        self.last_spectral_energy = (
            spectrum.abs().square().mean(dim=(1, 2)).sqrt().detach()
        )
        return target_memory + contribution


class PhysicalStateEncoder(nn.Module):
    """Summarize solar/ramp context for routing decisions.

    The selected channels are solar geometry, clear-sky/daylight clocks and
    historical target power when available.  Mean, variability, latest state
    and mean absolute ramp are used; all are computed from the history only.
    """
    def __init__(self, channels: int, physics_indices: Optional[Sequence[int]], d_model: int):
        super().__init__()
        idx = sorted(set(int(i) for i in (physics_indices or range(channels)) if 0 <= int(i) < channels))
        if not idx:
            idx = list(range(channels))
        self.register_buffer("indices", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(4 * len(idx), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.index_select(-1, self.indices)
        ramp = torch.diff(p, dim=1).abs().mean(dim=1) if p.shape[1] > 1 else torch.zeros_like(p[:, 0])
        summary = torch.cat([
            p.mean(dim=1),
            p.std(dim=1, unbiased=False),
            p[:, -1],
            ramp,
        ], dim=-1)
        return self.net(summary)


class SequentialPhysicalStateEncoder(nn.Module):
    """History-only operating-state encoder for PCATFR.

    A temporal convolution detects short physical transitions; a GRU retains
    their order and persistence.  A compact statistics skip prevents the
    sequential path from losing absolute level, variability and ramp cues.
    The output is a controller state, never a third forecasting branch.
    """
    def __init__(self, channels: int, physics_indices: Optional[Sequence[int]], d_model: int):
        super().__init__()
        idx = sorted(set(int(i) for i in (physics_indices or range(channels))
                         if 0 <= int(i) < channels))
        if not idx:
            idx = list(range(channels))
        self.register_buffer("indices", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(len(idx), d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.statistics_skip = nn.Linear(4 * len(idx), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.index_select(-1, self.indices)
        ramp = (torch.diff(p, dim=1).abs().mean(dim=1)
                if p.shape[1] > 1 else torch.zeros_like(p[:, 0]))
        summary = torch.cat([
            p.mean(dim=1), p.std(dim=1, unbiased=False), p[:, -1], ramp,
        ], dim=-1)
        local = self.temporal_conv(p.transpose(1, 2)).transpose(1, 2)
        sequential, _ = self.gru(local)
        return self.norm(sequential[:, -1] + self.statistics_skip(summary))


class MultiHorizonPhysicalStateEncoder(nn.Module):
    """Separate long-background state from the current local PV regime.

    A single summary over a 336/720-step history can average several weather
    regimes and collapse the router to dataset-level constants.  This encoder
    retains the full-window context while adding a recent 96-step summary,
    which corresponds to one day for 15-min data and eight hours for 5-min
    data.  No future information is used.
    """
    def __init__(self, channels: int, physics_indices: Optional[Sequence[int]], d_model: int):
        super().__init__()
        idx = sorted(set(int(i) for i in (physics_indices or range(channels)) if 0 <= int(i) < channels))
        if not idx:
            idx = list(range(channels))
        self.register_buffer("indices", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(8 * len(idx), d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
        )

    @staticmethod
    def _summary(p: torch.Tensor) -> torch.Tensor:
        ramp = torch.diff(p, dim=1).abs().mean(dim=1) if p.shape[1] > 1 else torch.zeros_like(p[:, 0])
        return torch.cat([p.mean(dim=1), p.std(dim=1, unbiased=False), p[:, -1], ramp], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.index_select(-1, self.indices)
        recent = p[:, -min(96, p.shape[1]):]
        return self.net(torch.cat([self._summary(p), self._summary(recent)], dim=-1))


class LContext(nn.Module):
    """Faithful local implementation of the supplied L-Drive block.

    It returns the enhanced sequence ``x + alpha * latent_delta``.  The
    previous harness returned only a GRU tensor after an extra Conv/LayerNorm,
    which changed the module's residual semantics and made its branch scale
    incomparable with GTR/MKAN.
    """
    def __init__(self, channels: int, enhance_ratio: float = 0.1):
        super().__init__()
        self.dx_gate = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())
        self.gru = nn.GRU(channels, channels, batch_first=True)
        self.enhance_weight = nn.Parameter(torch.tensor([enhance_ratio], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        mask = self.dx_gate(torch.cat([x, dx], dim=-1))
        latent, _ = self.gru(mask * dx)
        return x + self.enhance_weight * latent


# Explicit paper/module name used in experiment reports.
LatentContextBlock = LContext


class PatchEmbed(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B*C, Np, D]
        x = F.pad(x, (0, self.pad), mode="replicate")
        p = x.unfold(-1, self.patch_len, self.stride)
        b, c, n, _ = p.shape
        return self.drop(self.proj(p.reshape(b * c, n, self.patch_len)))


class MultiScalePatch(nn.Module):
    """PatchTST-style multi-scale tokenization with explicit scale bias."""
    def __init__(self, patch_len: int, stride: int, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        candidates = [(max(4, patch_len // 2), max(2, stride // 2)),
                      (patch_len, stride),
                      (min(seq_len, patch_len * 2), stride * 2)]
        scales = []
        for pl, st in candidates:
            if pl <= seq_len and (pl, st) not in scales:
                scales.append((pl, st))
        self.scales = scales
        self.blocks = nn.ModuleList([PatchEmbed(pl, st, d_model, dropout) for pl, st in scales])
        self.scale_bias = nn.Parameter(torch.zeros(len(scales), d_model))
        self.scale_logits = nn.Parameter(torch.zeros(len(scales)))
        # Put every scale on the middle-scale token grid before the shared
        # coherence encoder/head. Direct concatenation makes the head interpret
        # fine/coarse tokens as one uniformly sampled time axis.
        mid_pl, mid_st = scales[len(scales) // 2]
        self.n_tokens = int((seq_len + mid_st - mid_pl) / mid_st + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i, block in enumerate(self.blocks):
            y = block(x) + self.scale_bias[i].view(1, 1, -1)
            if y.shape[1] != self.n_tokens:
                y = F.interpolate(y.transpose(1, 2), size=self.n_tokens, mode="linear", align_corners=False).transpose(1, 2)
            outs.append(y)
        weights = torch.softmax(self.scale_logits, dim=0)
        return sum(weights[i] * outs[i] for i in range(len(outs)))


class ScaleStackPatch(nn.Module):
    """Multi-scale patching that preserves an explicit scale axis.

    Output shape is [B, C, S, N, D].  Unlike ``MultiScalePatch``, scale
    identity is not destroyed before the correlation encoder.
    """
    def __init__(self, patch_len: int, stride: int, d_model: int, seq_len: int,
                 dropout: float = 0.1, single_scale: bool = False,
                 physical_scales: Optional[Sequence[tuple[int, int]]] = None):
        super().__init__()
        if physical_scales:
            candidates = list(physical_scales)
            if single_scale:
                candidates = [candidates[len(candidates) // 2]]
        else:
            candidates = ([(patch_len, stride)] if single_scale else
                          [(max(4, patch_len // 2), max(2, stride // 2)),
                           (patch_len, stride),
                           (min(seq_len, patch_len * 2), stride * 2)])
        scales = []
        for pl, st in candidates:
            if pl <= seq_len and (pl, st) not in scales:
                scales.append((pl, st))
        self.scales = scales
        self.blocks = nn.ModuleList([PatchEmbed(pl, st, d_model, dropout) for pl, st in scales])
        self.scale_bias = nn.Parameter(torch.zeros(len(scales), d_model))
        mid_pl, mid_st = scales[len(scales) // 2]
        self.n_tokens = int((seq_len + mid_st - mid_pl) / mid_st + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        outs = []
        for i, block in enumerate(self.blocks):
            y = block(x) + self.scale_bias[i].view(1, 1, -1)
            if y.shape[1] != self.n_tokens:
                y = F.interpolate(y.transpose(1, 2), size=self.n_tokens, mode="linear", align_corners=False).transpose(1, 2)
            outs.append(y.reshape(b, c, self.n_tokens, -1))
        return torch.stack(outs, dim=2)


class ScaleWiseDilatedTokenMixer(nn.Module):
    """Refine local temporal structure inside every physical patch scale.

    ``ScaleStackPatch`` exposes tokens as ``[B,C,S,N,D]``.  The upstream
    GTR/L-Drive and spectral modules construct the time-frequency response,
    while the downstream CorPatch encoder already mixes hidden channels and
    variables.  This block therefore acts only on the token-time axis ``N``:
    every variable/latent channel is filtered independently by a small bank of
    depthwise dilated convolutions.  Keeping that responsibility narrow avoids
    duplicating either the temporal experts or the variable correlation block.

    The residual scale is initialized to zero.  Enabling the module is thus an
    exact identity transformation before training, which makes its ablation a
    low-risk addition to an already validated backbone.
    """

    def __init__(self, channels: int, n_scales: int, d_model: int,
                 dropout: float = 0.05, dilations: Sequence[int] = (1, 2, 4),
                 adaptive_aggregation: bool = True):
        super().__init__()
        if not dilations:
            raise ValueError("ScaleWiseDilatedTokenMixer requires at least one dilation")
        width = int(channels) * int(d_model)
        self.channels = int(channels)
        self.n_scales = int(n_scales)
        self.d_model = int(d_model)
        self.adaptive_aggregation = bool(adaptive_aggregation)
        self.branches = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    width, width, kernel_size=3, dilation=int(dilation),
                    padding=int(dilation), groups=width,
                )
                for dilation in dilations
            ])
            for _ in range(self.n_scales)
        ])
        if self.adaptive_aggregation:
            self.branch_logits = nn.Parameter(
                torch.zeros(self.n_scales, len(dilations))
            )
        else:
            self.register_buffer(
                "branch_logits",
                torch.zeros(self.n_scales, len(dilations)),
                persistent=False,
            )
        self.norms = nn.ModuleList([
            nn.GroupNorm(self.channels, width)
            for _ in range(self.n_scales)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(float(dropout)) for _ in range(self.n_scales)
        ])
        self.residual_scale = nn.Parameter(torch.zeros(self.n_scales))
        self.last_branch_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,S,N,D], got shape {tuple(x.shape)}")
        b, c, s, n, d = x.shape
        if c != self.channels or s != self.n_scales or d != self.d_model:
            raise ValueError(
                "scale-token shape mismatch: "
                f"expected C/S/D={self.channels}/{self.n_scales}/{self.d_model}, "
                f"got {c}/{s}/{d}"
            )
        weights = torch.softmax(self.branch_logits, dim=-1)
        outputs = []
        for scale_idx in range(s):
            # [B,C,N,D] -> [B,C*D,N]; each C-D channel remains independent.
            z = x[:, :, scale_idx].permute(0, 1, 3, 2).reshape(b, c * d, n)
            branch_values = torch.stack([
                branch(z) for branch in self.branches[scale_idx]
            ], dim=1)
            local = (
                branch_values
                * weights[scale_idx].view(1, -1, 1, 1)
            ).sum(dim=1)
            local = F.gelu(self.norms[scale_idx](local))
            local = self.dropouts[scale_idx](local)
            gain = torch.tanh(self.residual_scale[scale_idx])
            refined = z + gain * local
            outputs.append(
                refined.reshape(b, c, d, n).permute(0, 1, 3, 2)
            )
        self.last_branch_weights = weights.detach()
        return torch.stack(outputs, dim=2)


class CoherenceEncoder(nn.Module):
    def __init__(self, channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.var_mix = nn.Linear(channels, channels, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bc, n, d = x.shape
        b = bc // self.channels
        h = x.reshape(b, self.channels, n, d)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        qf, kf, vf = torch.fft.rfft(q, dim=2), torch.fft.rfft(k, dim=2), torch.fft.rfft(v, dim=2)
        coh = (qf * kf.conj()).abs() / (qf.abs() * kf.abs()).clamp_min(1e-5)
        w = torch.softmax(coh.mean(-1), dim=2)
        y = torch.fft.irfft(vf * w.unsqueeze(-1), n=n, dim=2)
        y = y + self.var_mix(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = y.reshape(bc, n, d)
        y = self.norm1(x + y)
        return self.norm2(y + self.ffn(y))


class FACTEncoder(nn.Module):
    def __init__(self, channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.dw1 = nn.Conv2d(d_model, d_model, (3, 3), padding=(1, 1), groups=d_model)
        self.dw2 = nn.Conv2d(d_model, d_model, (3, 3), padding=(2, 1), dilation=(2, 1), groups=d_model)
        self.pw = nn.Conv2d(d_model, d_model, 1)
        self.drop = nn.Dropout2d(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bc, n, d = x.shape
        b = bc // self.channels
        h = x.reshape(b, self.channels, n, d).permute(0, 3, 1, 2)
        y = F.gelu(self.pw(self.dw1(h) + self.dw2(h)))
        y = self.drop(y).permute(0, 2, 3, 1).reshape(bc, n, d)
        return self.norm(x + y)


class VariableAttentionEncoder(nn.Module):
    """Permutation-equivariant correlation encoding across variables.

    ``FACTEncoder`` applies a convolution along the variable axis, so its
    output depends on the arbitrary CSV column order.  This block treats the
    variables at every patch token as a set: reordering the input variables
    produces exactly the same reordering at the output.
    """
    def __init__(self, channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        heads = 4 if d_model % 4 == 0 else 1
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                semantic_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        bc, n, d = x.shape
        b = bc // self.channels
        # [B*C,N,D] -> [B*N,C,D]: attention is only across the set of variables.
        h = x.reshape(b, self.channels, n, d).permute(0, 2, 1, 3).reshape(b * n, self.channels, d)
        # A float attention mask is additive before softmax.  FSRA supplies a
        # small CxC relevance prior derived only from frozen channel metadata;
        # ``None`` recovers the exact permutation-equivariant numeric encoder.
        if semantic_bias is not None and h.is_cuda:
            # PyTorch 2.11 Flash-SDPA can fail in backward for a broadcast 2-D
            # additive mask ("LSE strideH").  C is small, so the deterministic
            # math kernel is inexpensive and keeps the intended gradients.
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                attended, _ = self.attn(
                    h, h, h, attn_mask=semantic_bias, need_weights=False,
                )
        else:
            attended, _ = self.attn(
                h, h, h, attn_mask=semantic_bias, need_weights=False,
            )
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ffn(h))
        return h.reshape(b, n, self.channels, d).permute(0, 2, 1, 3).reshape(bc, n, d)


class ScaleAwareCorPatchEncoder(nn.Module):
    """Within-scale correlation encoding plus explicit cross-scale attention."""
    def __init__(self, channels: int, n_scales: int, d_model: int, encoder: str,
                 dropout: float = 0.1, pre_scale_bias: bool = True):
        super().__init__()
        self.channels = channels
        self.n_scales = n_scales
        if encoder == "coherence":
            self.base = CoherenceEncoder(channels, d_model, dropout)
        elif encoder == "variable":
            self.base = VariableAttentionEncoder(channels, d_model, dropout)
        else:
            self.base = FACTEncoder(channels, d_model, dropout)
        self.pre_scale_bias = pre_scale_bias
        self.scale_embed = nn.Parameter(torch.zeros(n_scales, d_model))
        heads = 4 if d_model % 4 == 0 else 1
        self.cross_scale = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, scale_weights: torch.Tensor,
                semantic_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, S, N, D], weights: [B, S]
        b, c, s, n, d = x.shape
        within = x.permute(0, 2, 1, 3, 4).reshape(b * s * c, n, d)
        if isinstance(self.base, VariableAttentionEncoder):
            within = self.base(within, semantic_bias=semantic_bias)
        else:
            within = self.base(within)
        within = within.reshape(b, s, c, n, d).permute(0, 2, 3, 1, 4)  # [B,C,N,S,D]
        tokens = within + self.scale_embed.view(1, 1, 1, s, d)
        # Legacy models use the route both before attention and at aggregation.
        # The revised path uses it only once at aggregation to avoid squaring a
        # scale preference and to keep beta interpretable as a convex weight.
        if self.pre_scale_bias:
            tokens = tokens * (1.0 + scale_weights[:, None, None, :, None])
        flat = tokens.reshape(b * c * n, s, d)
        # ``B*C*N`` is an independent batch axis.  PyTorch's fused attention
        # kernel has a 65535 batch-index limit, reached by long look-backs with
        # many variables.  Chunking this axis is mathematically identical and
        # avoids changing the scale-token attention itself.
        attention_batch = 8192
        if flat.shape[0] > attention_batch:
            attended = torch.cat([
                self.cross_scale(chunk, chunk, chunk, need_weights=False)[0]
                for chunk in flat.split(attention_batch, dim=0)
            ], dim=0)
        else:
            attended, _ = self.cross_scale(flat, flat, flat, need_weights=False)
        flat = self.norm1(flat + attended)
        flat = self.norm2(flat + self.ffn(flat))
        cross = flat.reshape(b, c, n, s, d).permute(0, 1, 3, 2, 4)
        fused = (cross * scale_weights[:, None, :, None, None]).sum(dim=2)
        return fused.reshape(b * c, n, d)


class NWPConditionedForecastDecoder(nn.Module):
    """Optional NWP adapter that returns a history-memory correction only.

    The frozen historical forecaster owns the primary prediction. Future NWP
    appears only in Q, while K and V come from historical response memory. The
    missing query residual prevents a direct NWP-to-power bypass. Zero
    initialization makes enabling this adapter identical to the historical
    model before the adapter learns a repeatable correction.
    """
    def __init__(self, future_channels: int, horizon: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if future_channels <= 0:
            raise ValueError("NWP decoder requires at least one audited future covariate")
        self.horizon = horizon
        self.future_embed = nn.Sequential(
            nn.Linear(future_channels, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.lead_embed = nn.Embedding(horizon, d_model)
        self.history_state = nn.Linear(d_model, d_model)
        heads = 4 if d_model % 4 == 0 else 1
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.nwp_gate = nn.Linear(2 * d_model, 1)
        self.nwp_correction_head = nn.Linear(d_model, 1)
        # NWP starts as a conservative correction, never the primary source.
        nn.init.constant_(self.nwp_gate.bias, -2.0)
        nn.init.zeros_(self.nwp_correction_head.weight)
        nn.init.zeros_(self.nwp_correction_head.bias)

    def forward(self, memory: torch.Tensor, future_x: torch.Tensor,
                history_state: torch.Tensor) -> torch.Tensor:
        # memory: [B,N,D], future_x: [B,H,C_future], state: [B,D]
        if future_x.shape[1] != self.horizon:
            raise ValueError(
                f"future horizon mismatch: expected {self.horizon}, got {future_x.shape[1]}"
            )
        lead = torch.arange(self.horizon, device=future_x.device)
        state = self.history_state(history_state).unsqueeze(1).expand(-1, self.horizon, -1)
        nwp_query = self.future_embed(future_x) + self.lead_embed(lead).unsqueeze(0)
        attended, _ = self.cross_attention(nwp_query, memory, memory, need_weights=False)
        # Deliberately omit a query residual here.  NWP exists only in Q and
        # can change which historical values are retrieved; it cannot bypass
        # the memory and directly synthesize a power trajectory.
        decoded = self.norm1(attended)
        decoded = self.norm2(decoded + self.ffn(decoded))
        correction = self.nwp_correction_head(decoded).squeeze(-1)
        gate = torch.sigmoid(self.nwp_gate(torch.cat([state, decoded], dim=-1))).squeeze(-1)
        return gate * correction


class TokenLocalResidualRefiner(nn.Module):
    """Lightweight local refinement on the shared CorPatch token memory.

    The routed time-frequency and scale-aware encoder remains the only memory
    constructor.  This block merely restores short token-grid edges that can
    be attenuated by cross-scale averaging; it is not an extra forecasting
    branch.
    """

    def __init__(self, channels: int, d_model: int, dropout: float = 0.05,
                 dilations: Sequence[int] = (1, 2, 4)):
        super().__init__()
        width = channels * d_model
        self.channels = int(channels)
        self.d_model = int(d_model)
        self.depthwise = nn.ModuleList([
            nn.Conv1d(
                width, width, kernel_size=3, dilation=int(dilation),
                padding=int(dilation), groups=width,
            )
            for dilation in dilations
        ])
        self.norm = nn.GroupNorm(channels, width)
        self.gate = nn.Conv1d(width, 2 * width, kernel_size=1, groups=channels)
        self.project = nn.Conv1d(width, width, kernel_size=1, groups=channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: [B,C,N,D]
        b, c, n, d = memory.shape
        z = memory.permute(0, 1, 3, 2).reshape(b, c * d, n)
        local = torch.stack([conv(z) for conv in self.depthwise], dim=0).mean(dim=0)
        local = F.gelu(self.norm(local))
        local = F.glu(self.gate(local), dim=1)
        local = self.dropout(self.project(self.dropout(local)))
        refined = z + torch.tanh(self.residual_scale) * local
        return refined.reshape(b, c, d, n).permute(0, 1, 3, 2)


class ScaleMatchedPromptAdapter(nn.Module):
    """Zero-safe frozen-LLM residual alignment on the explicit scale axis.

    ``prompt`` contains one offline GPT-2 embedding for every physical patch
    scale.  Each prompt may affect only the matching scale.  Independent
    aligners expose different semantic response subspaces, while a
    sample/channel/scale router combines them.  Crucially, every path from the
    prompt to the residual is bias-free; an all-zero prompt is therefore the
    exact numerical backbone rather than another trainable constant branch.
    """

    def __init__(self, prompt_dim: int, d_model: int, n_scales: int,
                 n_heads: int = 3, dropout: float = 0.10,
                 contrastive_scale_alignment: bool = False):
        super().__init__()
        if prompt_dim <= 0 or n_scales <= 0 or n_heads <= 0:
            raise ValueError("scale-matched prompt dimensions must be positive")
        self.prompt_dim = int(prompt_dim)
        self.d_model = int(d_model)
        self.n_scales = int(n_scales)
        self.n_heads = int(n_heads)
        self.contrastive_scale_alignment = bool(contrastive_scale_alignment)
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_dim, elementwise_affine=False),
            nn.Linear(prompt_dim, d_model, bias=False), nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.aligners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model, bias=False), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, d_model, bias=False),
            )
            for _ in range(n_heads)
        ])
        self.head_router = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_heads),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.head_router[-1].weight)
        nn.init.zeros_(self.head_router[-1].bias)
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, -2.5)
        for aligner in self.aligners:
            nn.init.normal_(aligner[-1].weight, std=5e-3)
        self.last_gate = None
        self.last_head_weights = None
        self.last_prompt_state = None
        self.last_scale_match_accuracy = None
        self.auxiliary_loss = None

    def forward(self, tokens: torch.Tensor, prompt: torch.Tensor,
                strength: float = 1.0) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError(f"expected scale tokens [B,C,S,N,D], got {tuple(tokens.shape)}")
        if prompt.ndim != 3:
            raise ValueError(f"expected prompt bank [B,S,E], got {tuple(prompt.shape)}")
        b, c, s, n, d = tokens.shape
        if s != self.n_scales or d != self.d_model:
            raise ValueError(
                f"scale token mismatch: expected S/D={self.n_scales}/{self.d_model}, got {s}/{d}"
            )
        if prompt.shape[0] != b or prompt.shape[1] != s or prompt.shape[2] != self.prompt_dim:
            raise ValueError(
                "prompt bank mismatch: expected "
                f"[{b},{s},{self.prompt_dim}], got {list(prompt.shape)}"
            )

        semantic = self.prompt_projection(prompt.to(tokens.dtype))  # [B,S,D]
        semantic_cs = semantic[:, None].expand(-1, c, -1, -1)
        pooled_numeric = tokens.mean(dim=3)
        context = torch.cat([pooled_numeric, semantic_cs], dim=-1)
        head_weights = torch.softmax(self.head_router(context), dim=-1)

        semantic_grid = semantic_cs[:, :, :, None, :].expand(-1, -1, -1, n, -1)
        paired = torch.cat([semantic_grid, tokens * semantic_grid], dim=-1)
        head_outputs = torch.stack(
            [aligner(paired) for aligner in self.aligners], dim=4,
        )  # [B,C,S,N,H,D]
        delta = (head_outputs * head_weights[:, :, :, None, :, None]).sum(dim=4)
        gate = torch.sigmoid(self.residual_gate(context))
        refined = tokens + float(strength) * gate[:, :, :, None, :] * delta

        mean_head_use = head_weights.mean(dim=(0, 1, 2))
        uniform = torch.full_like(mean_head_use, 1.0 / self.n_heads)
        balance = (mean_head_use - uniform).square().mean()
        normalized = F.normalize(head_outputs, dim=-1)
        pairs = []
        for i in range(self.n_heads):
            for j in range(i + 1, self.n_heads):
                pairs.append(
                    (normalized[..., i, :] * normalized[..., j, :])
                    .sum(dim=-1).square().mean()
                )
        diversity = (torch.stack(pairs).mean() if pairs
                     else tokens.new_zeros(()))
        if self.contrastive_scale_alignment and self.n_scales > 1:
            # The numerical history encoder is frozen during prompt tuning.
            # Within each sample, the three prompt scales are therefore
            # contrasted only against the corresponding 2/4/8 h numerical
            # summaries.  This supplies scale identity, not a forecast target.
            numeric_scale = pooled_numeric.detach().mean(dim=1)
            semantic_unit = F.normalize(semantic, dim=-1)
            numeric_unit = F.normalize(numeric_scale, dim=-1)
            logits = torch.einsum("bsd,btd->bst", semantic_unit, numeric_unit) / 0.20
            labels = torch.arange(self.n_scales, device=tokens.device)
            forward_ce = F.cross_entropy(
                logits.reshape(-1, self.n_scales),
                labels.unsqueeze(0).expand(b, -1).reshape(-1),
            )
            reverse_ce = F.cross_entropy(
                logits.transpose(1, 2).reshape(-1, self.n_scales),
                labels.unsqueeze(0).expand(b, -1).reshape(-1),
            )
            scale_alignment = 0.5 * (forward_ce + reverse_ce)
            self.last_scale_match_accuracy = (
                logits.argmax(dim=-1)
                == labels.unsqueeze(0)
            ).float().mean().detach()
        else:
            scale_alignment = tokens.new_zeros(())
            self.last_scale_match_accuracy = None
        self.auxiliary_loss = scale_alignment + 0.05 * balance + 0.01 * diversity
        self.last_gate = gate.detach()
        self.last_head_weights = head_weights.detach()
        self.last_prompt_state = semantic.mean(dim=1)
        return refined


class PromptConditionedTFRouter(nn.Module):
    """Bias-free frozen-LLM residual on temporal/frequency route logits."""

    def __init__(self, prompt_dim: int, state_dim: int, routes: int = 3):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.routes = int(routes)
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_dim, elementwise_affine=False),
            nn.Linear(prompt_dim, state_dim, bias=False), nn.GELU(),
            nn.Linear(state_dim, state_dim, bias=False),
        )
        self.route_delta = nn.Sequential(
            nn.LayerNorm(2 * state_dim, elementwise_affine=False),
            nn.Linear(2 * state_dim, state_dim, bias=False), nn.GELU(),
            nn.Linear(state_dim, routes, bias=False),
        )
        nn.init.normal_(self.route_delta[-1].weight, std=1e-3)
        self.last_delta = None

    def forward(self, prompt: torch.Tensor,
                physical_state: torch.Tensor) -> torch.Tensor:
        if prompt.ndim != 3 or prompt.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"expected prompt bank [B,S,{self.prompt_dim}], got {tuple(prompt.shape)}"
            )
        semantic = self.prompt_projection(prompt.to(physical_state.dtype)).mean(dim=1)
        context = torch.cat([semantic, semantic * physical_state], dim=-1)
        delta = self.route_delta(context)
        self.last_delta = delta.detach()
        return delta


class PromptConditionedHierarchicalRouter(nn.Module):
    """Frozen-language prototype prior for TF and patch-scale routing.

    The prompt bank contains categorical PV operating-regime descriptions
    encoded offline by a frozen language model.  Numerical physical state is a
    query over those prototypes.  The selected semantic context contributes
    only residual biases to existing router logits; it is never a third
    forecasting feature stream and cannot directly alter the prediction head.
    """

    def __init__(self, prompt_dim: int, state_dim: int,
                 tf_routes: int, scale_routes: int,
                 centered_route_delta: bool = False,
                 content_pairing: bool = False):
        super().__init__()
        if min(prompt_dim, state_dim, tf_routes, scale_routes) <= 0:
            raise ValueError("hierarchical prompt router dimensions must be positive")
        self.prompt_dim = int(prompt_dim)
        self.state_dim = int(state_dim)
        self.tf_routes = int(tf_routes)
        self.scale_routes = int(scale_routes)
        self.centered_route_delta = bool(centered_route_delta)
        self.content_pairing = bool(content_pairing)
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_dim, elementwise_affine=False),
            nn.Linear(prompt_dim, state_dim, bias=False),
            nn.GELU(),
            nn.Linear(state_dim, state_dim, bias=False),
        )
        self.state_query = nn.Sequential(
            nn.LayerNorm(state_dim, elementwise_affine=False),
            nn.Linear(state_dim, state_dim, bias=False),
        )
        context_dim = 3 * state_dim
        self.tf_delta = nn.Sequential(
            nn.LayerNorm(context_dim, elementwise_affine=False),
            nn.Linear(context_dim, state_dim, bias=False), nn.GELU(),
            nn.Linear(state_dim, tf_routes, bias=False),
        )
        self.scale_delta = nn.Sequential(
            nn.LayerNorm(context_dim, elementwise_affine=False),
            nn.Linear(context_dim, state_dim, bias=False), nn.GELU(),
            nn.Linear(state_dim, scale_routes, bias=False),
        )
        nn.init.normal_(self.tf_delta[-1].weight, std=1e-3)
        nn.init.normal_(self.scale_delta[-1].weight, std=1e-3)
        self.log_temperature = nn.Parameter(torch.tensor(-1.0))
        self.last_prototype_weights = None
        self.last_tf_delta = None
        self.last_scale_delta = None
        self.pairing_loss = None

    def forward(self, prompt: torch.Tensor,
                physical_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt.ndim != 3 or prompt.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"expected prompt bank [B,P,{self.prompt_dim}], got {tuple(prompt.shape)}"
            )
        if physical_state.ndim != 2 or physical_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected physical state [B,{self.state_dim}], "
                f"got {tuple(physical_state.shape)}"
            )
        prototypes = self.prompt_projection(prompt.to(physical_state.dtype))
        query = self.state_query(physical_state)
        temperature = self.log_temperature.exp().clamp(0.05, 2.0)
        scores = torch.einsum(
            "bd,bpd->bp",
            F.normalize(query, dim=-1),
            F.normalize(prototypes, dim=-1),
        ) / temperature
        weights = torch.softmax(scores, dim=-1)
        semantic = torch.einsum("bp,bpd->bd", weights, prototypes)
        context = torch.cat(
            [physical_state, semantic, physical_state * semantic], dim=-1,
        )
        # Only relative logits affect a softmax router.  Removing the common
        # component prevents the semantic adapter from spending capacity on an
        # unidentifiable all-routes bias.  Tanh keeps the frozen-GPT residual
        # prior bounded while the outer semantic strength is calibrated.
        tf_delta = self.tf_delta(context)
        scale_delta = self.scale_delta(context)
        if self.centered_route_delta:
            tf_delta = torch.tanh(
                tf_delta - tf_delta.mean(dim=-1, keepdim=True)
            )
            scale_delta = torch.tanh(
                scale_delta - scale_delta.mean(dim=-1, keepdim=True)
            )

        # A weak content-pairing objective makes the adapter identify the
        # correct historical-state/prompt relationship.  Exact duplicate
        # prompt banks in a batch are treated as additional positives, avoiding
        # the false-negative problem of ordinary sample-index InfoNCE.
        if self.content_pairing:
            query_norm = F.normalize(query, dim=-1)
            semantic_norm = F.normalize(semantic, dim=-1)
            logits = query_norm @ semantic_norm.transpose(0, 1) / 0.10
            with torch.no_grad():
                signatures = F.normalize(
                    prompt.to(torch.float32).flatten(start_dim=1), dim=-1,
                )
                positive_mask = signatures @ signatures.transpose(0, 1) > 0.9999
                positive_mask.fill_diagonal_(True)
            numerator = torch.logsumexp(
                logits.masked_fill(~positive_mask, float("-inf")), dim=-1,
            )
            denominator = torch.logsumexp(logits, dim=-1)
            self.pairing_loss = (denominator - numerator).mean()
        else:
            self.pairing_loss = None
        self.last_prototype_weights = weights.detach()
        self.last_tf_delta = tf_delta.detach()
        self.last_scale_delta = scale_delta.detach()
        return tf_delta, scale_delta


class RPGTR(nn.Module):
    def __init__(self, seq_len: int, horizon: int, channels: int, cycle_len: int, patch_len: int, stride: int,
                 d_model: int, encoder: str, mode: str, revin: bool = False,
                 physics_indices: Optional[Sequence[int]] = None,
                 quality_indices: Optional[Sequence[int]] = None,
                 output_anchor: bool = False,
                 recent_physics: bool = False,
                 recent_only_physics: bool = False,
                 single_scale: bool = False,
                 disable_gtr: bool = False,
                 disable_ldrive: bool = False,
                 disable_frequency: bool = False,
                 clean_frequency_ablation: bool = False,
                 static_output_anchor: bool = False,
                 null_route: bool = False,
                 future_covariates: bool = False,
                 nwp_conditioned_decoder: bool = False,
                 future_channels: int = 0,
                 frequency_indices: Optional[Sequence[int]] = None,
                 physical_scales: Optional[Sequence[tuple[int, int]]] = None,
                 permutation_correlation: bool = False,
                 solar_anchor_config: Optional[Dict] = None,
                 reference_ensemble: bool = False,
                 daily_shape_anchor: bool = False,
                 sequential_physics: bool = False,
                 state_conditioned_spectrum: bool = False,
                 solar_residual_output: bool = False,
                 physical_band_spectrum: bool = False,
                 sample_hours: Optional[float] = None,
                 state_auxiliary: bool = False,
                 state_conditioned_bands: bool = True,
                 spectral_disturbance: Optional[bool] = None,
                 semantic_prompt_dim: int = 0,
                 adaptive_semantic_alignment: bool = False,
                 semantic_output_calibration: bool = False,
                 prompt_only_semantic_output: bool = False,
                 zero_safe_semantic: bool = False,
                 semantic_cma_heads: int = 0,
                 semantic_contrastive_alignment: bool = True,
                 semantic_prompt_encoder_layers: int = 0,
                 semantic_channel_residual: bool = False,
                 channel_semantic_embeddings: Optional[torch.Tensor] = None,
                 channel_semantic_trainable: bool = False,
                 fsra_residual: bool = False,
                 fsra_relation_bias: bool = False,
                 fsra_channel_alignment: bool = False,
                 fsra_cma_heads: int = 0,
                 fsra_transferable: bool = False,
                 fsra_transfer_bins: int = 0,
                 fsra_paired_alignment: bool = False,
                 probabilistic_output: bool = False,
                 probabilistic_state: bool = True,
                 quantile_output: bool = False,
                 quantile_state: bool = True,
                 quantile_members: int = 19,
                 output_correction_floor: float = 0.0,
                  output_router_bias: Optional[float] = None,
                  local_token_refiner: bool = False,
                  local_refiner_dropout: float = 0.05,
                  scale_token_mixer: bool = False,
                  scale_token_mixer_adaptive: bool = True,
                  scale_token_mixer_dropout: float = 0.05,
                  scale_prompt_dim: int = 0,
                  scale_prompt_heads: int = 3,
                 scale_prompt_output_gate: bool = True,
                 reference_residual_gate: bool = False,
                 frequency_v2_mode: Optional[str] = None,
                 frequency_v3_mode: Optional[str] = None,
                 frequency_v4_mode: Optional[str] = None,
                 frequency_v5_mode: Optional[str] = None,
                 frequency_v6_mode: Optional[str] = None,
                 frequency_v7_mode: Optional[str] = None,
                 frequency_v8_mode: Optional[str] = None,
                 frequency_v9_mode: Optional[str] = None,
                 spectral_mkan_mode: str = "legacy",
                 router_activation: str = "gelu",
                 expert_drop_path: float = 0.0,
                 gtr_dropout: float = 0.10):
        super().__init__()
        self.channels, self.cycle_len = channels, cycle_len
        self.prompt_d_model = int(d_model)
        self.prompt_horizon = int(horizon)
        self.mode = mode
        self.revin = revin
        self.output_anchor = output_anchor
        self.output_correction_floor = float(output_correction_floor)
        if not 0.0 <= self.output_correction_floor < 1.0:
            raise ValueError("output_correction_floor must be in [0, 1)")
        self.output_router_bias = (None if output_router_bias is None
                                   else float(output_router_bias))
        self.output_correction_scale = 1.0
        self.local_token_refiner = bool(local_token_refiner)
        self.scale_token_mixer_enabled = bool(scale_token_mixer)
        self.scale_prompt_dim = int(scale_prompt_dim)
        self.scale_prompt_heads = int(scale_prompt_heads)
        self.scale_prompt_output_gate = bool(scale_prompt_output_gate)
        self.scale_prompt_enabled = self.scale_prompt_dim > 0
        self.reference_residual_enabled = bool(reference_residual_gate)
        if router_activation not in {"gelu", "relu"}:
            raise ValueError("router activation must be gelu or relu")
        self.router_activation = router_activation
        router_act = nn.GELU if router_activation == "gelu" else nn.ReLU
        self.expert_drop_path = float(expert_drop_path)
        if not 0.0 <= self.expert_drop_path < 1.0:
            raise ValueError("expert drop path must be in [0, 1)")
        self.gtr_dropout = float(gtr_dropout)
        if not 0.0 <= self.gtr_dropout < 1.0:
            raise ValueError("GTR dropout must be in [0, 1)")
        self.uses_prompt_features = (
            int(semantic_prompt_dim) > 0 or self.scale_prompt_enabled
        )
        self.recent_only_physics = recent_only_physics
        self.disable_gtr = disable_gtr
        self.disable_ldrive = disable_ldrive
        self.disable_frequency = bool(disable_frequency)
        self.clean_frequency_ablation = bool(clean_frequency_ablation)
        if self.disable_frequency and self.clean_frequency_ablation:
            raise ValueError(
                "legacy spectral-zero and clean frequency ablation are mutually exclusive"
            )
        if spectral_mkan_mode not in SpectralMKAN.MODES:
            raise ValueError(f"unknown SpectralMKAN mode: {spectral_mkan_mode}")
        self.spectral_mkan_mode = spectral_mkan_mode
        self.frequency_v2_mode = frequency_v2_mode
        if (self.frequency_v2_mode is not None
                and self.frequency_v2_mode
                not in SelectiveDualBandSpectralResidual.MODES):
            raise ValueError(
                f"unknown frequency-v2 mode: {self.frequency_v2_mode}"
            )
        self.frequency_v3_mode = frequency_v3_mode
        if (self.frequency_v3_mode is not None
                and self.frequency_v3_mode
                not in EndogenousSpectralTargetGate.MODES):
            raise ValueError(
                f"unknown frequency-v3 mode: {self.frequency_v3_mode}"
            )
        if (self.frequency_v2_mode is not None
                and self.frequency_v3_mode is not None):
            raise ValueError("frequency-v2 and frequency-v3 are mutually exclusive")
        self.frequency_v4_mode = frequency_v4_mode
        if (self.frequency_v4_mode is not None
                and self.frequency_v4_mode
                not in OriginalPositionDualScaleSpectralResidual.MODES):
            raise ValueError(
                f"unknown frequency-v4 mode: {self.frequency_v4_mode}"
            )
        self.frequency_v5_mode = frequency_v5_mode
        if (self.frequency_v5_mode is not None
                and self.frequency_v5_mode
                not in AbstentionGatedSpectralMKAN.MODES):
            raise ValueError(
                f"unknown frequency-v5 mode: {self.frequency_v5_mode}"
            )
        self.frequency_v6_mode = frequency_v6_mode
        if (self.frequency_v6_mode is not None
                and self.frequency_v6_mode
                not in EndogenousHierarchicalCalibratedSpectralMKAN.MODES):
            raise ValueError(
                f"unknown frequency-v6 mode: {self.frequency_v6_mode}"
            )
        self.frequency_v7_mode = frequency_v7_mode
        if (self.frequency_v7_mode is not None
                and self.frequency_v7_mode
                not in PredictabilityGatedLocalSpectralResidual.MODES):
            raise ValueError(
                f"unknown frequency-v7 mode: {self.frequency_v7_mode}"
            )
        self.frequency_v8_mode = frequency_v8_mode
        if (self.frequency_v8_mode is not None
                and self.frequency_v8_mode
                not in RobustGlobalMultivariateSpectralMKAN.MODES):
            raise ValueError(
                f"unknown frequency-v8 mode: {self.frequency_v8_mode}"
            )
        self.frequency_v9_mode = frequency_v9_mode
        if (self.frequency_v9_mode is not None
                and self.frequency_v9_mode
                not in EndogenousTrendSpectralAdapter.MODES):
            raise ValueError(
                f"unknown frequency-v9 mode: {self.frequency_v9_mode}"
            )
        if sum(mode is not None for mode in (
                self.frequency_v2_mode, self.frequency_v3_mode,
                self.frequency_v4_mode, self.frequency_v5_mode,
                self.frequency_v6_mode, self.frequency_v7_mode,
                self.frequency_v8_mode, self.frequency_v9_mode)) > 1:
            raise ValueError(
                "frequency-v2 through v9 are mutually exclusive"
            )
        self.static_output_anchor = static_output_anchor
        self.null_route = null_route
        self.future_covariates = future_covariates
        self.nwp_conditioned_decoder = nwp_conditioned_decoder
        self.spectral_disturbance = (physical_band_spectrum if spectral_disturbance is None
                                     else spectral_disturbance)
        self.solar_anchor_config = dict(solar_anchor_config or {})
        self.solar_trajectory_anchor = bool(self.solar_anchor_config)
        if self.reference_residual_enabled and not self.solar_trajectory_anchor:
            raise ValueError(
                "reference residual gate requires deterministic solar geometry"
            )
        self.reference_ensemble = bool(reference_ensemble)
        self.daily_shape_anchor = bool(daily_shape_anchor)
        self.semantic_prompt_dim = int(semantic_prompt_dim)
        self.adaptive_semantic_alignment = bool(adaptive_semantic_alignment)
        self.semantic_output_calibration = bool(semantic_output_calibration)
        self.prompt_only_semantic_output = bool(prompt_only_semantic_output)
        self.zero_safe_semantic = bool(zero_safe_semantic)
        self.semantic_cma_heads = max(0, int(semantic_cma_heads))
        self.semantic_contrastive_alignment = bool(semantic_contrastive_alignment)
        self.semantic_prompt_encoder_layers = max(
            0, int(semantic_prompt_encoder_layers)
        )
        self.semantic_channel_residual = bool(semantic_channel_residual)
        self.fsra_residual_enabled = bool(fsra_residual)
        self.fsra_relation_enabled = bool(fsra_relation_bias)
        self.fsra_channel_alignment = bool(fsra_channel_alignment)
        self.fsra_cma_heads = max(0, int(fsra_cma_heads))
        self.fsra_transferable = bool(fsra_transferable)
        self.fsra_transfer_bins = max(0, int(fsra_transfer_bins))
        self.fsra_paired_alignment = bool(fsra_paired_alignment)
        self.probabilistic_output = bool(probabilistic_output)
        self.probabilistic_state = bool(probabilistic_state)
        self.quantile_output = bool(quantile_output)
        self.quantile_state = bool(quantile_state)
        self.quantile_members = int(quantile_members)
        if self.fsra_transfer_bins and not self.fsra_transferable:
            raise ValueError("fixed transfer bins require fsra_transferable=True")
        if self.reference_ensemble and not self.solar_trajectory_anchor:
            raise ValueError("reference ensemble requires deterministic solar geometry")
        if self.daily_shape_anchor and not self.solar_trajectory_anchor:
            raise ValueError("daily-shape anchor requires the audited anchor path")
        self.reference_logits = (nn.Parameter(torch.zeros(3))
                                 if self.reference_ensemble else None)
        self.last_reference_weights = None
        # Prompt modules are optional adapters. Preserve RNG state around their
        # construction so every non-prompt parameter has exactly the same
        # initialization as the PCARR-v3 backbone under an identical seed.
        semantic_rng = torch.random.get_rng_state()
        semantic_cuda_rng = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        if (self.semantic_prompt_dim > 0 and self.adaptive_semantic_alignment
                and self.zero_safe_semantic):
            # A zero prompt must be the exact numerical backbone, not a
            # trainable constant calibration path.  Bias-free projections make
            # the counterfactual identifiable: only prompt content can create a
            # residual, while shuffled prompts retain the same capacity.
            hidden = max(64, 2 * d_model)
            heads = 4 if d_model % 4 == 0 else 1
            self.semantic_projection = nn.Sequential(
                nn.LayerNorm(self.semantic_prompt_dim, elementwise_affine=False),
                nn.Linear(self.semantic_prompt_dim, hidden, bias=False),
                nn.GELU(), nn.Dropout(0.10),
                nn.Linear(hidden, d_model, bias=False),
            )
            self.semantic_query = nn.Linear(channels, d_model, bias=False)
            self.semantic_attention = nn.MultiheadAttention(
                d_model, heads, dropout=0.10, bias=False, batch_first=True,
            )
            self.semantic_ffn = nn.Sequential(
                nn.LayerNorm(d_model, elementwise_affine=False),
                nn.Linear(d_model, 2 * d_model, bias=False), nn.GELU(),
                nn.Dropout(0.10), nn.Linear(2 * d_model, d_model, bias=False),
            )
            self.semantic_back = nn.Linear(d_model, channels, bias=False)
            self.semantic_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, channels),
            )
            nn.init.normal_(self.semantic_back.weight, std=1e-3)
            nn.init.zeros_(self.semantic_gate[-1].weight)
            nn.init.constant_(self.semantic_gate[-1].bias, -1.5)
            self.semantic_dropout = nn.Dropout(0.10)
            if self.semantic_output_calibration:
                output_context_dim = d_model if self.prompt_only_semantic_output else 2 * d_model
                self.semantic_output_head = nn.Sequential(
                    nn.LayerNorm(output_context_dim, elementwise_affine=False),
                    nn.Linear(output_context_dim, d_model, bias=False), nn.GELU(),
                    nn.Dropout(0.10), nn.Linear(d_model, horizon, bias=False),
                )
                self.semantic_output_gate = nn.Linear(output_context_dim, horizon, bias=False)
                nn.init.normal_(self.semantic_output_head[-1].weight, std=1e-3)
                nn.init.zeros_(self.semantic_output_gate.weight)
            else:
                self.semantic_output_head = None
                self.semantic_output_gate = None
        elif self.semantic_prompt_dim > 0 and self.adaptive_semantic_alignment:
            hidden = max(64, 2 * d_model)
            heads = 4 if d_model % 4 == 0 else 1
            self.semantic_projection = nn.Sequential(
                nn.LayerNorm(self.semantic_prompt_dim),
                nn.Linear(self.semantic_prompt_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, d_model),
            )
            self.semantic_query = nn.Linear(channels, d_model)
            self.semantic_attention = nn.MultiheadAttention(
                d_model, heads, dropout=0.10, batch_first=True,
            )
            self.semantic_ffn = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                nn.GELU(), nn.Dropout(0.10), nn.Linear(2 * d_model, d_model),
            )
            self.semantic_back = nn.Linear(d_model, channels)
            self.semantic_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, channels),
            )
            nn.init.normal_(self.semantic_back.weight, std=1e-3)
            nn.init.zeros_(self.semantic_back.bias)
            nn.init.zeros_(self.semantic_gate[-1].weight)
            nn.init.constant_(self.semantic_gate[-1].bias, -1.5)
            self.semantic_dropout = nn.Dropout(0.10)
            if self.semantic_output_calibration:
                output_context_dim = d_model if self.prompt_only_semantic_output else 2 * d_model
                self.semantic_output_head = nn.Sequential(
                    nn.LayerNorm(output_context_dim), nn.Linear(output_context_dim, d_model),
                    nn.GELU(), nn.Dropout(0.10), nn.Linear(d_model, horizon),
                )
                self.semantic_output_gate = nn.Linear(output_context_dim, horizon)
                nn.init.normal_(self.semantic_output_head[-1].weight, std=1e-3)
                nn.init.zeros_(self.semantic_output_head[-1].bias)
                nn.init.zeros_(self.semantic_output_gate.weight)
                nn.init.constant_(self.semantic_output_gate.bias, -2.0)
            else:
                self.semantic_output_head = None
                self.semantic_output_gate = None
        elif self.semantic_prompt_dim > 0:
            # The frozen LLM is offline.  Only this compact residual alignment
            # layer is trainable.  Single-head attention avoids pretending the
            # small number of semantic tokens warrants a large attention bank.
            self.semantic_projection = nn.Linear(self.semantic_prompt_dim, channels)
            self.semantic_attention = nn.MultiheadAttention(channels, 1, batch_first=True)
            self.semantic_gate = nn.Linear(d_model, channels)
            nn.init.zeros_(self.semantic_gate.weight)
            nn.init.constant_(self.semantic_gate.bias, -1.0)
            # An exactly-zero output projection postpones gradients into the
            # prompt projection and made the five-epoch screening run ignore
            # semantics.  A tiny residual is still conservative but trainable
            # from the first update.
            nn.init.normal_(self.semantic_attention.out_proj.weight, std=1e-3)
            nn.init.zeros_(self.semantic_attention.out_proj.bias)
        else:
            self.semantic_projection = None
            self.semantic_attention = None
            self.semantic_gate = None
            self.semantic_output_head = None
            self.semantic_output_gate = None
        if self.semantic_projection is not None and self.semantic_cma_heads > 0:
            # T3Time uses several independently learned cross-modal alignment
            # blocks and then routes across their outputs.  This is different
            # from merely splitting one attention operation into internal
            # heads.  Keeping every projection bias-free preserves the exact
            # zero-prompt identity required by the causal ablation.
            self.semantic_attention = None
            self.semantic_cma = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model, 1, dropout=0.10, bias=False, batch_first=True,
                )
                for _ in range(self.semantic_cma_heads)
            ])
            self.semantic_head_router = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, self.semantic_cma_heads),
            )
            nn.init.zeros_(self.semantic_head_router[-1].weight)
            nn.init.zeros_(self.semantic_head_router[-1].bias)
        else:
            self.semantic_cma = None
            self.semantic_head_router = None
        if (self.semantic_projection is not None
                and self.semantic_prompt_encoder_layers > 0):
            # T3Time-style trainable prompt encoder.  GPT-2 stays completely
            # frozen/offline; this compact encoder alone adapts its final-token
            # vectors to the numerical forecasting space.
            prompt_heads = 4 if d_model % 4 == 0 else 1
            prompt_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=prompt_heads,
                dim_feedforward=2 * d_model,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            )
            self.semantic_prompt_encoder = nn.TransformerEncoder(
                prompt_layer,
                num_layers=self.semantic_prompt_encoder_layers,
                norm=nn.LayerNorm(d_model, elementwise_affine=False),
            )
        else:
            self.semantic_prompt_encoder = None
        if (self.semantic_projection is not None
                and self.semantic_channel_residual):
            # Channel-wise residual coefficient from T3Time.  A small initial
            # value keeps the established numerical path dominant, while
            # end-to-end gradients can increase only the channels that benefit
            # from language-conditioned alignment.
            initial_mix = 0.10
            initial_logit = math.log(initial_mix / (1.0 - initial_mix))
            self.semantic_channel_logit = nn.Parameter(
                torch.full((channels,), initial_logit)
            )
        else:
            self.semantic_channel_logit = None
        if not self.adaptive_semantic_alignment:
            self.semantic_query = None
            self.semantic_ffn = None
            self.semantic_back = None
            self.semantic_dropout = None
        self.last_semantic_gate = None
        self.last_semantic_channel_mix = None
        self.last_semantic_output_gate = None
        self.last_semantic_head_weights = None
        self.semantic_alignment_loss = None
        self.last_scale_prompt_gate = None
        self.last_scale_prompt_head_weights = None
        self.last_scale_prompt_state = None
        self.last_scale_prompt_match_accuracy = None
        self.semantic_tf_prompt_router = None
        self.semantic_hierarchical_router = None
        self.last_semantic_tf_route_delta = None
        self.last_semantic_scale_route_delta = None
        self.last_semantic_prototype_weights = None
        self.semantic_enabled = True
        # Validation-calibrated scalar; zero recovers the exact PCARR-v3
        # backbone and one applies the full prompt correction.
        self.semantic_strength = 1.0
        torch.random.set_rng_state(semantic_rng)
        if semantic_cuda_rng is not None:
            torch.cuda.set_rng_state_all(semantic_cuda_rng)
        # Frozen Semantic Relation Adapter (FSRA).  The text encoder runs
        # offline; this module receives only one fixed vector per channel.
        # Preserve RNG state so correct/shuffle/zero/random controls keep an
        # exactly matched numerical backbone initialization.
        fsra_rng = torch.random.get_rng_state()
        fsra_cuda_rng = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        if fsra_transferable:
            # Cross-site checkpoint aggregation requires head h to start from
            # the same semantic coordinate on every site.  Backbone dimensions
            # consume different amounts of RNG, so decouple transferable FSRA
            # initialization from the site-specific construction history.
            torch.manual_seed(271828)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(271828)
        if channel_semantic_embeddings is not None:
            channel_semantic_embeddings = torch.as_tensor(
                channel_semantic_embeddings, dtype=torch.float32,
            )
            if channel_semantic_embeddings.ndim != 2:
                raise ValueError("channel semantic embeddings must have shape [C,E]")
            if channel_semantic_embeddings.shape[0] != channels:
                raise ValueError(
                    "channel semantic embedding count must equal model channels"
                )
            if channel_semantic_trainable:
                self.fsra_channel_embeddings = nn.Parameter(
                    channel_semantic_embeddings.clone(), requires_grad=True,
                )
            else:
                self.register_buffer(
                    "fsra_channel_embeddings", channel_semantic_embeddings.clone(),
                    persistent=True,
                )
            semantic_dim = int(channel_semantic_embeddings.shape[1])
            heads = 4 if d_model % 4 == 0 else 1
            self.fsra_semantic_projection = nn.Sequential(
                nn.LayerNorm(semantic_dim, elementwise_affine=False),
                nn.Linear(semantic_dim, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
            if self.fsra_transfer_bins:
                # A fixed solar-day phase grid preserves intraday ramps and
                # cloud transitions while remaining compatible with 5-min and
                # 15-min histories.  All one-day windows are aligned to the
                # same number of physical phase slots before semantic fusion.
                self.fsra_numeric_projection = nn.Sequential(
                    nn.LayerNorm(self.fsra_transfer_bins, elementwise_affine=False),
                    nn.Linear(self.fsra_transfer_bins, d_model, bias=False),
                    nn.GELU(), nn.Linear(d_model, d_model, bias=False),
                )
            elif self.fsra_transferable:
                # Resolution-independent numerical query.  These summaries are
                # computed from the normalized historical representation and
                # have the same meaning for 5-min and 15-min sites.
                self.fsra_numeric_projection = nn.Sequential(
                    nn.LayerNorm(6, elementwise_affine=False),
                    nn.Linear(6, d_model, bias=False), nn.GELU(),
                    nn.Linear(d_model, d_model, bias=False),
                )
            else:
                self.fsra_numeric_projection = nn.Linear(seq_len, d_model, bias=False)
            if self.fsra_cma_heads > 0:
                # T3-inspired independent cross-modal aligners.  These are
                # genuinely separate attention blocks, not internal heads of
                # one MHA.  The router remains inside a zero-safe residual
                # adapter, so it cannot replace the numerical PV backbone.
                self.fsra_attention = None
                if self.fsra_paired_alignment:
                    # Channel-bound alignment fixes a fundamental ambiguity of
                    # set attention: jointly permuting K and V leaves its
                    # output unchanged.  Each head now sees the channel's own
                    # frozen semantic vector and its multiplicative agreement
                    # with the numerical token.  Bias-free layers make an
                    # all-zero semantic control an exact backbone identity.
                    self.fsra_cma = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(2 * d_model, d_model, bias=False),
                            nn.GELU(), nn.Dropout(0.10),
                            nn.Linear(d_model, d_model, bias=False),
                        )
                        for _ in range(self.fsra_cma_heads)
                    ])
                else:
                    self.fsra_cma = nn.ModuleList([
                        nn.MultiheadAttention(
                            d_model, 1, dropout=0.10, bias=False, batch_first=True,
                        )
                        for _ in range(self.fsra_cma_heads)
                    ])
                self.fsra_head_router = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.GELU(),
                    nn.Dropout(0.10), nn.Linear(d_model, self.fsra_cma_heads),
                )
                nn.init.zeros_(self.fsra_head_router[-1].weight)
                nn.init.zeros_(self.fsra_head_router[-1].bias)
            else:
                self.fsra_attention = nn.MultiheadAttention(
                    d_model, heads, dropout=0.10, bias=False, batch_first=True,
                )
                self.fsra_cma = None
                self.fsra_head_router = None
            self.fsra_back = nn.Linear(
                d_model,
                (self.fsra_transfer_bins if self.fsra_transfer_bins
                 else (4 if self.fsra_transferable else seq_len)),
                bias=False,
            )
            self.fsra_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 1),
            )
            self.fsra_relation_logit = nn.Parameter(torch.tensor(-2.0))
            nn.init.normal_(
                self.fsra_back.weight,
                std=(5e-3 if self.fsra_cma_heads > 0 else 1e-3),
            )
            nn.init.zeros_(self.fsra_gate[-1].weight)
            nn.init.constant_(
                self.fsra_gate[-1].bias,
                (-2.5 if self.fsra_cma_heads > 0 else -4.0),
            )
        else:
            self.fsra_channel_embeddings = None
            self.fsra_semantic_projection = None
            self.fsra_numeric_projection = None
            self.fsra_attention = None
            self.fsra_cma = None
            self.fsra_head_router = None
            self.fsra_back = None
            self.fsra_gate = None
            self.fsra_relation_logit = None
        self.fsra_strength = 1.0
        self.fsra_enabled = channel_semantic_embeddings is not None
        self.fsra_has_content = bool(
            channel_semantic_embeddings is not None
            and torch.as_tensor(channel_semantic_embeddings).abs().sum().item() > 0.0
        )
        self.fsra_alignment_loss = None
        self.last_fsra_gate = None
        self.last_fsra_relation_scale = None
        self.last_fsra_head_weights = None
        self.fsra_head_specialization_loss = None
        torch.random.set_rng_state(fsra_rng)
        if fsra_cuda_rng is not None:
            torch.cuda.set_rng_state_all(fsra_cuda_rng)
        self.solar_residual_output = solar_residual_output
        if solar_residual_output and not self.solar_trajectory_anchor:
            raise ValueError("solar residual output requires a deterministic solar anchor")
        if future_covariates and nwp_conditioned_decoder:
            raise ValueError("legacy future residual and NWP-conditioned decoder are mutually exclusive")
        quality_idx = sorted(set(int(i) for i in (quality_indices or []) if 0 <= int(i) < channels))
        self.register_buffer("quality_indices", torch.tensor(quality_idx, dtype=torch.long), persistent=False)
        spectral_idx = sorted(set(int(i) for i in (frequency_indices or range(channels))
                                  if 0 <= int(i) < channels))
        if not spectral_idx:
            spectral_idx = list(range(channels))
        self.register_buffer("frequency_indices", torch.tensor(spectral_idx, dtype=torch.long), persistent=False)
        self.scale_aware = mode in {
            "parallel_ms_phys", "tf_parallel_ms", "tf_parallel_ms_static",
            "tf_residual_ms", "tf_residual_ms_static",
        }
        self.time_frequency = mode in {
            "tf_parallel_ms", "tf_parallel_ms_static", "tf_residual_ms", "tf_residual_ms_static",
        }
        self.residual_frequency = mode.startswith("tf_residual")
        self.norm = nn.LayerNorm(channels)
        self.gtr = None if disable_gtr else GTR(
            seq_len, channels, dropout=self.gtr_dropout,
        )
        self.mkan = MKAN(channels)
        self.lcontext = None if disable_ldrive else LContext(channels)
        self.frequency_v2 = (
            SelectiveDualBandSpectralResidual(
                len(spectral_idx), d_model, float(sample_hours), horizon,
                self.frequency_v2_mode,
            )
            if self.time_frequency and self.frequency_v2_mode is not None
            else None
        )
        target_spectral_position = (
            spectral_idx.index(channels - 1)
            if (channels - 1) in spectral_idx else None
        )
        if self.frequency_v4_mode is not None and target_spectral_position is None:
            raise ValueError("frequency-v4 requires the historical Target channel")
        if self.frequency_v6_mode is not None and target_spectral_position is None:
            raise ValueError("frequency-v6 requires the historical Target channel")
        if self.frequency_v7_mode is not None and target_spectral_position is None:
            raise ValueError("frequency-v7 requires the historical Target channel")
        if self.frequency_v9_mode is not None and target_spectral_position is None:
            raise ValueError("frequency-v9 requires the historical Target channel")
        self.frequency_v4 = (
            OriginalPositionDualScaleSpectralResidual(
                len(spectral_idx), int(target_spectral_position), d_model,
                float(sample_hours), horizon, self.frequency_v4_mode,
            )
            if self.time_frequency and self.frequency_v4_mode is not None
            else None
        )
        self.frequency_v5 = (
            AbstentionGatedSpectralMKAN(
                len(spectral_idx), d_model, float(sample_hours), horizon,
                self.frequency_v5_mode,
            )
            if self.time_frequency and self.frequency_v5_mode is not None
            else None
        )
        self.frequency_v6 = (
            EndogenousHierarchicalCalibratedSpectralMKAN(
                len(spectral_idx), int(target_spectral_position), d_model,
                float(sample_hours), horizon, self.frequency_v6_mode,
            )
            if self.time_frequency and self.frequency_v6_mode is not None
            else None
        )
        self.frequency_v7 = (
            PredictabilityGatedLocalSpectralResidual(
                len(spectral_idx), int(target_spectral_position), d_model,
                float(sample_hours), horizon, self.frequency_v7_mode,
            )
            if self.time_frequency and self.frequency_v7_mode is not None
            else None
        )
        self.frequency_v8 = (
            RobustGlobalMultivariateSpectralMKAN(
                len(spectral_idx), d_model, float(sample_hours), horizon,
                self.frequency_v8_mode,
            )
            if self.time_frequency and self.frequency_v8_mode is not None
            else None
        )
        self.frequency_v9 = (
            EndogenousTrendSpectralAdapter(
                len(spectral_idx), int(target_spectral_position), d_model,
                float(sample_hours), horizon, self.frequency_v9_mode,
            )
            if self.time_frequency and self.frequency_v9_mode is not None
            else None
        )
        self.spectral_mkan = (SpectralMKAN(
            len(spectral_idx), state_dim=(d_model if state_conditioned_spectrum else None),
            sample_hours=sample_hours, physical_bands=physical_band_spectrum,
            state_conditioned_bands=state_conditioned_bands,
            router_activation=router_activation,
            mode=self.spectral_mkan_mode,
        ) if (self.time_frequency and self.frequency_v2 is None
              and self.frequency_v3_mode is None
              and self.frequency_v4 is None
              and self.frequency_v5 is None
              and self.frequency_v6 is None
              and self.frequency_v7 is None
              and self.frequency_v8 is None
              and self.frequency_v9 is None) else None)
        # These legacy modules are retained in the state dictionary so that
        # the verified forward graph and seeded initialization stay exactly
        # reproducible, but they are unreachable in the time-frequency path.
        # Freezing them makes trainable-parameter and optimizer-cost reporting
        # honest without changing any prediction.
        if self.time_frequency:
            for module in (self.mkan,):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        # M5 initializes the global cycle memory to zero; random memory injects
        # a large untrained signal into the first epochs of the new branches.
        self.q_memory = nn.Parameter(torch.zeros(cycle_len, channels), requires_grad=not disable_gtr)
        self.gate = nn.Linear(3 * channels, 3)
        self.gate2 = nn.Linear(2 * channels, 2)
        self.branch_norm = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.branch_scale = nn.Parameter(torch.ones(3) * 0.5)
        if self.time_frequency:
            for module in (self.gate, self.gate2):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.branch_scale.requires_grad_(False)
            for norm in self.branch_norm[1:]:
                for parameter in norm.parameters():
                    parameter.requires_grad_(False)
        # The optional third logit is a null residual route.  It lets the
        # model preserve the identity path when neither GTR nor L-Drive is
        # appropriate for the current physical state, instead of forcing
        # 100% of the mass onto potentially harmful increments.
        self.temporal_gate = (nn.Linear(2 * channels, 3 if null_route else 2)
                              if self.time_frequency and not disable_gtr and not disable_ldrive else None)
        self.tf_branch_scale = nn.Parameter(torch.ones(2) * 0.5) if self.time_frequency else None
        self.temporal_drop_path = ResidualDropPath(self.expert_drop_path)
        self.spectral_drop_path = ResidualDropPath(self.expert_drop_path)
        if self.scale_aware:
            if sequential_physics:
                encoder_cls = SequentialPhysicalStateEncoder
            else:
                encoder_cls = MultiHorizonPhysicalStateEncoder if recent_physics else PhysicalStateEncoder
            self.phys_encoder = encoder_cls(channels, physics_indices, d_model)
            self.state_classifier = nn.Linear(d_model, 4) if state_auxiliary else None
        else:
            self.phys_encoder = None
            self.state_classifier = None
        # A small learnable absolute-state anchor complements the relative
        # RevIN shape without letting seasonal level shifts dominate routing.
        self.abs_route_logit = nn.Parameter(torch.tensor(-2.0)) if self.scale_aware else None
        if self.scale_aware:
            self.patch = ScaleStackPatch(
                patch_len, stride, d_model, seq_len, single_scale=single_scale,
                physical_scales=physical_scales,
            )
            # Optional refiners must not perturb the validated backbone's
            # seeded initialization.  Restore RNG after constructing it so a
            # zero residual scale gives exact with/without-module parity.
            token_mixer_rng = torch.random.get_rng_state()
            token_mixer_cuda_rng = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.scale_token_refiner = (
                ScaleWiseDilatedTokenMixer(
                    channels, len(self.patch.scales), d_model,
                    dropout=float(scale_token_mixer_dropout),
                    adaptive_aggregation=bool(scale_token_mixer_adaptive),
                )
                if self.scale_token_mixer_enabled else None
            )
            torch.random.set_rng_state(token_mixer_rng)
            if token_mixer_cuda_rng is not None:
                torch.cuda.set_rng_state_all(token_mixer_cuda_rng)
            prompt_rng = torch.random.get_rng_state()
            prompt_cuda_rng = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.scale_prompt_adapter = (
                ScaleMatchedPromptAdapter(
                    self.scale_prompt_dim, d_model, len(self.patch.scales),
                    n_heads=self.scale_prompt_heads,
                )
                if self.scale_prompt_enabled else None
            )
            self.scale_prompt_output_router = (
                nn.Sequential(
                    nn.LayerNorm(2 * d_model, elementwise_affine=False),
                    nn.Linear(2 * d_model, d_model, bias=False), nn.GELU(),
                    nn.Linear(d_model, horizon, bias=False),
                )
                if self.scale_prompt_enabled and self.scale_prompt_output_gate else None
            )
            if self.scale_prompt_output_router is not None:
                nn.init.normal_(self.scale_prompt_output_router[-1].weight, std=1e-3)
            # Optional prompt modules must not change initialization of the
            # verified numerical backbone when their residual is disabled.
            torch.random.set_rng_state(prompt_rng)
            if prompt_cuda_rng is not None:
                torch.cuda.set_rng_state_all(prompt_cuda_rng)
            self.scale_router = nn.Sequential(
                nn.Linear(d_model, d_model), router_act(),
                nn.Linear(d_model, len(self.patch.scales)),
            )
            self.scale_logits = nn.Parameter(torch.zeros(len(self.patch.scales)))
            tf_routes = 3 if null_route else 2
            self.tf_router = (nn.Sequential(
                nn.Linear(d_model, d_model), router_act(), nn.Linear(d_model, tf_routes),
            )
                              if self.time_frequency else None)
            if self.residual_frequency:
                tf_init = torch.tensor([0.5, -0.5, 0.0]) if null_route else torch.tensor([0.5, -0.5])
            else:
                tf_init = torch.zeros(tf_routes)
            self.tf_logits = nn.Parameter(tf_init) if self.time_frequency else None
            self.register_buffer("tf_prior", tf_init.clone(), persistent=False)
            npatch = self.patch.n_tokens
            revised_corr = permutation_correlation or encoder == "variable"
            self.encoder = ScaleAwareCorPatchEncoder(
                channels, len(self.patch.scales), d_model, encoder,
                pre_scale_bias=not revised_corr,
            )
        else:
            self.patch = MultiScalePatch(patch_len, stride, d_model, seq_len) if mode.endswith("_ms") else PatchEmbed(patch_len, stride, d_model)
            self.scale_token_refiner = None
            self.scale_prompt_adapter = None
            self.scale_prompt_output_router = None
            npatch = self.patch.n_tokens if isinstance(self.patch, MultiScalePatch) else int((seq_len + stride - patch_len) / stride + 1)
            self.encoder = CoherenceEncoder(channels, d_model) if encoder == "coherence" else FACTEncoder(channels, d_model)
        self.frequency_v3 = (
            EndogenousSpectralTargetGate(
                d_model=d_model, n_tokens=npatch, mode=self.frequency_v3_mode,
            )
            if self.frequency_v3_mode is not None else None
        )
        # Always preserve the original historical forecasting head.  The NWP
        # adapter below may add a correction but never replaces this head.
        self.head = nn.Linear(d_model * npatch, horizon)
        self.local_refiner = (
            TokenLocalResidualRefiner(
                channels, d_model, dropout=float(local_refiner_dropout),
            )
            if self.local_token_refiner else None
        )
        # GenSolar-inspired two-stage uncertainty interface.  The deterministic
        # forecast remains the mean; this head is fitted only after the mean
        # model is frozen and therefore cannot trade point accuracy for wider
        # intervals.  RNG preservation keeps the paired point forecaster
        # exactly matched when probabilistic output is enabled.
        uncertainty_rng = torch.random.get_rng_state()
        uncertainty_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.probabilistic_output:
            self.register_buffer(
                "uncertainty_calibration", torch.tensor(1.0), persistent=True,
            )
            self.uncertainty_log_scale = nn.Parameter(
                torch.full((horizon,), -1.5),
            )
            if self.probabilistic_state:
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Linear(d_model, horizon),
                )
                nn.init.zeros_(self.uncertainty_head[-1].weight)
                nn.init.zeros_(self.uncertainty_head[-1].bias)
            else:
                self.uncertainty_head = None
        else:
            self.register_buffer(
                "uncertainty_calibration", torch.tensor(1.0), persistent=False,
            )
            self.register_parameter("uncertainty_log_scale", None)
            self.uncertainty_head = None
        if self.quantile_output:
            levels = torch.arange(1, self.quantile_members + 1, dtype=torch.float32)
            levels = levels / float(self.quantile_members + 1)
            self.register_buffer("quantile_levels", levels, persistent=True)
            normal = torch.distributions.Normal(0.0, 1.0).icdf(levels)
            self.quantile_base = nn.Parameter(
                0.20 * normal.view(1, -1).expand(horizon, -1).clone()
            )
            if self.quantile_state:
                self.quantile_head = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Linear(d_model, horizon * self.quantile_members),
                )
                nn.init.zeros_(self.quantile_head[-1].weight)
                nn.init.zeros_(self.quantile_head[-1].bias)
            else:
                self.quantile_head = None
            self.register_buffer(
                "quantile_interval_adjustment", torch.tensor(0.0), persistent=True,
            )
        else:
            self.register_buffer(
                "quantile_levels", torch.empty(0), persistent=False,
            )
            self.register_parameter("quantile_base", None)
            self.quantile_head = None
            self.register_buffer(
                "quantile_interval_adjustment", torch.tensor(0.0), persistent=False,
            )
        torch.random.set_rng_state(uncertainty_rng)
        if uncertainty_cuda_rng is not None:
            torch.cuda.set_rng_state_all(uncertainty_cuda_rng)
        if solar_residual_output:
            # Start at the deterministic solar trajectory, while retaining a
            # non-zero path for gradients to reach the routed history memory.
            nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.head.bias)
        self.nwp_decoder = (NWPConditionedForecastDecoder(future_channels, horizon, d_model)
                            if nwp_conditioned_decoder else None)
        if future_covariates:
            self.future_encoder = nn.Sequential(
                nn.Linear(channels, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.future_head = nn.Linear(horizon * d_model, horizon)
            self.future_gate = (nn.Linear(d_model, horizon) if self.scale_aware
                                else nn.Linear(channels, horizon))
            nn.init.zeros_(self.future_head.weight)
            nn.init.zeros_(self.future_head.bias)
            nn.init.zeros_(self.future_gate.weight)
            nn.init.constant_(self.future_gate.bias, -2.0)
        else:
            self.future_encoder = None
            self.future_head = None
            self.future_gate = None
        if output_anchor and not static_output_anchor:
            if not self.scale_aware:
                raise ValueError("physical persistence anchor requires a physical routing state")
            self.output_router = nn.Linear(d_model, horizon)
            nn.init.zeros_(self.output_router.weight)
            # A deterministic solar-trajectory anchor is substantially stronger
            # than raw persistence.  Start close to abstention and require the
            # learned history representation to earn a larger correction.
            anchor_bias = (self.output_router_bias
                           if self.output_router_bias is not None else
                           -3.0 if self.solar_trajectory_anchor else -1.4)
            nn.init.constant_(self.output_router.bias, anchor_bias)
        else:
            self.output_router = None
        self.output_anchor_logits = (nn.Parameter(torch.full((horizon,), -1.4))
                                     if output_anchor and static_output_anchor else None)
        # The physical reference is an optional residual correction to the
        # history-only candidate, never the primary forecast.  Its reliability
        # is inferred only from the encoded historical state and three
        # target-free historical diagnostics.  Zero strength therefore returns
        # the exact numerical backbone prediction.
        reference_rng = torch.random.get_rng_state()
        reference_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.reference_residual_enabled:
            self.reference_reliability_router = nn.Sequential(
                nn.LayerNorm(d_model + 3, elementwise_affine=False),
                nn.Linear(d_model + 3, d_model),
                nn.GELU(),
                nn.Linear(d_model, horizon),
            )
            nn.init.zeros_(self.reference_reliability_router[-1].weight)
            nn.init.constant_(self.reference_reliability_router[-1].bias, -2.0)
        else:
            self.reference_reliability_router = None
        torch.random.set_rng_state(reference_rng)
        if reference_cuda_rng is not None:
            torch.cuda.set_rng_state_all(reference_cuda_rng)
        self.reference_residual_active = True
        self.reference_strength = 1.0
        self.last_reference_reliability = None
        self.last_reference_history_features = None
        self.last_scale_weights = None
        self.last_tf_weights = None
        self.last_temporal_weights = None
        self.last_output_correction = None
        self.last_spectral_attention = None
        self.last_spectral_residual_gain = None
        self.last_spectral_raw_contribution_ratio = None
        self.last_spectral_contribution_ratio = None
        self.last_frequency_v2_gate = None
        self.last_frequency_v2_scale = None
        self.last_frequency_v2_contribution_ratio = None
        self.last_frequency_v2_source_energy = None
        self.last_frequency_v3_gate = None
        self.last_frequency_v3_scale = None
        self.last_frequency_v3_contribution_ratio = None
        self.last_frequency_v3_spectral_energy = None
        self.last_frequency_v4_gate = None
        self.last_frequency_v4_scale = None
        self.last_frequency_v4_contribution_ratio = None
        self.last_frequency_v4_source_energy = None
        self.last_frequency_v4_local_gate = None
        self.last_frequency_v5_null_probability = None
        self.last_frequency_v5_channel_gate = None
        self.last_frequency_v5_residual_scale = None
        self.last_frequency_v5_contribution_ratio = None
        self.last_frequency_v6_band_gate = None
        self.last_frequency_v6_band_energy_fraction = None
        self.last_frequency_v6_residual_scale = None
        self.last_frequency_v6_contribution_ratio = None
        self.last_frequency_v6_magnitude_adjustment = None
        self.last_frequency_v6_phase_adjustment = None
        self.last_frequency_v7_band_gate = None
        self.last_frequency_v7_coherence = None
        self.last_frequency_v7_stability = None
        self.last_frequency_v7_horizon_prior = None
        self.last_frequency_v7_clip_factor = None
        self.last_frequency_v7_contribution_ratio = None
        self.last_frequency_v7_magnitude_adjustment = None
        self.last_frequency_v7_phase_adjustment = None
        self.last_frequency_v8_band_channel_gate = None
        self.last_frequency_v8_source_energy_fraction = None
        self.last_frequency_v8_robust_clip_fraction = None
        self.last_frequency_v8_residual_cap = None
        self.last_frequency_v8_residual_clip_fraction = None
        self.last_frequency_v8_raw_contribution_ratio = None
        self.last_frequency_v8_contribution_ratio = None
        self.last_frequency_v8_channel_contribution_ratio = None
        self.last_frequency_v9_trend_mix = None
        self.last_frequency_v9_band_gate = None
        self.last_frequency_v9_residual_scale = None
        self.last_frequency_v9_trend_energy_fraction = None
        self.last_frequency_v9_raw_contribution_ratio = None
        self.last_frequency_v9_contribution_ratio = None
        self.last_frequency_v9_target_contribution_ratio = None
        self.last_frequency_v9_robust_clip_fraction = None
        self.last_routing_state = None
        self.last_state_logits = None
        self.last_candidate_pred = None
        self.force_tf_route = None
        # Counterfactual evaluation only: break the sample--state connection
        # after encoding while leaving every forecasting feature untouched.
        # This tests whether the claimed physical controller causally affects
        # routing, scale selection and output correction.
        self.routing_state_intervention = None
        self.tf_state_intervention = None
        self.scale_state_intervention = None
        self.output_state_intervention = None
        self.route_scale_weights = None
        self.route_tf_weights = None

    def attach_scale_prompt_adapter(self, prompt_dim: int = 1536,
                                    n_heads: int = 3,
                                    output_gate: bool = True,
                                    contrastive_scale_alignment: bool = False) -> None:
        """Attach the frozen-LLM adapter after the numerical model is trained.

        Delayed attachment is stronger than merely freezing a module during
        stage 1: the validated numerical/FSRA checkpoint is produced with the
        exact original parameter allocation and execution graph.  The new
        adapter is then the only trainable component, so strength zero is the
        literal pre-attachment forecast rather than a nominally paired rerun.
        """
        if not self.scale_aware:
            raise ValueError("scale-matched prompts require the scale-aware encoder")
        if self.scale_prompt_adapter is not None:
            return
        device = next(self.parameters()).device
        self.scale_prompt_dim = int(prompt_dim)
        self.scale_prompt_heads = int(n_heads)
        self.scale_prompt_output_gate = bool(output_gate)
        self.scale_prompt_enabled = True
        self.uses_prompt_features = True
        self.scale_prompt_adapter = ScaleMatchedPromptAdapter(
            self.scale_prompt_dim, self.prompt_d_model, len(self.patch.scales),
            n_heads=self.scale_prompt_heads,
            contrastive_scale_alignment=contrastive_scale_alignment,
        ).to(device)
        if self.scale_prompt_output_gate:
            self.scale_prompt_output_router = nn.Sequential(
                nn.LayerNorm(2 * self.prompt_d_model, elementwise_affine=False),
                nn.Linear(2 * self.prompt_d_model, self.prompt_d_model, bias=False),
                nn.GELU(),
                nn.Linear(self.prompt_d_model, self.prompt_horizon, bias=False),
            ).to(device)
            nn.init.normal_(self.scale_prompt_output_router[-1].weight, std=1e-3)
        else:
            self.scale_prompt_output_router = None

    def attach_tf_prompt_router(self, prompt_dim: int = 1792) -> None:
        """Attach an LLM controller only to the existing TF residual router."""
        if not self.time_frequency or self.tf_router is None:
            raise ValueError("prompt-conditioned TF routing requires a dynamic TF model")
        if self.semantic_tf_prompt_router is not None:
            return
        device = next(self.parameters()).device
        routes = int(self.tf_prior.numel())
        self.semantic_tf_prompt_router = PromptConditionedTFRouter(
            prompt_dim, self.prompt_d_model, routes=routes,
        ).to(device)
        self.uses_prompt_features = True

    def attach_hierarchical_prompt_router(
            self, prompt_dim: int = 1536,
            centered_route_delta: bool = False,
            content_pairing: bool = False) -> None:
        """Attach frozen-LLM prototypes only to TF and scale router logits.

        Unlike the earlier prompt adapter, this module cannot modify tokens or
        the prediction head.  It supplies a state-conditioned semantic prior to
        the two already-existing physical routing decisions.
        """
        if not self.time_frequency or self.tf_router is None:
            raise ValueError("hierarchical prompt routing requires a dynamic TF model")
        if not self.scale_aware or self.scale_router is None:
            raise ValueError("hierarchical prompt routing requires scale-aware patching")
        if self.semantic_hierarchical_router is not None:
            return
        device = next(self.parameters()).device
        self.semantic_hierarchical_router = PromptConditionedHierarchicalRouter(
            prompt_dim=prompt_dim,
            state_dim=self.prompt_d_model,
            tf_routes=int(self.tf_prior.numel()),
            scale_routes=len(self.patch.scales),
            centered_route_delta=centered_route_delta,
            content_pairing=content_pairing,
        ).to(device)
        self.uses_prompt_features = True

    @staticmethod
    def _intervene_state(state: torch.Tensor, intervention: Optional[str]) -> torch.Tensor:
        if intervention is None:
            return state
        if intervention == "zero":
            return torch.zeros_like(state)
        if intervention == "shuffle":
            return torch.roll(state, shifts=1, dims=0)
        raise ValueError(f"unknown state intervention={intervention!r}")

    def _solar_trajectory_base(self, routing_x: torch.Tensor,
                               future_x: Optional[torch.Tensor]) -> torch.Tensor:
        """Leakage-free smart persistence in the model's target scale.

        The latest valid historical clear-sky-normalized power index is held
        constant while deterministic future clear-sky irradiance follows the
        solar trajectory.  ``future_x`` contains only audited solar geometry;
        target power and observation flags are excluded by the loader.
        """
        if future_x is None:
            raise ValueError("solar trajectory anchor requires audited future solar geometry")
        cfg = self.solar_anchor_config
        clear_hist_idx = int(cfg["clear_hist_idx"])
        clear_future_pos = int(cfg["clear_future_pos"])
        target_mu, target_sd = float(cfg["target_mu"]), float(cfg["target_sd"])
        clear_mu, clear_sd = float(cfg["clear_mu"]), float(cfg["clear_sd"])
        clear_max = max(float(cfg["clear_max"]), 1e-6)
        capacity = max(float(cfg["capacity"]), 1e-6)

        power = routing_x[..., -1] * target_sd + target_mu
        clear_hist = (routing_x[..., clear_hist_idx] * clear_sd + clear_mu).clamp_min(0.0)
        clear_future = (future_x[..., clear_future_pos] * clear_sd + clear_mu).clamp_min(0.0)
        clear_hist_norm = clear_hist / clear_max
        power_norm = power / capacity
        valid = clear_hist_norm > 0.02
        # Select the latest valid daylight performance index without looking at
        # any future target.  At night this falls back to the preceding daylight
        # observation contained in the historical window.
        positions = torch.arange(routing_x.shape[1], device=routing_x.device).view(1, -1)
        latest = torch.where(valid, positions, positions.new_full(positions.shape, -1)).max(dim=1).values
        safe_latest = latest.clamp_min(0)
        index_series = power_norm / clear_hist_norm.clamp_min(0.02)
        kpv = index_series.gather(1, safe_latest[:, None]).squeeze(1)
        kpv = torch.where(latest >= 0, kpv, torch.zeros_like(kpv)).clamp(0.0, 1.5)
        base_phys = capacity * kpv[:, None] * (clear_future / clear_max)

        daylight_future_pos = cfg.get("daylight_future_pos")
        if daylight_future_pos is not None:
            mask_mu = float(cfg.get("daylight_mu", 0.0))
            mask_sd = float(cfg.get("daylight_sd", 1.0))
            daylight = future_x[..., int(daylight_future_pos)] * mask_sd + mask_mu
            base_phys = base_phys * (daylight > 0.5).to(base_phys.dtype)
        if self.reference_ensemble or self.daily_shape_anchor:
            persistence_phys = power[:, -1:].expand(-1, clear_future.shape[1])
            cycle = min(int(self.cycle_len), routing_x.shape[1])
            start = routing_x.shape[1] - cycle
            idx = start + torch.arange(clear_future.shape[1], device=routing_x.device) % cycle
            daily_phys = power.index_select(1, idx)
        if self.reference_ensemble:
            weights = torch.softmax(self.reference_logits, dim=0)
            self.last_reference_weights = weights.detach()
            base_phys = (weights[0] * base_phys + weights[1] * persistence_phys
                         + weights[2] * daily_phys)
        elif self.daily_shape_anchor:
            base_phys = daily_phys
        return (base_phys - target_mu) / target_sd

    def _historical_reference_features(self,
                                       routing_x: torch.Tensor) -> torch.Tensor:
        """History-only diagnostics of physical-reference trustworthiness.

        The three dimensionless statistics are (1) mean absolute PV ramp,
        (2) sign-reversal rate of consecutive ramps and (3) daylight
        clear-sky performance-index dispersion.  They use no future target and
        make the gate answer a narrow question: how much should the physical
        reference correct this history-only forecast?
        """
        cfg = self.solar_anchor_config
        target_mu, target_sd = float(cfg["target_mu"]), float(cfg["target_sd"])
        clear_mu, clear_sd = float(cfg["clear_mu"]), float(cfg["clear_sd"])
        capacity = max(float(cfg["capacity"]), 1e-6)
        clear_max = max(float(cfg["clear_max"]), 1e-6)
        clear_hist_idx = int(cfg["clear_hist_idx"])

        power = routing_x[..., -1] * target_sd + target_mu
        power_norm = (power / capacity).clamp(-0.25, 2.0)
        delta = power_norm[:, 1:] - power_norm[:, :-1]
        ramp = delta.abs().mean(dim=1)
        if delta.shape[1] > 1:
            reversal = (
                (delta[:, 1:] * delta[:, :-1] < 0.0).to(delta.dtype).mean(dim=1)
            )
        else:
            reversal = torch.zeros_like(ramp)

        clear_hist = (
            routing_x[..., clear_hist_idx] * clear_sd + clear_mu
        ).clamp_min(0.0)
        clear_norm = clear_hist / clear_max
        daylight = (clear_norm > 0.02).to(power_norm.dtype)
        performance = power_norm / clear_norm.clamp_min(0.02)
        count = daylight.sum(dim=1).clamp_min(1.0)
        mean_performance = (performance * daylight).sum(dim=1) / count
        dispersion = (
            (performance - mean_performance[:, None]).abs() * daylight
        ).sum(dim=1) / count
        features = torch.stack([ramp, reversal, dispersion], dim=-1)
        return features.clamp(0.0, 2.0)

    def _frequency_source(self, x: torch.Tensor) -> torch.Tensor:
        if self.spectral_disturbance:
            # First differences attenuate the smooth diurnal envelope.  The
            # temporal expert keeps direction and persistence, whereas the
            # spectrum sees oscillatory disturbance energy by physical band.
            return torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        if not self.residual_frequency:
            return x
        # About one hour at either 5-min (cycle=288) or 15-min (cycle=96)
        # resolution.  This removes the smooth solar envelope before FFT and
        # leaves cloud/ramp fluctuations to the spectral branch.
        kernel = max(3, self.cycle_len // 24)
        if kernel % 2 == 0:
            kernel += 1
        xt = x.transpose(1, 2)
        smooth = F.avg_pool1d(F.pad(xt, (kernel // 2, kernel // 2), mode="replicate"), kernel_size=kernel, stride=1)
        return x - smooth.transpose(1, 2)

    def _cycle_transition_temporal_delta(self, x: torch.Tensor,
                                         cycle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified cycle--transition temporal encoding used by PSRC-v2.

        GTR-style periodic retrieval and L-Drive-style latent transition
        tracking are internal mechanisms of one temporal encoder.  Exposing a
        single residual and a single diagnostic gate prevents the paper-facing
        architecture from treating them as independent prediction branches.
        """
        periodic_delta = transition_delta = None
        if not self.disable_gtr:
            positions = cycle[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]
            q = self.q_memory[positions % self.cycle_len].permute(0, 2, 1)
            periodic_delta = self.gtr(x.permute(0, 2, 1), q).permute(0, 2, 1)
        if not self.disable_ldrive:
            transition_delta = self.lcontext(x) - x

        if periodic_delta is not None and transition_delta is not None:
            weights = torch.softmax(
                self.temporal_gate(torch.cat([periodic_delta, transition_delta], dim=-1).mean(dim=1)),
                dim=-1,
            )
            delta = (weights[:, 0, None, None] * self.branch_norm[0](periodic_delta)
                     + weights[:, 1, None, None] * transition_delta)
        elif periodic_delta is not None:
            weights = x.new_tensor([1.0, 0.0]).view(1, 2).expand(x.shape[0], -1)
            delta = self.branch_norm[0](periodic_delta)
        elif transition_delta is not None:
            weights = x.new_tensor([0.0, 1.0]).view(1, 2).expand(x.shape[0], -1)
            delta = transition_delta
        else:
            raise ValueError("periodic retrieval and latent transition tracking cannot both be disabled")
        return delta, weights

    def _fsra_relation_bias(self, dtype: torch.dtype,
                            device: torch.device) -> Optional[torch.Tensor]:
        """Return a centered channel-relevance prior with a learned scale.

        Centering only off-diagonal similarities removes the common language
        direction shared by fixed-schema descriptions.  The diagonal is kept
        at zero because the numerical encoder already has an identity path.
        A zero embedding bank therefore produces an exact zero bias.
        """
        if (self.fsra_channel_embeddings is None or not self.fsra_enabled
                or not self.fsra_has_content
                or not self.fsra_relation_enabled):
            return None
        embeddings = self.fsra_channel_embeddings.to(device=device, dtype=dtype)
        norms = embeddings.norm(dim=-1, keepdim=True)
        normalized = embeddings / norms.clamp_min(1e-6)
        similarity = normalized @ normalized.transpose(0, 1)
        c = similarity.shape[0]
        off_diag = ~torch.eye(c, dtype=torch.bool, device=device)
        valid_channels = norms.squeeze(-1) > 1e-6
        valid_pairs = off_diag & valid_channels[:, None] & valid_channels[None, :]
        if valid_pairs.any():
            values = similarity[valid_pairs]
            mean = values.mean()
            std = values.std(unbiased=False).clamp_min(1e-4)
            centered = torch.where(valid_pairs, (similarity - mean) / std, torch.zeros_like(similarity))
            centered = centered.clamp(-2.0, 2.0)
        else:
            centered = torch.zeros_like(similarity)
        scale = F.softplus(self.fsra_relation_logit)
        self.last_fsra_relation_scale = scale.detach()
        return self.fsra_strength * scale * centered

    def forward(self, x: torch.Tensor, cycle: torch.Tensor,
                future_x: Optional[torch.Tensor] = None,
                prompt_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.last_scale_weights = None
        self.last_tf_weights = None
        self.last_temporal_weights = None
        self.last_output_correction = None
        self.last_spectral_attention = None
        self.last_frequency_v2_gate = None
        self.last_frequency_v2_scale = None
        self.last_frequency_v2_contribution_ratio = None
        self.last_frequency_v2_source_energy = None
        self.last_frequency_v3_gate = None
        self.last_frequency_v3_scale = None
        self.last_frequency_v3_contribution_ratio = None
        self.last_frequency_v3_spectral_energy = None
        self.last_frequency_v4_gate = None
        self.last_frequency_v4_scale = None
        self.last_frequency_v4_contribution_ratio = None
        self.last_frequency_v4_source_energy = None
        self.last_frequency_v4_local_gate = None
        self.last_frequency_v5_null_probability = None
        self.last_frequency_v5_channel_gate = None
        self.last_frequency_v5_residual_scale = None
        self.last_frequency_v5_contribution_ratio = None
        self.last_frequency_v6_band_gate = None
        self.last_frequency_v6_band_energy_fraction = None
        self.last_frequency_v6_residual_scale = None
        self.last_frequency_v6_contribution_ratio = None
        self.last_frequency_v6_magnitude_adjustment = None
        self.last_frequency_v6_phase_adjustment = None
        self.last_frequency_v7_band_gate = None
        self.last_frequency_v7_coherence = None
        self.last_frequency_v7_stability = None
        self.last_frequency_v7_horizon_prior = None
        self.last_frequency_v7_clip_factor = None
        self.last_frequency_v7_contribution_ratio = None
        self.last_frequency_v7_magnitude_adjustment = None
        self.last_frequency_v7_phase_adjustment = None
        self.last_frequency_v8_band_channel_gate = None
        self.last_frequency_v8_source_energy_fraction = None
        self.last_frequency_v8_robust_clip_fraction = None
        self.last_frequency_v8_residual_cap = None
        self.last_frequency_v8_residual_clip_fraction = None
        self.last_frequency_v8_raw_contribution_ratio = None
        self.last_frequency_v8_contribution_ratio = None
        self.last_frequency_v8_channel_contribution_ratio = None
        self.last_frequency_v9_trend_mix = None
        self.last_frequency_v9_band_gate = None
        self.last_frequency_v9_residual_scale = None
        self.last_frequency_v9_trend_energy_fraction = None
        self.last_frequency_v9_raw_contribution_ratio = None
        self.last_frequency_v9_contribution_ratio = None
        self.last_frequency_v9_target_contribution_ratio = None
        self.last_frequency_v9_robust_clip_fraction = None
        self.last_routing_state = None
        self.last_state_logits = None
        self.last_candidate_pred = None
        self.last_semantic_gate = None
        self.last_semantic_channel_mix = None
        self.last_semantic_output_gate = None
        self.semantic_alignment_loss = None
        self.last_scale_prompt_gate = None
        self.last_scale_prompt_head_weights = None
        self.last_scale_prompt_state = None
        self.last_semantic_tf_route_delta = None
        self.last_semantic_scale_route_delta = None
        self.last_semantic_prototype_weights = None
        self.last_reference_reliability = None
        self.last_reference_history_features = None
        self.last_fsra_gate = None
        self.last_fsra_relation_scale = None
        self.last_fsra_head_weights = None
        self.last_predictive_scale = None
        self.last_predictive_members = None
        self.fsra_head_specialization_loss = None
        self.fsra_alignment_loss = None
        self.route_scale_weights = None
        self.route_tf_weights = None
        # Preserve absolute solar/quality state for routing. RevIN below is
        # beneficial for forecasting, but would erase the window's absolute
        # zenith/daylight level and weaken the physical meaning of the router.
        routing_x = x
        if self.quality_indices.numel() > 0:
            # Quality flags are intentionally kept in their original [0, 1]
            # scale by ``load_windows``.  Their window mean is the fraction of
            # genuinely observed samples, rather than interpolated values.
            spectral_reliability = routing_x.index_select(-1, self.quality_indices).mean(dim=(1, 2)).clamp(0.0, 1.0)
        else:
            spectral_reliability = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - means) / stdev
        else:
            means = stdev = None
            x = self.norm(x)
        routing_state = None
        if self.scale_aware:
            # Absolute solar position/quality and relative within-window shape
            # are complementary. The router starts mostly relative and learns
            # how much absolute solar state is transferable across seasons.
            abs_weight = torch.sigmoid(self.abs_route_logit)
            route_rel = x[:, -min(96, x.shape[1]):] if self.recent_only_physics else x
            route_abs = routing_x[:, -min(96, routing_x.shape[1]):] if self.recent_only_physics else routing_x
            routing_state = ((1.0 - abs_weight) * self.phys_encoder(route_rel)
                             + abs_weight * self.phys_encoder(route_abs))
            routing_state = self._intervene_state(
                routing_state, self.routing_state_intervention
            )
            self.last_routing_state = routing_state
            if self.state_classifier is not None:
                self.last_state_logits = self.state_classifier(routing_state)
        semantic_tf_delta = None
        semantic_scale_delta = None
        if (self.semantic_hierarchical_router is not None
                and self.semantic_enabled):
            if prompt_x is None:
                raise ValueError(
                    "hierarchical frozen-LLM routing requires prototype prompts"
                )
            semantic_tf_delta, semantic_scale_delta = (
                self.semantic_hierarchical_router(prompt_x, routing_state)
            )
            self.last_semantic_tf_route_delta = semantic_tf_delta.detach()
            self.last_semantic_scale_route_delta = semantic_scale_delta.detach()
            self.last_semantic_prototype_weights = (
                self.semantic_hierarchical_router.last_prototype_weights
            )
            self.semantic_alignment_loss = (
                self.semantic_hierarchical_router.pairing_loss
            )
        if self.mode == "patch_only":
            fused = x
        elif self.mode == "mkan_only":
            fused = self.mkan(x)
        elif self.mode == "sequential":
            m = self.mkan(x)
            q = self.q_memory[(cycle[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]) % self.cycle_len].permute(0, 2, 1)
            g = self.gtr(m.permute(0, 2, 1), q).permute(0, 2, 1)
            fused = x + g
        elif self.time_frequency:
            temporal_delta, temporal_weights = self._cycle_transition_temporal_delta(x, cycle)
            # Only genuinely dynamic variables enter the spectral response
            # path.  Deterministic clocks, solar masks and quality flags would
            # otherwise create artificial spectral peaks.
            frequency_input = x.index_select(-1, self.frequency_indices)
            state = self._intervene_state(routing_state, self.tf_state_intervention)
            if self.mode.endswith("_static"):
                tf_logits = self.tf_logits.unsqueeze(0).expand(x.shape[0], -1)
            else:
                tf_logits = self.tf_router(state) + self.tf_prior
                if (self.semantic_tf_prompt_router is not None
                        and self.semantic_enabled):
                    if prompt_x is None:
                        raise ValueError("prompt-conditioned TF router requires frozen-LLM prompts")
                    tf_delta = self.semantic_tf_prompt_router(prompt_x, state)
                    tf_logits = tf_logits + float(self.semantic_strength) * tf_delta
                    self.last_semantic_tf_route_delta = tf_delta.detach()
                if semantic_tf_delta is not None:
                    tf_logits = (
                        tf_logits
                        + float(self.semantic_strength) * semantic_tf_delta
                    )
            temporal_residual = self.temporal_drop_path(temporal_delta)
            self.last_temporal_weights = temporal_weights.detach()
            if self.frequency_v2 is not None or self.frequency_v3_mode is not None:
                # The temporal/identity decision is normalized independently
                # of the spectrum.  Disabling frequency therefore leaves the
                # exact same temporal baseline instead of wasting softmax mass
                # on an all-zero spectral route.
                if tf_logits.shape[-1] == 3:
                    temporal_identity = torch.softmax(
                        tf_logits[:, [0, 2]], dim=-1,
                    )
                    temporal_weight = temporal_identity[:, 0]
                else:
                    temporal_identity = x.new_ones((x.shape[0], 1))
                    temporal_weight = temporal_identity[:, 0]
                if self.frequency_v2 is not None:
                    selected_delta = self.frequency_v2(
                        frequency_input, routing_state,
                    )
                    frequency_delta = torch.zeros_like(x)
                    frequency_delta.index_copy_(
                        -1, self.frequency_indices, selected_delta,
                    )
                    frequency_delta = (
                        frequency_delta
                        * spectral_reliability[:, None, None]
                    )
                    frequency_residual = self.spectral_drop_path(frequency_delta)
                else:
                    # Frequency-v3 is deliberately moved behind CorPatch.  At
                    # the input, every v3 control therefore shares this exact
                    # temporal/identity baseline with no hidden spectral path.
                    frequency_residual = torch.zeros_like(x)
                fused = (
                    x
                    + self.tf_branch_scale[0]
                    * temporal_weight[:, None, None]
                    * temporal_residual
                    + frequency_residual
                )
                if self.frequency_v2 is not None:
                    self.last_frequency_v2_gate = (
                        self.frequency_v2.last_branch_gate
                    )
                    self.last_frequency_v2_scale = (
                        self.frequency_v2.last_branch_scale
                    )
                    self.last_frequency_v2_contribution_ratio = (
                        self.frequency_v2.last_contribution_ratio
                    )
                    self.last_frequency_v2_source_energy = (
                        self.frequency_v2.last_source_energy
                    )
                # Dedicated v2 diagnostics are used because these two values
                # mean temporal-vs-identity, not temporal-vs-frequency.
                self.last_tf_weights = None
                self.route_tf_weights = None
            else:
                frequency_source = self._frequency_source(frequency_input)
                if self.frequency_v4 is not None:
                    selected_delta = self.frequency_v4(
                        frequency_input, routing_state,
                    )
                    self.last_spectral_attention = None
                    self.last_frequency_v4_gate = (
                        self.frequency_v4.last_branch_gate
                    )
                    self.last_frequency_v4_scale = (
                        self.frequency_v4.last_branch_scale
                    )
                    self.last_frequency_v4_contribution_ratio = (
                        self.frequency_v4.last_contribution_ratio
                    )
                    self.last_frequency_v4_source_energy = (
                        self.frequency_v4.last_source_energy
                    )
                    self.last_frequency_v4_local_gate = (
                        self.frequency_v4.last_local_spectral_gate
                    )
                elif self.frequency_v5 is not None:
                    selected_delta = (
                        self.frequency_v5(frequency_source, routing_state)
                        - frequency_source
                    )
                    self.last_spectral_attention = (
                        self.frequency_v5.last_band_attention[:, :3]
                    )
                    self.last_frequency_v5_null_probability = (
                        self.frequency_v5.last_null_probability
                    )
                    self.last_frequency_v5_channel_gate = (
                        self.frequency_v5.last_channel_gate
                    )
                    self.last_frequency_v5_residual_scale = (
                        self.frequency_v5.last_residual_scale
                    )
                    self.last_frequency_v5_contribution_ratio = (
                        self.frequency_v5.last_contribution_ratio
                    )
                elif self.frequency_v6 is not None:
                    selected_delta = self.frequency_v6(
                        frequency_input, routing_state,
                    )
                    self.last_spectral_attention = (
                        self.frequency_v6.last_band_gate
                    )
                    self.last_frequency_v6_band_gate = (
                        self.frequency_v6.last_band_gate
                    )
                    self.last_frequency_v6_band_energy_fraction = (
                        self.frequency_v6.last_band_energy_fraction
                    )
                    self.last_frequency_v6_residual_scale = (
                        self.frequency_v6.last_residual_scale
                    )
                    self.last_frequency_v6_contribution_ratio = (
                        self.frequency_v6.last_contribution_ratio
                    )
                    self.last_frequency_v6_magnitude_adjustment = (
                        self.frequency_v6.last_magnitude_adjustment
                    )
                    self.last_frequency_v6_phase_adjustment = (
                        self.frequency_v6.last_phase_adjustment
                    )
                elif self.frequency_v7 is not None:
                    selected_delta = self.frequency_v7(
                        frequency_input, routing_state,
                    )
                    self.last_spectral_attention = (
                        self.frequency_v7.last_band_gate
                    )
                    self.last_frequency_v7_band_gate = (
                        self.frequency_v7.last_band_gate
                    )
                    self.last_frequency_v7_coherence = (
                        self.frequency_v7.last_coherence
                    )
                    self.last_frequency_v7_stability = (
                        self.frequency_v7.last_stability
                    )
                    self.last_frequency_v7_horizon_prior = (
                        self.frequency_v7.last_horizon_prior
                    )
                    self.last_frequency_v7_clip_factor = (
                        self.frequency_v7.last_clip_factor
                    )
                    self.last_frequency_v7_contribution_ratio = (
                        self.frequency_v7.last_contribution_ratio
                    )
                    self.last_frequency_v7_magnitude_adjustment = (
                        self.frequency_v7.last_magnitude_adjustment
                    )
                    self.last_frequency_v7_phase_adjustment = (
                        self.frequency_v7.last_phase_adjustment
                    )
                elif self.frequency_v8 is not None:
                    selected_delta = self.frequency_v8(
                        frequency_input, routing_state,
                    )
                    self.last_spectral_attention = None
                    self.last_frequency_v8_band_channel_gate = (
                        self.frequency_v8.last_band_channel_gate
                    )
                    self.last_frequency_v8_source_energy_fraction = (
                        self.frequency_v8.last_source_energy_fraction
                    )
                    self.last_frequency_v8_robust_clip_fraction = (
                        self.frequency_v8.last_robust_clip_fraction
                    )
                    self.last_frequency_v8_residual_cap = (
                        self.frequency_v8.last_residual_cap
                    )
                    self.last_frequency_v8_residual_clip_fraction = (
                        self.frequency_v8.last_residual_clip_fraction
                    )
                    self.last_frequency_v8_raw_contribution_ratio = (
                        self.frequency_v8.last_raw_contribution_ratio
                    )
                    self.last_frequency_v8_contribution_ratio = (
                        self.frequency_v8.last_contribution_ratio
                    )
                    self.last_frequency_v8_channel_contribution_ratio = (
                        self.frequency_v8.last_channel_contribution_ratio
                    )
                elif self.frequency_v9 is not None:
                    selected_delta = self.frequency_v9(
                        frequency_input, routing_state,
                    )
                    self.last_spectral_attention = (
                        self.frequency_v9.last_band_gate
                    )
                    self.last_frequency_v9_trend_mix = (
                        self.frequency_v9.last_trend_mix
                    )
                    self.last_frequency_v9_band_gate = (
                        self.frequency_v9.last_band_gate
                    )
                    self.last_frequency_v9_residual_scale = (
                        self.frequency_v9.last_residual_scale
                    )
                    self.last_frequency_v9_trend_energy_fraction = (
                        self.frequency_v9.last_trend_energy_fraction
                    )
                    self.last_frequency_v9_raw_contribution_ratio = (
                        self.frequency_v9.last_raw_contribution_ratio
                    )
                    self.last_frequency_v9_contribution_ratio = (
                        self.frequency_v9.last_contribution_ratio
                    )
                    self.last_frequency_v9_target_contribution_ratio = (
                        self.frequency_v9.last_target_contribution_ratio
                    )
                    self.last_frequency_v9_robust_clip_fraction = (
                        self.frequency_v9.last_robust_clip_fraction
                    )
                elif self.disable_frequency or self.clean_frequency_ablation:
                    # Strict spectral-expert ablation: retain the current temporal
                    # path, physical router, identity route, patches and decoder;
                    # remove only the frequency-domain residual contribution.
                    selected_delta = torch.zeros_like(frequency_source)
                    self.last_spectral_attention = None
                else:
                    selected_delta = (
                        self.spectral_mkan(frequency_source, routing_state)
                        - frequency_source
                    )
                    self.last_spectral_attention = (
                        self.spectral_mkan.last_band_attention
                    )
                    self.last_spectral_residual_gain = (
                        self.spectral_mkan.last_residual_gain
                    )
                    self.last_spectral_raw_contribution_ratio = (
                        self.spectral_mkan.last_raw_contribution_ratio
                    )
                    self.last_spectral_contribution_ratio = (
                        self.spectral_mkan.last_contribution_ratio
                    )
                frequency_delta = torch.zeros_like(x)
                frequency_delta.index_copy_(-1, self.frequency_indices, selected_delta)
                # Interpolation boundaries create artificial high-frequency
                # energy.  Reliability gating keeps the spectral branch active on
                # clean windows and continuously suppresses it on low-quality ones.
                frequency_delta = frequency_delta * spectral_reliability[:, None, None]
                tf_weights = torch.softmax(tf_logits, dim=-1)
                if (self.clean_frequency_ablation
                        or self.frequency_v5_mode == "off"
                        or self.frequency_v6_mode == "off"
                        or self.spectral_mkan_mode == "residual_off"):
                    # Strict original-frequency ablation.  The original
                    # SpectralMKAN, the three-logit router and every parameter
                    # remain instantiated, but DeltaH_freq is excluded from
                    # fusion.  Re-normalize the remaining temporal/identity
                    # decision so the removed branch cannot consume softmax
                    # probability and indirectly suppress DeltaH_time.
                    if tf_logits.shape[-1] == 3:
                        temporal_identity = torch.softmax(
                            tf_logits[:, [0, 2]], dim=-1,
                        )
                        tf_weights = torch.stack([
                            temporal_identity[:, 0],
                            torch.zeros_like(temporal_identity[:, 0]),
                            temporal_identity[:, 1],
                        ], dim=-1)
                    else:
                        tf_weights = torch.stack([
                            torch.ones_like(tf_logits[:, 0]),
                            torch.zeros_like(tf_logits[:, 0]),
                        ], dim=-1)
                elif self.force_tf_route is not None:
                    tf_weights = F.one_hot(
                        torch.full((x.shape[0],), int(self.force_tf_route), device=x.device),
                        num_classes=tf_weights.shape[-1],
                    ).to(x.dtype)
                # If a third route is present its probability is deliberately not
                # added: that mass means "use the identity path only".
                frequency_residual = self.spectral_drop_path(frequency_delta)
                fused = (x
                         + self.tf_branch_scale[0] * tf_weights[:, 0, None, None] * temporal_residual
                         + self.tf_branch_scale[1] * tf_weights[:, 1, None, None] * frequency_residual)
                self.last_tf_weights = tf_weights.detach()
                self.route_tf_weights = tf_weights
        else:
            q = self.q_memory[(cycle[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]) % self.cycle_len].permute(0, 2, 1)
            g = self.gtr(x.permute(0, 2, 1), q).permute(0, 2, 1)
            m = self.mkan(x)
            # L-Drive is a residual block; only its latent increment should be
            # fused, otherwise the raw x path is counted twice.
            l_delta = self.lcontext(x) - x
            # Do not normalize the L-Drive increment: its learnable 0.1
            # enhancement weight is the stability prior of the original block;
            # LayerNorm would amplify that small physical change to unit scale.
            branches = [self.branch_norm[0](g), self.branch_norm[1](m), l_delta]
            if self.mode.startswith("parallel_no_l"):
                active = branches[:2]
                w = torch.softmax(self.gate2(torch.cat(active, dim=-1).mean(1)), dim=-1)
            else:
                active = branches
                w = torch.softmax(self.gate(torch.cat(active, dim=-1).mean(1)), dim=-1)
            fused = x
            for i, branch in enumerate(active):
                fused = fused + self.branch_scale[i] * w[:, i, None, None] * branch
        prompt_state = None
        if self.semantic_projection is not None and self.semantic_enabled:
            if prompt_x is None:
                raise ValueError("semantic prompt model requires precomputed prompt tokens")
            prompt_tokens = self.semantic_projection(prompt_x.to(fused.dtype))
            if self.semantic_prompt_encoder is not None:
                prompt_tokens = self.semantic_prompt_encoder(prompt_tokens)
            if self.adaptive_semantic_alignment:
                query_tokens = self.semantic_query(fused)
                prompt_state = prompt_tokens.mean(dim=1)
                if self.semantic_cma is not None:
                    head_hidden = []
                    for attention in self.semantic_cma:
                        hidden_i, _ = attention(
                            query_tokens, prompt_tokens, prompt_tokens,
                            need_weights=False,
                        )
                        head_hidden.append(hidden_i)
                    head_weights = torch.softmax(
                        self.semantic_head_router(torch.cat([routing_state, prompt_state], dim=-1)),
                        dim=-1,
                    )
                    stacked = torch.stack(head_hidden, dim=1)
                    semantic_hidden = (stacked * head_weights[:, :, None, None]).sum(dim=1)
                    self.last_semantic_head_weights = head_weights.detach()
                else:
                    semantic_hidden, _ = self.semantic_attention(
                        query_tokens, prompt_tokens, prompt_tokens, need_weights=False,
                    )
                semantic_hidden = semantic_hidden + self.semantic_ffn(semantic_hidden)
                semantic_delta = self.semantic_back(semantic_hidden)
                if self.semantic_channel_logit is not None:
                    # H = H_num + gamma_c * DeltaH_sem.  This is the
                    # channel-wise residual fusion used by the T3-style path;
                    # gamma is shared across samples and cannot itself encode
                    # the target.  Sample dependence comes only from Q/K/V.
                    semantic_gate = torch.sigmoid(
                        self.semantic_channel_logit
                    ).unsqueeze(0).expand(fused.shape[0], -1)
                    self.last_semantic_channel_mix = semantic_gate[0].detach()
                else:
                    semantic_gate = torch.sigmoid(
                        self.semantic_gate(
                            torch.cat([routing_state, prompt_state], dim=-1)
                        )
                    )
                fused = fused + self.semantic_strength * self.semantic_dropout(
                    semantic_gate[:, None, :] * semantic_delta
                )
                if self.semantic_contrastive_alignment:
                    q_pool = F.normalize(query_tokens.mean(dim=1), dim=-1)
                    p_pool = F.normalize(prompt_state, dim=-1)
                    # Sample-level InfoNCE is retained for the earlier v11/v12
                    # experiments. Prototype prompts intentionally disable it:
                    # multiple windows may share the same valid regime, so
                    # treating them as negatives would create false negatives.
                    logits = q_pool @ p_pool.transpose(0, 1) / 0.10
                    labels = torch.arange(logits.shape[0], device=logits.device)
                    self.semantic_alignment_loss = 0.5 * (
                        F.cross_entropy(logits, labels)
                        + F.cross_entropy(logits.transpose(0, 1), labels)
                    )
            else:
                semantic_delta, _ = self.semantic_attention(
                    fused, prompt_tokens, prompt_tokens, need_weights=False,
                )
                semantic_gate = torch.sigmoid(self.semantic_gate(routing_state))
                fused = fused + self.semantic_strength * semantic_gate[:, None, :] * semantic_delta
            self.last_semantic_gate = semantic_gate.detach()
        if (self.fsra_channel_embeddings is not None and self.fsra_enabled
                and self.fsra_has_content
                and self.fsra_residual_enabled):
            # [B,L,C] -> one numeric query per channel.  Frozen description
            # tokens are K,V.  Bias-free projections guarantee that an all-zero
            # semantic bank gives exactly the original time-frequency feature.
            channel_series = fused.transpose(1, 2)
            if self.fsra_transfer_bins:
                phase_grid = F.adaptive_avg_pool1d(
                    channel_series, self.fsra_transfer_bins,
                )
                numeric_tokens = self.fsra_numeric_projection(phase_grid)
            elif self.fsra_transferable:
                recent_len = max(1, fused.shape[1] // 4)
                channel_stats = torch.stack([
                    channel_series.mean(dim=-1),
                    channel_series.std(dim=-1, unbiased=False),
                    channel_series[:, :, -1],
                    channel_series[:, :, -1] - channel_series[:, :, 0],
                    channel_series.diff(dim=-1).abs().mean(dim=-1),
                    channel_series[:, :, -recent_len:].mean(dim=-1)
                    - channel_series.mean(dim=-1),
                ], dim=-1)
                numeric_tokens = self.fsra_numeric_projection(channel_stats)
            else:
                numeric_tokens = self.fsra_numeric_projection(channel_series)
            semantic_tokens = self.fsra_semantic_projection(
                self.fsra_channel_embeddings.to(fused.dtype)
            )
            semantic_batch = semantic_tokens.unsqueeze(0).expand(fused.shape[0], -1, -1)
            if self.fsra_cma is not None:
                head_outputs = []
                if self.fsra_paired_alignment:
                    paired_input = torch.cat([
                        semantic_batch, numeric_tokens * semantic_batch,
                    ], dim=-1)
                    for aligner in self.fsra_cma:
                        head_outputs.append(aligner(paired_input))
                else:
                    for aligner in self.fsra_cma:
                        head_hidden, _ = aligner(
                            numeric_tokens, semantic_batch, semantic_batch,
                            need_weights=False,
                        )
                        head_outputs.append(head_hidden)
                # [B,C,H,D].  The gate is channel- and sample-dependent because
                # it sees both current numerical dynamics and frozen PV roles.
                stacked_heads = torch.stack(head_outputs, dim=2)
                head_logits = self.fsra_head_router(
                    torch.cat([numeric_tokens, semantic_batch], dim=-1)
                )
                head_weights = torch.softmax(head_logits, dim=-1)
                semantic_hidden = (
                    head_weights.unsqueeze(-1) * stacked_heads
                ).sum(dim=2)
                self.last_fsra_head_weights = head_weights.detach()
                # Avoid the common failure where all independently learned
                # aligners collapse to one head.  Balance is computed over the
                # training batch and channels; no test label or physical rule
                # assigns a particular role to a particular head.
                mean_head_use = head_weights.mean(dim=(0, 1))
                uniform = torch.full_like(mean_head_use, 1.0 / self.fsra_cma_heads)
                balance = (mean_head_use - uniform).square().mean()
                pairwise = []
                normalized_heads = F.normalize(stacked_heads, dim=-1)
                for i in range(self.fsra_cma_heads):
                    for j in range(i + 1, self.fsra_cma_heads):
                        pairwise.append(
                            (normalized_heads[:, :, i] * normalized_heads[:, :, j])
                            .sum(dim=-1).square().mean()
                        )
                diversity = (torch.stack(pairwise).mean() if pairwise
                             else torch.zeros((), device=fused.device))
                self.fsra_head_specialization_loss = balance + 0.1 * diversity
            else:
                semantic_hidden, _ = self.fsra_attention(
                    numeric_tokens, semantic_batch, semantic_batch, need_weights=False,
                )
            fsra_gate = torch.sigmoid(
                self.fsra_gate(torch.cat([numeric_tokens, semantic_batch], dim=-1))
            )
            if self.fsra_transfer_bins:
                grid_delta = self.fsra_back(semantic_hidden)
                semantic_delta = F.interpolate(
                    grid_delta, size=fused.shape[1], mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            elif self.fsra_transferable:
                coefficients = self.fsra_back(semantic_hidden)
                time_axis = torch.linspace(
                    -1.0, 1.0, fused.shape[1], device=fused.device,
                    dtype=fused.dtype,
                )
                quadratic = time_axis.square() - time_axis.square().mean()
                recent = torch.exp(4.0 * (time_axis - 1.0))
                temporal_basis = torch.stack([
                    torch.ones_like(time_axis), time_axis, quadratic, recent,
                ], dim=-1)
                semantic_delta = torch.einsum(
                    "bck,lk->blc", coefficients, temporal_basis,
                )
            else:
                semantic_delta = self.fsra_back(semantic_hidden).transpose(1, 2)
            fused = fused + self.fsra_strength * (
                fsra_gate.transpose(1, 2) * semantic_delta
            )
            self.last_fsra_gate = fsra_gate.detach()
            if self.fsra_channel_alignment:
                numeric_norm = F.normalize(numeric_tokens, dim=-1)
                semantic_norm = F.normalize(semantic_batch, dim=-1)
                logits = numeric_norm @ semantic_norm.transpose(1, 2) / 0.10
                labels = torch.arange(self.channels, device=fused.device)
                labels = labels.unsqueeze(0).expand(fused.shape[0], -1).reshape(-1)
                self.fsra_alignment_loss = F.cross_entropy(
                    logits.reshape(-1, self.channels), labels,
                )
                if self.fsra_head_specialization_loss is not None:
                    self.fsra_alignment_loss = (
                        self.fsra_alignment_loss
                        + 0.1 * self.fsra_head_specialization_loss
                    )
        if self.scale_aware:
            state = self._intervene_state(routing_state, self.scale_state_intervention)
            scale_tokens = self.patch(fused.permute(0, 2, 1))
            if (self.scale_prompt_adapter is not None and self.scale_prompt_enabled
                    and self.semantic_enabled):
                if prompt_x is None:
                    raise ValueError(
                        "scale-matched GPT-2 adapter requires one prompt token per patch scale"
                    )
                scale_tokens = self.scale_prompt_adapter(
                    scale_tokens, prompt_x, strength=self.semantic_strength,
                )
                self.last_scale_prompt_gate = self.scale_prompt_adapter.last_gate
                self.last_scale_prompt_head_weights = (
                    self.scale_prompt_adapter.last_head_weights
                )
                self.last_scale_prompt_state = (
                    self.scale_prompt_adapter.last_prompt_state
                )
                self.last_scale_prompt_match_accuracy = (
                    self.scale_prompt_adapter.last_scale_match_accuracy
                )
                # Reuse the two-stage semantic auxiliary hook.  It regularizes
                # head utilization only; no target or hand-assigned head role
                # enters the alignment loss.
                self.semantic_alignment_loss = (
                    self.scale_prompt_adapter.auxiliary_loss
                )
            if self.scale_token_refiner is not None:
                scale_tokens = self.scale_token_refiner(scale_tokens)
            if self.mode.endswith("_static"):
                scale_weights = torch.softmax(self.scale_logits, dim=0).unsqueeze(0).expand(x.shape[0], -1)
            else:
                scale_logits = self.scale_router(state)
                if semantic_scale_delta is not None:
                    scale_logits = (
                        scale_logits
                        + float(self.semantic_strength) * semantic_scale_delta
                    )
                scale_weights = torch.softmax(scale_logits, dim=-1)
            p = self.encoder(
                scale_tokens, scale_weights,
                semantic_bias=self._fsra_relation_bias(fused.dtype, fused.device),
            )
            self.last_scale_weights = scale_weights.detach()
            self.route_scale_weights = scale_weights
        else:
            p = self.patch(fused.permute(0, 2, 1))
            p = self.encoder(p)
        b = x.shape[0]
        n = p.shape[1]
        d = p.shape[2]
        memory_by_channel = p.reshape(b, self.channels, n, d)
        if self.local_refiner is not None:
            memory_by_channel = self.local_refiner(memory_by_channel)
        if self.frequency_v3 is not None:
            target_memory = self.frequency_v3(
                x[:, :, -1], memory_by_channel[:, -1],
            )
            memory_by_channel = torch.cat([
                memory_by_channel[:, :-1], target_memory.unsqueeze(1),
            ], dim=1)
            self.last_frequency_v3_gate = self.frequency_v3.last_gate
            self.last_frequency_v3_scale = self.frequency_v3.last_scale
            self.last_frequency_v3_contribution_ratio = (
                self.frequency_v3.last_contribution_ratio
            )
            self.last_frequency_v3_spectral_energy = (
                self.frequency_v3.last_spectral_energy
            )
        h = memory_by_channel.permute(0, 1, 3, 2).reshape(b, self.channels, d * n)
        out = self.head(h).permute(0, 2, 1)
        raw_target_residual = out[:, :, -1]
        if self.nwp_conditioned_decoder:
            if future_x is None:
                raise ValueError("NWP-conditioned decoder requires audited future covariates")
            # FACT/CorPatch already mixes variables. The target-channel tokens
            # are K,V; NWP is Q only. The untouched historical head above is
            # the primary prediction and this adapter contributes a correction.
            target_memory = memory_by_channel[:, -1]
            correction = self.nwp_decoder(target_memory, future_x, routing_state)
            target_out = out[:, :, -1] + correction
            out = torch.cat([out[:, :, :-1], target_out.unsqueeze(-1)], dim=-1)
        if self.revin:
            out = out * stdev[:, 0, -1].view(-1, 1, 1) + means[:, 0, -1].view(-1, 1, 1)
        pred = out[:, :, -1]
        if self.future_covariates and future_x is not None:
            future_state = self.future_encoder(future_x).reshape(x.shape[0], -1)
            future_delta = self.future_head(future_state)
            if self.revin:
                future_delta = future_delta * stdev[:, 0, -1].view(-1, 1)
            gate_source = routing_state if routing_state is not None else x.mean(dim=1)
            future_gate = torch.sigmoid(self.future_gate(gate_source))
            pred = pred + future_gate * future_delta
        # Training may attach a weak auxiliary objective here so the routed
        # experts receive gradients before the conservative solar anchor.
        self.last_candidate_pred = pred
        if (self.reference_residual_enabled
                and self.reference_residual_active):
            base = self._solar_trajectory_base(routing_x, future_x)
            reference_features = self._historical_reference_features(routing_x)
            output_state = self._intervene_state(
                routing_state, self.output_state_intervention
            )
            reliability_logits = self.reference_reliability_router(
                torch.cat([output_state, reference_features], dim=-1)
            )
            reliability = torch.clamp(
                float(self.reference_strength)
                * torch.sigmoid(reliability_logits),
                0.0, 1.0,
            )
            pred = pred + reliability * (base - pred)
            self.last_reference_reliability = reliability.detach()
            self.last_reference_history_features = reference_features.detach()
            self.last_output_correction = (1.0 - reliability).detach()
        elif self.solar_residual_output:
            base = self._solar_trajectory_base(routing_x, future_x)
            residual = (raw_target_residual * stdev[:, 0, -1].view(-1, 1)
                        if self.revin else raw_target_residual)
            pred = base + residual
        elif self.output_anchor:
            # Very-short-term PV has a strong persistence prior.  The physical
            # state decides how far each horizon may depart from that anchor:
            # stable/clear windows can retain persistence, whereas ramp/cloud
            # states can admit a larger learned correction.
            if self.solar_trajectory_anchor:
                base = self._solar_trajectory_base(routing_x, future_x)
            else:
                base = routing_x[:, -1, -1].unsqueeze(1).expand_as(pred)
            if self.static_output_anchor:
                correction = torch.sigmoid(self.output_anchor_logits).unsqueeze(0).expand(pred.shape[0], -1)
            else:
                output_state = self._intervene_state(
                    routing_state, self.output_state_intervention
                )
                correction_logits = self.output_router(output_state)
                if (self.scale_prompt_output_router is not None
                        and self.last_scale_prompt_state is not None
                        and self.semantic_enabled):
                    prompt_state = self.last_scale_prompt_state
                    prompt_context = torch.cat([
                        prompt_state, prompt_state * output_state,
                    ], dim=-1)
                    correction_logits = (
                        correction_logits
                        + float(self.semantic_strength)
                        * self.scale_prompt_output_router(prompt_context)
                    )
                correction = torch.sigmoid(correction_logits)
            if self.output_correction_floor > 0.0:
                correction = (self.output_correction_floor
                              + (1.0 - self.output_correction_floor) * correction)
            correction = torch.clamp(
                float(self.output_correction_scale) * correction, 0.0, 1.0,
            )
            pred = base + correction * (pred - base)
            self.last_output_correction = correction.detach()
        if (self.semantic_output_head is not None and self.semantic_enabled
                and prompt_state is not None):
            semantic_context = (prompt_state if self.prompt_only_semantic_output else
                                torch.cat([routing_state, prompt_state], dim=-1))
            semantic_output_gate = torch.sigmoid(self.semantic_output_gate(semantic_context))
            semantic_output_delta = self.semantic_output_head(semantic_context)
            pred = pred + self.semantic_strength * semantic_output_gate * semantic_output_delta
            self.last_semantic_output_gate = semantic_output_gate.detach()
        if self.probabilistic_output:
            scale_logits = self.uncertainty_log_scale.unsqueeze(0).expand(
                pred.shape[0], -1,
            )
            if self.uncertainty_head is not None:
                if routing_state is None:
                    raise ValueError(
                        "state-conditioned uncertainty requires a physical routing state"
                    )
                scale_logits = scale_logits + self.uncertainty_head(routing_state)
            self.last_predictive_scale = (
                F.softplus(scale_logits) + 1e-4
            ) * self.uncertainty_calibration
        if self.quantile_output:
            residual_members = self.quantile_base.unsqueeze(0).expand(
                pred.shape[0], -1, -1,
            )
            if self.quantile_head is not None:
                if routing_state is None:
                    raise ValueError(
                        "state-conditioned quantiles require a physical routing state"
                    )
                residual_members = residual_members + self.quantile_head(
                    routing_state
                ).reshape(pred.shape[0], pred.shape[1], self.quantile_members)
            # Sorting is differentiable almost everywhere and prevents
            # crossing quantiles without changing the deterministic mean.
            residual_members = torch.sort(residual_members, dim=-1).values
            self.last_predictive_members = pred.unsqueeze(-1) + residual_members
        return pred


class Windows(Dataset):
    def __init__(self, x, starts, seq_len, horizon, cycle_len, cycle_ids=None):
        self.x, self.starts, self.seq_len, self.horizon, self.cycle_len = x, starts, seq_len, horizon, cycle_len
        self.cycle_ids = cycle_ids

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        cycle = int(self.cycle_ids[s]) if self.cycle_ids is not None else int(s % self.cycle_len)
        return (torch.tensor(self.x[s:s + self.seq_len]), torch.tensor(self.x[s + self.seq_len:s + self.seq_len + self.horizon, -1]), torch.tensor(cycle))


def load_windows(path: str, seq_len: int, horizon: int, max_windows: int,
                 target_col: str = "auto", cycle_len: int = 96,
                 observed_policy: str = "all", rated_capacity: float | None = None,
                 future_feature_policy: str = "none",
                 forecast_start_hour: int | None = None,
                 max_train_windows: int | None = None,
                 split_train_fraction: float = 0.70,
                 split_validation_end_fraction: float = 0.85,
                 train_sampling_policy: str = "uniform",
                 ramp_enrichment_fraction: float = 0.30,
                 max_val_windows: int | None = None,
                 max_test_windows: int | None = None):
    """Load the legacy Target format and the downloaded Australian CSV formats.

    The loader keeps only numeric channels with at least 80% finite values. This
    prevents DKASC/AEMO metadata columns (region/type) and sparsely populated
    optional sensors from wiping out the whole sequence.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk", on_bad_lines="skip", engine="python")
    if target_col == "auto":
        candidates = ["Target", "Power", "Active_Power", "SolarGeneration", "power_mw", "POWER"]
        target_col = next((c for c in candidates if c in df.columns), None)
    if target_col is None or target_col not in df.columns:
        raise ValueError(f"Cannot infer target column in {path}; columns={list(df.columns)}")
    target = pd.to_numeric(df[target_col], errors="coerce")
    quality_names = {"TargetObserved", "ObservedFlag", "QualityFlag", "DataQuality"}
    observed_col = next((c for c in quality_names if c in df.columns), None)
    if observed_col is not None:
        target_observed = pd.to_numeric(df[observed_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    else:
        target_observed = target.notna().to_numpy()
    time_col = next((c for c in ["timestamp", "Timestamp", "Time", "date", "date.1", "interval_datetime"] if c in df.columns), None)
    timestamps = pd.to_datetime(df[time_col], errors="coerce") if time_col is not None else None
    if timestamps is not None:
        positive_dt = timestamps.sort_values().diff().dt.total_seconds().div(3600.0)
        positive_dt = positive_dt[(positive_dt > 0) & np.isfinite(positive_dt)]
        sample_hours = float(positive_dt.median()) if len(positive_dt) else 24.0 / float(cycle_len)
    else:
        sample_hours = 24.0 / float(cycle_len)
    # ``kpv`` in the released GEFCom2014/TabPFN files is computed directly
    # from the response (kpv = y * 1000 / gtic).  Keeping it would leak the
    # target into every historical feature tensor and make the benchmark
    # uninterpretable.  Exclude it centrally so raw author CSVs stay usable.
    target_derived = {"kpv"}
    excluded = {target_col, "date", "date.1", "timestamp", "Timestamp", "Time",
                "interval_datetime", "lastchanged", *target_derived}
    num = df[[c for c in df.columns if c not in excluded]].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if float(num[c].notna().mean()) >= 0.80]
    feat = num[keep].to_numpy(np.float32)
    feature_names = list(keep)
    if timestamps is not None:
        # Explicit daily phase/daylight channels make the learned cycle
        # interpretable and provide the solar model with a stable clock.
        minute = (timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60.0).to_numpy()
        phase = minute / 1440.0
        # Deterministic solar geometry derived only from timestamp.  This is
        # exactly recoverable from ClockCos, so it adds no information to the
        # learned models, but gives SmartPersistence an explicit audited
        # clear-sky trajectory on datasets that do not ship one.
        solar_geometry = np.maximum(0.0, -np.cos(2 * np.pi * phase))
        time_feat = np.column_stack([
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            solar_geometry,
            ((minute >= 360) & (minute <= 1200)).astype(np.float32),
        ])
        feat = np.column_stack([feat, time_feat]).astype(np.float32)
        feature_names.extend([
            "ClockSin", "ClockCos", "SolarGeometryProxy", "DaylightHeuristic",
        ])
    raw = np.column_stack([feat, target.to_numpy(np.float32)]).astype(np.float32)
    feature_names.append(target_col)
    valid = np.isfinite(raw).all(1)
    if timestamps is not None:
        valid &= timestamps.notna().to_numpy()
    raw = raw[valid]
    target_observed = target_observed[valid]
    if timestamps is not None:
        timestamps = timestamps.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    n = len(raw)
    if not 0.0 < split_train_fraction < split_validation_end_fraction < 1.0:
        raise ValueError(
            "split fractions must satisfy "
            "0 < split_train_fraction < split_validation_end_fraction < 1"
        )
    split1 = int(split_train_fraction * n)
    split2 = int(split_validation_end_fraction * n)
    mu, sd = raw[:split1].mean(0, keepdims=True), raw[:split1].std(0, keepdims=True) + 1e-6
    quality_indices = [i for i, name in enumerate(feature_names) if name in quality_names]
    # Preserve the semantic 0/1 scale of quality flags so the model can use a
    # window mean directly as an observation reliability.  Standardizing a
    # binary flag would make the gate dataset-dependent and hard to interpret.
    if quality_indices:
        mu[:, quality_indices] = 0.0
        sd[:, quality_indices] = 1.0
    x = (raw - mu) / sd

    # A future tensor is a different information set from the historical
    # tensor.  It must be explicitly audited instead of reusing every numeric
    # column at t+1...t+H.  In particular, target availability flags and any
    # target-derived quantities are forbidden even when they look harmless.
    gefcom_nwp_names = {
        "GHI", "ClearSkyGHI", "DNI", "ClearSkyDNI", "DHI", "ClearSkyDHI",
        "GTI", "ClearSkyGTI", "ClearSkyProxy", "SolarZenithCos",
        "SolarAzimuthSin", "SolarAzimuthCos", "SolarDaySin", "SolarDayCos",
        "CellTemperature", "ModuleTemperature", "SurfaceLongwaveRadiation",
        "SurfaceSolarRadiation", "CloudLiquidWater", "CloudIceWater",
        "SurfacePressure", "RelativeHumidity", "TotalCloudCover",
        "AirTemperature", "Precipitation", "WindSpeed",
        "ExtraterrestrialIrradiance", "ClockSin", "ClockCos", "DaylightHeuristic",
        # Names in the unmodified 2026 Solar Energy supplementary release.
        "ghi", "ghic", "bni", "bnic", "dhi", "dhic", "gti", "gtic",
        "zen", "tcell", "tmod", "strd", "tsr", "tclw", "tciw", "sp",
        "rh", "tcc", "t2m", "tp", "ws", "azi", "ext",
    }
    gefcom_core_names = {
        "GHI", "DNI", "AirTemperature", "RelativeHumidity", "WindSpeed",
        "TotalCloudCover",
    }
    if future_feature_policy == "none":
        future_indices = []
    elif future_feature_policy == "gefcom_nwp":
        nwp_signature = {"TotalCloudCover", "CloudLiquidWater", "tcc", "tclw"}
        if not nwp_signature.intersection(feature_names):
            raise ValueError(
                "gefcom_nwp policy requested, but ECMWF/GEFCom NWP signature columns are absent"
            )
        future_indices = [i for i, name in enumerate(feature_names) if name in gefcom_nwp_names]
    elif future_feature_policy == "gefcom_nwp_core":
        nwp_signature = {"TotalCloudCover", "GHI", "DNI"}
        if not nwp_signature.issubset(feature_names):
            raise ValueError(
                "gefcom_nwp_core policy requested, but required GEFCom columns are absent"
            )
        future_indices = [i for i, name in enumerate(feature_names) if name in gefcom_core_names]
    else:
        raise ValueError(
            "future_feature_policy must be none/gefcom_nwp/gefcom_nwp_core, "
            f"got {future_feature_policy}"
        )
    forbidden_future = {target_col, *quality_names, *target_derived}
    leaked = [feature_names[i] for i in future_indices if feature_names[i] in forbidden_future]
    if leaked or (len(feature_names) - 1) in future_indices:
        raise AssertionError(f"target or target-derived feature entered future tensor: {leaked}")
    if future_feature_policy != "none" and not future_indices:
        raise ValueError("future feature audit produced an empty covariate set")

    starts = np.arange(n - seq_len - horizon + 1)
    total_len = seq_len + horizon
    if timestamps is not None and len(starts):
        # Rows separated by a missing timestamp must never become adjacent in
        # a forecasting window.  The prefix sum makes this check O(n).
        dt = timestamps.diff().dt.total_seconds().div(3600.0).to_numpy()
        bad_transition = np.zeros(n, dtype=np.int64)
        bad_transition[1:] = (~np.isfinite(dt[1:]) |
                              (np.abs(dt[1:] - sample_hours) > max(sample_hours * 0.05, 1e-6)))
        bad_prefix = np.concatenate([[0], np.cumsum(bad_transition)])
        end = starts + total_len
        continuous = (bad_prefix[end] - bad_prefix[starts + 1]) == 0
        starts = starts[continuous]
    if forecast_start_hour is not None:
        if timestamps is None:
            raise ValueError("fixed forecast start hour requires a timestamp column")
        if not 0 <= forecast_start_hour <= 23:
            raise ValueError("forecast_start_hour must be in [0, 23]")
        forecast_time = timestamps.iloc[starts + seq_len]
        issued_block = ((forecast_time.dt.hour.to_numpy() == forecast_start_hour)
                        & (forecast_time.dt.minute.to_numpy() == 0)
                        & (forecast_time.dt.second.to_numpy() == 0))
        starts = starts[issued_block]
    if observed_policy not in {"all", "any", "none"}:
        raise ValueError(f"observed_policy must be all/any/none, got {observed_policy}")
    if observed_policy != "none" and len(starts):
        obs_prefix = np.concatenate([[0], np.cumsum(target_observed.astype(np.int64))])
        y_start = starts + seq_len
        observed_count = obs_prefix[y_start + horizon] - obs_prefix[y_start]
        keep_observed = observed_count == horizon if observed_policy == "all" else observed_count > 0
        starts = starts[keep_observed]
    if timestamps is not None:
        minute = (timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60.0).to_numpy()
        cycle_ids = np.rint(minute / 1440.0 * cycle_len).astype(np.int64) % cycle_len
    else:
        cycle_ids = np.arange(n, dtype=np.int64) % cycle_len
    train_all = starts[starts + seq_len + horizon <= split1]
    val_all = starts[(starts >= split1 - seq_len) & (starts + seq_len + horizon <= split2)]
    test_all = starts[starts >= split2 - seq_len]

    def evenly_sample(candidates: np.ndarray, limit: int) -> np.ndarray:
        """Cover the entire chronological split without concentrating on one regime.

        Taking the first/last contiguous block can accidentally select only
        night-time or one season.  Uniform deterministic subsampling preserves
        the chronological split while covering its full weather/solar range.
        """
        if len(candidates) <= limit:
            return candidates
        pos = np.linspace(0, len(candidates) - 1, num=limit, dtype=np.int64)
        return candidates[pos]

    # Score only the historical part of each training window.  The score is
    # the mean absolute PV ramp over genuinely observed adjacent points.  It
    # never uses the forecast target interval, validation data or test data.
    pair_ramp = np.abs(np.diff(raw[:, -1])).astype(np.float64)
    pair_observed = (target_observed[:-1] & target_observed[1:]).astype(np.float64)
    ramp_sum_prefix = np.concatenate([[0.0], np.cumsum(pair_ramp * pair_observed)])
    ramp_count_prefix = np.concatenate([[0.0], np.cumsum(pair_observed)])

    def historical_ramp_score(candidates: np.ndarray) -> np.ndarray:
        if not len(candidates):
            return np.empty(0, dtype=np.float64)
        end = candidates + seq_len - 1
        sums = ramp_sum_prefix[end] - ramp_sum_prefix[candidates]
        counts = ramp_count_prefix[end] - ramp_count_prefix[candidates]
        return sums / np.maximum(counts, 1.0)

    def ramp_enriched_sample(candidates: np.ndarray, limit: int) -> np.ndarray:
        """Mix chronological coverage with high-ramp historical regimes.

        Seventy percent of the default candidate is uniformly distributed over
        time.  The remainder is sampled chronologically from the highest-ramp
        training-only pool.  This keeps seasons covered while exposing the
        model to more cloud-transition histories.
        """
        if len(candidates) <= limit:
            return candidates
        if not 0.0 <= ramp_enrichment_fraction < 1.0:
            raise ValueError("ramp_enrichment_fraction must be in [0, 1)")
        hard_n = int(round(limit * ramp_enrichment_fraction))
        uniform_n = limit - hard_n
        uniform = evenly_sample(candidates, uniform_n)
        remaining = candidates[~np.isin(candidates, uniform, assume_unique=True)]
        if hard_n <= 0 or not len(remaining):
            return evenly_sample(candidates, limit)

        scores = historical_ramp_score(remaining)
        # Use a broad high-ramp pool and then sample it chronologically.  Taking
        # only the absolute top windows would select many near-duplicates from
        # one storm and reduce seasonal coverage.
        pool_n = min(len(remaining), max(hard_n, 5 * hard_n))
        if pool_n < len(remaining):
            top_pos = np.argpartition(scores, -pool_n)[-pool_n:]
            hard_pool = np.sort(remaining[top_pos])
        else:
            hard_pool = remaining
        hard = evenly_sample(hard_pool, min(hard_n, len(hard_pool)))
        selected = np.unique(np.concatenate([uniform, hard]))
        if len(selected) < limit:
            unused = candidates[~np.isin(candidates, selected, assume_unique=True)]
            fill = evenly_sample(unused, min(limit - len(selected), len(unused)))
            selected = np.unique(np.concatenate([selected, fill]))
        return np.sort(selected[:limit])

    # ``max_windows`` remains the frozen evaluation-budget control used by all
    # historical experiments.  ``max_train_windows`` can densify only the
    # training split, leaving the exact validation/test timestamps unchanged.
    # This isolates the effect of greater training-data utilisation without
    # changing the forecasting graph or the evaluation question.
    train_limit = max_windows if max_train_windows is None else max_train_windows
    if train_limit < 1:
        raise ValueError("max_train_windows must be at least 1")
    if train_sampling_policy == "uniform":
        train = evenly_sample(train_all, train_limit)
    elif train_sampling_policy == "ramp_enriched":
        train = ramp_enriched_sample(train_all, train_limit)
    elif train_sampling_policy == "epoch_uniform":
        # ``train`` remains the fixed, chronology-covering subset used by
        # frozen-language prompt preparation.  The runner consumes the full
        # private pool through an epoch-varying stratified sampler only during
        # numerical/FSRA training.  Validation and test starts are unchanged.
        train = evenly_sample(train_all, train_limit)
    else:
        raise ValueError(
            "train_sampling_policy must be uniform/ramp_enriched/epoch_uniform, "
            f"got {train_sampling_policy}"
        )
    default_eval_limit = max(1, max_windows // 4)
    val_limit = default_eval_limit if max_val_windows is None else max_val_windows
    test_limit = default_eval_limit if max_test_windows is None else max_test_windows
    if val_limit == 0:
        val = val_all
    elif val_limit > 0:
        val = evenly_sample(val_all, val_limit)
    else:
        raise ValueError("max_val_windows must be non-negative; 0 means all")
    if test_limit == 0:
        test = test_all
    elif test_limit > 0:
        test = evenly_sample(test_all, test_limit)
    else:
        raise ValueError("max_test_windows must be non-negative; 0 means all")
    empirical_capacity = float(raw[:split1, -1].max())
    capacity = float(rated_capacity) if rated_capacity is not None else empirical_capacity
    stats = {
        "target_mu": float(mu[0, -1]),
        "target_sd": float(sd[0, -1]),
        "seq_len": int(seq_len),
        "cycle_len": int(cycle_len),
        "capacity": capacity,
        "capacity_source": "rated_metadata" if rated_capacity is not None else "training_empirical_max",
        "empirical_training_max": empirical_capacity,
        "dt_hours": sample_hours,
        "horizon_hours": float(horizon * sample_hours),
        "target_observed": target_observed.astype(np.float32),
        "observed_policy": observed_policy,
        "observed_fraction_all": float(target_observed.mean()),
        "future_feature_policy": future_feature_policy,
        "future_indices": future_indices,
        "future_features": [feature_names[i] for i in future_indices],
        "core_nwp_history_indices": [i for i, name in enumerate(feature_names)
                                     if name in gefcom_core_names],
        "core_history_indices": [i for i, name in enumerate(feature_names)
                                 if name in gefcom_core_names or name == target_col],
        "forecast_start_hour": forecast_start_hour,
        "available_train_windows": int(len(train_all)),
        "available_val_windows": int(len(val_all)),
        "available_test_windows": int(len(test_all)),
        "selected_train_windows": int(len(train)),
        "selected_val_windows": int(len(val)),
        "selected_test_windows": int(len(test)),
        "max_train_windows": int(train_limit),
        "max_eval_windows": int(max_windows),
        "max_val_windows": int(val_limit),
        "max_test_windows": int(test_limit),
        "split_train_fraction": float(split_train_fraction),
        "split_validation_end_fraction": float(split_validation_end_fraction),
        "train_sampling_policy": train_sampling_policy,
        "_epoch_train_pool": (
            train_all.copy() if train_sampling_policy == "epoch_uniform" else None
        ),
        "ramp_enrichment_fraction": (
            float(ramp_enrichment_fraction)
            if train_sampling_policy == "ramp_enriched" else 0.0
        ),
        "available_train_history_ramp_mean": float(
            historical_ramp_score(train_all).mean()
        ) if len(train_all) else 0.0,
        "selected_train_history_ramp_mean": float(
            historical_ramp_score(train).mean()
        ) if len(train) else 0.0,
    }
    physics_names = {
        "SolarZenithCos", "SolarGeometryProxy", "ClearSkyGHIProxy", "ClearSkyProxy",
        "SolarDaySin", "SolarDayCos",
        "SolarElevationMask", "ClockSin", "ClockCos", "DaylightHeuristic",
        "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation",
        "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted",
        "GHI", "DNI", "DHI", "TSI", "Weather_Daily_Rainfall",
        "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
        "AirTemperature", "RelativeHumidity", "WindSpeed", "Wind_Speed",
        "ClearSkyGHI", "ClearSkyDNI", "ClearSkyDHI", "GTI", "ClearSkyGTI",
        "SolarAzimuthSin", "SolarAzimuthCos", "CellTemperature", "ModuleTemperature",
        "SurfaceLongwaveRadiation", "SurfaceSolarRadiation", "CloudLiquidWater",
        "CloudIceWater", "SurfacePressure", "TotalCloudCover", "Precipitation",
        "ExtraterrestrialIrradiance",
        # GEFCom2014 names in the 2026 Solar Energy TabPFN release.
        "ghi", "ghic", "bni", "bnic", "dhi", "dhic", "gti", "gtic",
        "zen", "tcell", "tmod", "strd", "tsr", "tclw", "tciw", "sp",
        "rh", "tcc", "t2m", "tp", "ws", "azi", "ext",
        target_col,
        *quality_names,
    }
    stats["feature_names"] = feature_names
    stats["feature_mu"] = mu.reshape(-1).astype(float).tolist()
    stats["feature_sd"] = sd.reshape(-1).astype(float).tolist()
    stats["physics_indices"] = [i for i, name in enumerate(feature_names) if name in physics_names]
    stats["physics_features"] = [feature_names[i] for i in stats["physics_indices"]]
    stats["quality_indices"] = quality_indices
    stats["quality_features"] = [feature_names[i] for i in quality_indices]
    clear_candidates = (
        "ClearSkyGHIProxy", "ClearSkyGHI", "ClearSkyGTI", "ghic", "gtic",
        "SolarGeometryProxy", "SolarZenithCos", "ClearSkyProxy",
    )
    clear_name = next((name for name in clear_candidates if name in feature_names), None)
    daylight_candidates = ("SolarElevationMask", "DaylightHeuristic")
    daylight_name = next((name for name in daylight_candidates if name in feature_names), None)
    solar_future_names = ([clear_name] if clear_name is not None else [])
    if daylight_name is not None and daylight_name not in solar_future_names:
        solar_future_names.append(daylight_name)
    stats["solar_future_indices"] = [feature_names.index(name) for name in solar_future_names]
    stats["solar_future_features"] = solar_future_names
    stats["clear_sky_feature"] = clear_name
    stats["daylight_feature"] = daylight_name
    if clear_name is not None:
        clear_idx = feature_names.index(clear_name)
        stats["clear_sky_index"] = clear_idx
        stats["clear_sky_training_max"] = float(max(raw[:split1, clear_idx].max(), 1e-6))
    else:
        stats["clear_sky_index"] = None
        stats["clear_sky_training_max"] = None

    non_spectral_names = {
        "SolarZenithCos", "SolarGeometryProxy", "ClearSkyGHIProxy", "ClearSkyProxy",
        "ClearSkyGHI", "ClearSkyDNI", "ClearSkyDHI", "ClearSkyGTI",
        "SolarDaySin", "SolarDayCos", "SolarAzimuthSin", "SolarAzimuthCos",
        "SolarElevationMask", "ClockSin", "ClockCos", "DaylightHeuristic",
        "ExtraterrestrialIrradiance", "ext", "ghic", "bnic", "dhic", "gtic",
        *quality_names,
    }
    stats["spectral_indices"] = [i for i, name in enumerate(feature_names)
                                 if name not in non_spectral_names]
    stats["spectral_features"] = [feature_names[i] for i in stats["spectral_indices"]]
    return x, train, val, test, cycle_ids, stats


def criterion(pred, y, kind, lo, hi, stats):
    loss = F.mse_loss(pred, y)
    if "ramp" in kind and pred.shape[1] > 1:
        if "phys" in kind:
            p_phys = pred * stats["target_sd"] + stats["target_mu"]
            y_phys = y * stats["target_sd"] + stats["target_mu"]
            cap = max(stats["capacity"], 1e-6)
            loss = loss + .05 * F.l1_loss(torch.diff(p_phys, dim=1) / (stats["dt_hours"] * cap),
                                          torch.diff(y_phys, dim=1) / (stats["dt_hours"] * cap))
        else:
            loss = loss + .2 * F.l1_loss(torch.diff(pred, dim=1), torch.diff(y, dim=1))
    if "bound" in kind:
        loss = loss + .05 * (F.relu(lo - pred).mean() + F.relu(pred - hi).mean())
    if "phys" in kind:
        p_phys = pred * stats["target_sd"] + stats["target_mu"]
        y_phys = y * stats["target_sd"] + stats["target_mu"]
        cap = stats["capacity"]
        loss = loss + .02 * (F.relu(-p_phys / cap).mean() + F.relu((p_phys - cap) / cap).mean())
        energy_gap = torch.abs(p_phys.mean(dim=1) - y_phys.mean(dim=1)).mean() / cap
        loss = loss + .02 * energy_gap
    return loss


@torch.no_grad()
def evaluate(model, loader, device, kind, lo, hi, stats):
    model.eval(); ys = []; ps = []; ls = []; scale_ws = []; tf_ws = []; temporal_ws = []
    for x, y, c in loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        p = model(x, c)
        if model.last_scale_weights is not None:
            scale_ws.append(model.last_scale_weights.cpu().numpy())
        if model.last_tf_weights is not None:
            tf_ws.append(model.last_tf_weights.cpu().numpy())
        if model.last_temporal_weights is not None:
            temporal_ws.append(model.last_temporal_weights.cpu().numpy())
        ls.append(criterion(p, y, kind, lo, hi, stats).item()); ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y_seq, p_seq = np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)
    y, p = y_seq.ravel(), p_seq.ravel()
    mse = float(np.mean((p - y) ** 2))
    p_phys = p_seq * stats["target_sd"] + stats["target_mu"]
    y_phys = y_seq * stats["target_sd"] + stats["target_mu"]
    violation = float(np.mean(np.maximum(-p_phys, 0.0) + np.maximum(p_phys - stats["capacity"], 0.0)))
    energy_gap = float(np.mean(np.abs(p_phys.sum(axis=1) - y_phys.sum(axis=1))) / (stats["capacity"] * p_seq.shape[1] + 1e-6))
    err_phys = p_phys - y_phys
    mse_phys = float(np.mean(err_phys ** 2))
    rmse_phys = float(np.sqrt(mse_phys))
    mae_phys = float(np.mean(np.abs(err_phys)))
    capacity = max(float(stats["capacity"]), 1e-6)
    denom = float(np.sum((y_phys - y_phys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err_phys ** 2)) / max(denom, 1e-12)
    if p_seq.shape[1] > 1:
        ramp_err = np.diff(p_phys, axis=1) - np.diff(y_phys, axis=1)
        ramp_mae_norm = float(np.mean(np.abs(ramp_err)) / (capacity * stats["dt_hours"] + 1e-6))
    else:
        ramp_mae_norm = 0.0
    result = {"loss": float(np.mean(ls)), "mse": mse, "rmse": float(np.sqrt(mse)),
              "mae": float(np.mean(np.abs(p - y))), "rmse_physical": rmse_phys,
              "mae_physical": mae_phys, "nrmse_capacity": rmse_phys / capacity,
              "nmae_capacity": mae_phys / capacity, "r2": r2,
              "ramp_mae_norm": ramp_mae_norm, "bound_violation": violation,
              "energy_gap": energy_gap}
    if scale_ws:
        result["scale_weights"] = np.concatenate(scale_ws, axis=0).mean(axis=0).tolist()
    if tf_ws:
        result["time_frequency_weights"] = np.concatenate(tf_ws, axis=0).mean(axis=0).tolist()
    if temporal_ws:
        result["gtr_ldrive_weights"] = np.concatenate(temporal_ws, axis=0).mean(axis=0).tolist()
    if model.abs_route_logit is not None:
        result["absolute_route_weight"] = float(torch.sigmoid(model.abs_route_logit).detach().cpu())
    return result


def run_variant(name, args, data, device):
    x, tr, va, te, cycle_ids, stats = data
    channels = x.shape[1]
    mode, enc, kind = name.split("|")
    loaders = [DataLoader(Windows(x, idx, args.seq_len, args.horizon, args.cycle_len, cycle_ids), args.batch, shuffle=(i == 0)) for i, idx in enumerate([tr, va, te])]
    model = RPGTR(args.seq_len, args.horizon, channels, args.cycle_len, args.patch_len, args.stride,
                  args.d_model, enc, mode, revin=args.revin,
                  physics_indices=stats.get("physics_indices"),
                  quality_indices=stats.get("quality_indices")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lo, hi = float(x[: len(x) * 7 // 10, -1].min()), float(x[: len(x) * 7 // 10, -1].max())
    best_val = None
    for _ in range(args.epochs):
        model.train()
        for xb, yb, cb in loaders[0]:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb, cb), yb, kind, lo, hi, stats)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val = evaluate(model, loaders[1], device, kind, lo, hi, stats)
        if best_val is None or val["loss"] < best_val["loss"]:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, loaders[2], device, kind, lo, hi, stats)
    return {"variant": name, "val": best_val, "test": test, "params": sum(p.numel() for p in model.parameters())}


def main():
    here = Path(__file__).resolve().parents[1]
    default_data = next(here.rglob("feng2019.csv"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(default_data))
    ap.add_argument("--out", default=str(Path(__file__).with_name("rp_gtr_ablation_results.json")))
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--cycle-len", type=int, default=96)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument("--target-col", default="auto")
    ap.add_argument("--revin", action="store_true", help="use M5-style per-window normalization and de-normalization")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args(); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_windows(args.data, args.seq_len, args.horizon, args.max_windows, args.target_col, args.cycle_len)
    variants = args.variants or [
        "patch_only|coherence|mse",
        "mkan_only|coherence|mse",
        "sequential|coherence|mse",
        "parallel|coherence|mse",
        "parallel_ms|coherence|mse",
        "parallel_ms_phys|coherence|mse",
        "tf_parallel_ms_static|coherence|mse",
        "tf_parallel_ms|coherence|mse",
        "tf_residual_ms_static|coherence|mse",
        "tf_residual_ms|coherence|mse",
        "parallel|fact|mse",
        "parallel|coherence|mse_ramp",
        "parallel|fact|mse_ramp_bound",
    ]
    results = []
    print(f"device={device} data={args.data}")
    for v in variants:
        print("running", v)
        # Reset every variant so model order does not change initialization or
        # DataLoader shuffling in an ablation table.
        seed_all(args.seed)
        r = run_variant(v, args, data, device); results.append(r); print(json.dumps(r, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
