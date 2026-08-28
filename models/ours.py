"""Time-frequency PV forecasting with CMA and multi-scale CorPatch fusion."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from layers.semantic_alignment import AdaptiveMultiHeadCMAResidual


class RevIN(nn.Module):
    """Reversible instance normalization over historical time."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, channels))

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalize_target(
        self, prediction: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        weight = self.weight[..., -1:].clamp_min(self.eps)
        restored = (prediction.unsqueeze(-1) - self.bias[..., -1:]) / weight
        return (restored * std[..., -1:] + mean[..., -1:]).squeeze(-1)


class SourceGTR(nn.Module):
    """Cycle retrieval and local/global fusion from the source GTR block."""

    def __init__(
        self,
        seq_len: int,
        channels: int,
        cycle_len: int,
        period_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.cycle_len = cycle_len
        self.period_len = max(2, min(int(period_len), seq_len))
        self.cycle_query = nn.Parameter(torch.zeros(cycle_len, channels))
        self.mapping = nn.Linear(seq_len, seq_len)
        kernel = 1 + 2 * (self.period_len // 2)
        self.local_global_fusion = nn.Conv2d(
            1,
            1,
            kernel_size=(2, kernel),
            padding=(0, self.period_len // 2),
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        if (length, channels) != (self.seq_len, self.channels):
            raise ValueError(
                f"GTR expected [B,{self.seq_len},{self.channels}], got {tuple(x.shape)}"
            )
        offsets = torch.arange(length, device=x.device)
        positions = (cycle[:, None] + offsets[None, :]) % self.cycle_len
        query = self.cycle_query[positions].transpose(1, 2)
        local = x.transpose(1, 2)
        paired = torch.stack([local, self.mapping(query)], dim=2)
        paired = paired.reshape(batch * channels, 1, 2, length)
        fused = self.local_global_fusion(paired).reshape(batch, channels, length)
        return self.dropout(fused).transpose(1, 2)


class LDrive(nn.Module):
    """Local derivative-context dynamics with a gated GRU residual."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())
        self.gru = nn.GRU(channels, channels, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.enhance_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 1, 0))
        filtered = delta * self.gate(torch.cat([x, delta], dim=-1))
        context, _ = self.gru(filtered)
        return x + self.enhance_weight * self.dropout(context)


class TemporalBranchGate(nn.Module):
    """Fuse GTR and L-Drive per sample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, 2 * channels)
        self.router = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, gtr: torch.Tensor, ldrive: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary = torch.cat([gtr.mean(1), ldrive.mean(1)], dim=-1)
        weights = torch.softmax(self.router(summary), dim=-1)
        fused = weights[:, 0, None, None] * gtr
        fused = fused + weights[:, 1, None, None] * ldrive
        return fused, weights


class ChebyKANLinear(nn.Module):
    """Chebyshev-polynomial KAN mapping."""

    def __init__(self, input_dim: int, output_dim: int, degree: int = 3) -> None:
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.coefficients, std=1.0 / (input_dim * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        polynomials = [torch.ones_like(bounded), bounded]
        for _ in range(2, self.degree + 1):
            polynomials.append(2 * bounded * polynomials[-1] - polynomials[-2])
        basis = torch.stack(polynomials[: self.degree + 1], dim=-1)
        return torch.einsum("...id,iod->...o", basis, self.coefficients)


class SpectralMKAN(nn.Module):
    """rFFT → complex MKAN → irFFT frequency residual."""

    def __init__(self, channels: int, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.real_mkan = ChebyKANLinear(channels, d_model)
        self.imag_mkan = ChebyKANLinear(channels, d_model)
        self.back = nn.Linear(d_model, channels)
        self.frequency_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        nn.init.zeros_(self.back.weight)
        nn.init.zeros_(self.back.bias)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        gate = self.frequency_gate(state).unsqueeze(1)
        real = self.back(self.real_mkan(spectrum.real) * gate)
        imag = self.back(self.imag_mkan(spectrum.imag) * gate)
        corrected = torch.complex(real, imag)
        return torch.fft.irfft(corrected, n=self.seq_len, dim=1)


class PhysicalStateEncoder(nn.Module):
    """Encode historical physical state without future covariates."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        physics_indices: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        indices = list(physics_indices or range(channels))
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.encoder = nn.GRU(len(indices), d_model, batch_first=True)
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x.index_select(-1, self.indices))
        return self.summary(encoded[:, -1])


class TimeFrequencyRouter(nn.Module):
    """Route temporal and spectral residuals from physical state."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.prior_logits = nn.Parameter(torch.tensor([-0.5, -1.0]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(state) + self.prior_logits)


class MultiScaleVariablePatch(nn.Module):
    """Patch each variable independently at physical time scales."""

    def __init__(
        self, seq_len: int, d_model: int, patch_lengths: Iterable[int]
    ) -> None:
        super().__init__()
        self.patch_lengths = list(
            dict.fromkeys(max(1, min(seq_len, int(p))) for p in patch_lengths)
        )
        self.strides = [max(1, patch // 2) for patch in self.patch_lengths]
        self.embeddings = nn.ModuleList(
            nn.Linear(patch, d_model) for patch in self.patch_lengths
        )
        self.positions = nn.ParameterList()
        for patch, stride in zip(self.patch_lengths, self.strides, strict=True):
            count = 1 + (seq_len - patch) // stride
            self.positions.append(nn.Parameter(torch.zeros(1, 1, count, d_model)))
        self.scales = list(zip(self.patch_lengths, self.strides, strict=True))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        source = x.transpose(1, 2)
        outputs = []
        for patch, stride, embedding, position in zip(
            self.patch_lengths,
            self.strides,
            self.embeddings,
            self.positions,
            strict=True,
        ):
            windows = source.unfold(-1, patch, stride)
            outputs.append(embedding(windows) + position)
        return outputs


class TemporalPatchMixer(nn.Module):
    """Linear-complexity temporal mixing for one patch scale."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.token_dropout = nn.Dropout(dropout)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.token_norm(tokens).transpose(1, 2)
        mixed = self.pointwise(F.gelu(self.depthwise(mixed))).transpose(1, 2)
        tokens = tokens + self.token_dropout(mixed)
        return tokens + self.channel_mlp(self.channel_norm(tokens))


class MultiScaleCorPatchEncoder(nn.Module):
    """Mix time and variables per scale, then route scale summaries."""

    def __init__(
        self,
        d_model: int,
        scales: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by CorPatch attention heads")
        self.temporal_mixers = nn.ModuleList(
            TemporalPatchMixer(d_model, dropout) for _ in range(scales)
        )
        self.variable_attentions = nn.ModuleList(
            nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            for _ in range(scales)
        )
        self.state_router = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, scales)
        )
        self.variable_scale_router = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, patches: list[torch.Tensor], state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summaries = []
        for tokens, temporal_mixer, variable_attention in zip(
            patches,
            self.temporal_mixers,
            self.variable_attentions,
            strict=True,
        ):
            batch, variables, patch_count, width = tokens.shape
            temporal = temporal_mixer(
                tokens.reshape(batch * variables, patch_count, width)
            ).reshape(batch, variables, patch_count, width)
            variable_tokens = temporal.mean(dim=2)
            related, _ = variable_attention(
                variable_tokens,
                variable_tokens,
                variable_tokens,
                need_weights=False,
            )
            summaries.append(variable_tokens + related)

        scale_summaries = torch.stack(summaries, dim=2)
        logits = self.state_router(state).unsqueeze(1)
        logits = logits + self.variable_scale_router(scale_summaries).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        fused = (scale_summaries * weights.unsqueeze(-1)).sum(dim=2)
        return self.output_norm(fused), weights


class Model(nn.Module):
    """RevIN → time/frequency fusion → CMA → multi-scale CorPatch → head."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.features = getattr(configs, "features", "MS")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.channels = self.enc_in
        self.d_model = int(configs.d_model)
        self.cycle_len = int(getattr(configs, "cycle", 96))
        self.revin_mode = str(getattr(configs, "revin_mode", "full"))
        if self.revin_mode not in {"full", "center", "global"}:
            raise ValueError(f"Unsupported RevIN mode: {self.revin_mode}")

        dropout = float(getattr(configs, "dropout", 0.1))
        self.source_forecast_interface = True
        self.revin = RevIN(self.enc_in)
        self.physical_state = PhysicalStateEncoder(
            self.enc_in,
            self.d_model,
            getattr(configs, "physics_indices", None),
        )
        self.gtr = SourceGTR(
            self.seq_len,
            self.enc_in,
            self.cycle_len,
            period_len=int(getattr(configs, "gtr_period", 24)),
            dropout=dropout,
        )
        self.ldrive = LDrive(self.enc_in, dropout)
        self.temporal_gate = TemporalBranchGate(self.enc_in)
        self.spectral = SpectralMKAN(self.enc_in, self.d_model, self.seq_len)
        self.tf_router = TimeFrequencyRouter(self.d_model)

        self.prompt_dim = int(getattr(configs, "semantic_prompt_dim", 768))
        self.semantic_adapter = AdaptiveMultiHeadCMAResidual(
            seq_len=self.seq_len,
            d_model=self.d_model,
            prompt_dim=self.prompt_dim,
            num_heads=int(getattr(configs, "cma_heads", 4)),
            dropout=float(getattr(configs, "cma_dropout", dropout)),
        )
        self.semantic_enabled = True
        self.semantic_strength = 1.0

        sample_hours = float(getattr(configs, "sample_hours", 0.25))
        patch_hours = tuple(getattr(configs, "patch_hours", (1.0, 2.0, 4.0, 8.0)))
        patch_lengths = [
            max(1, min(self.seq_len, round(hours / sample_hours)))
            for hours in patch_hours
        ]
        self.patch = MultiScaleVariablePatch(self.seq_len, self.d_model, patch_lengths)
        self.corpatch = MultiScaleCorPatchEncoder(
            self.d_model,
            len(self.patch.patch_lengths),
            heads=int(getattr(configs, "corpatch_heads", 4)),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

        self.last_tf_weights: torch.Tensor | None = None
        self.last_temporal_weights: torch.Tensor | None = None
        self.last_scale_weights: torch.Tensor | None = None
        self.last_semantic_gate: torch.Tensor | None = None
        self.last_semantic_head_weights: torch.Tensor | None = None
        self.last_semantic_strength: torch.Tensor | None = None
        self.route_tf_weights: torch.Tensor | None = None
        self.route_scale_weights: torch.Tensor | None = None

    def _cycle_from_mark(
        self, mark: torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        if mark is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        cycle = mark if torch.is_tensor(mark) else torch.as_tensor(mark)
        if cycle.ndim == 1:
            selected = cycle
        elif cycle.ndim == 2:
            selected = cycle[:, 0]
        else:
            selected = cycle[:, 0, -1]
        return selected.to(device=device).long().reshape(batch) % self.cycle_len

    def _prompt_from_mark(self, mark: torch.Tensor | None) -> torch.Tensor | None:
        if not torch.is_tensor(mark) or mark.ndim != 3:
            return None
        if mark.shape[1:] == (self.enc_in, self.prompt_dim):
            return mark
        if mark.shape[1:] == (self.prompt_dim, self.enc_in):
            return mark.transpose(1, 2)
        return None

    def _normalize_history(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.revin_mode == "full":
            return self.revin.normalize(x)
        if self.revin_mode == "center":
            mean = x.mean(dim=1, keepdim=True).detach()
            return x - mean, mean, None
        return x, None, None

    def _forecast_core(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        x, mean, std = self._normalize_history(x_enc)
        state = self.physical_state(x_enc)
        cycle = self._cycle_from_mark(x_mark_enc, x_enc.size(0), x_enc.device)

        gtr_feature = self.gtr(x, cycle)
        ldrive_feature = self.ldrive(x)
        temporal_feature, temporal_weights = self.temporal_gate(
            gtr_feature, ldrive_feature
        )
        temporal_residual = temporal_feature - x
        spectral_residual = self.spectral(x, state)
        tf_weights = self.tf_router(state)
        fused = x + tf_weights[:, 0, None, None] * temporal_residual
        fused = fused + tf_weights[:, 1, None, None] * spectral_residual

        prompt = self._prompt_from_mark(x_mark_dec)
        semantic_gate = None
        semantic_strength = None
        semantic_weights = None
        if (
            prompt is not None
            and self.semantic_enabled
            and float(self.semantic_strength) != 0.0
        ):
            residual, semantic_gate, semantic_strength, semantic_weights = (
                self.semantic_adapter(fused, prompt, state)
            )
            fused = fused + float(self.semantic_strength) * residual

        variables, scale_weights = self.corpatch(self.patch(fused), state)
        target_token = variables[:, -1]
        global_token = variables.mean(dim=1)
        prediction = self.head(torch.cat([target_token, global_token], dim=-1))

        self.last_tf_weights = tf_weights.detach()
        self.last_temporal_weights = temporal_weights.detach()
        self.last_scale_weights = scale_weights.detach()
        self.last_semantic_gate = (
            semantic_gate.detach() if semantic_gate is not None else None
        )
        self.last_semantic_head_weights = (
            semantic_weights.detach() if semantic_weights is not None else None
        )
        self.last_semantic_strength = (
            semantic_strength.detach() if semantic_strength is not None else None
        )
        self.route_tf_weights = tf_weights
        self.route_scale_weights = scale_weights.mean(dim=1)

        if self.revin_mode == "full":
            return self.revin.denormalize_target(prediction, mean, std)
        if self.revin_mode == "center":
            return prediction + mean[..., -1]
        return prediction

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        del x_dec
        return self._forecast_core(x_enc, x_mark_enc, x_mark_dec).unsqueeze(-1)

    def forecast_multi(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask=None,
    ) -> torch.Tensor | None:
        del mask
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            return None
        output = (
            self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.features == "M"
            else self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        )
        return output[:, -self.pred_len :, :]
