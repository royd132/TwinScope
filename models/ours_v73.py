"""Time-frequency PV forecaster with semantic residuals and CorPatch encoding.

The source five-argument interface is preserved. Time and frequency branches
produce history features, which are fused before the frozen-LM residual and
multi-scale patching. Therefore there is exactly one forecast head after
Variable-aware CorPatch.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.t3time_fusion import AdaptiveMultiHeadCMAResidual


class RevIN(nn.Module):
    """Reversible instance normalisation over the history axis."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, channels))

    def normalize(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalize_target(
        self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        weight = self.weight[..., -1:].clamp_min(self.eps)
        restored = (y.unsqueeze(-1) - self.bias[..., -1:]) / weight
        return (restored * std[..., -1:] + mean[..., -1:]).squeeze(-1)


class SourceGTR(nn.Module):
    """Cycle retrieval plus local/global 2-D fusion from ``models/GTR.py``.

    The wrapper uses the real cycle index supplied by the dataset rather than
    generating a random index, while retaining the source GTR computation.
    """

    def __init__(
        self,
        seq_len: int,
        channels: int,
        cycle_len: int,
        period_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.cycle_len = cycle_len
        self.period_len = max(2, min(int(period_len), seq_len))
        self.cycle_query = nn.Parameter(torch.zeros(cycle_len, channels))
        self.mapping = nn.Linear(seq_len, seq_len)
        kernel = 1 + 2 * (self.period_len // 2)
        self.local_global_fusion = nn.Conv2d(
            1, 1, kernel_size=(2, kernel), stride=1,
            padding=(0, self.period_len // 2), padding_mode="zeros", bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError(
                f"GTR expected [B,{self.seq_len},{self.channels}], got {tuple(x.shape)}"
            )
        offsets = torch.arange(length, device=x.device)
        positions = (cycle[:, None] + offsets[None, :]) % self.cycle_len
        query = self.cycle_query[positions].transpose(1, 2)
        local = x.transpose(1, 2)
        global_query = self.mapping(query)
        paired = torch.stack([local, global_query], dim=2)
        paired = paired.reshape(batch * channels, 1, 2, length)
        fused = self.local_global_fusion(paired).reshape(batch, channels, length)
        return self.dropout(fused).transpose(1, 2)


class LDrive(nn.Module):
    """Derivative-context block preserving time and variable axes."""

    def __init__(self, channels: int, dropout: float = 0.1):
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
    """Sample-wise fusion of parallel GTR and L-Drive features."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, 2 * channels)
        self.router = nn.Sequential(
            nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, gtr: torch.Tensor, ldrive: torch.Tensor):
        summary = torch.cat([gtr.mean(1), ldrive.mean(1)], dim=-1)
        weights = torch.softmax(self.router(summary), dim=-1)
        fused = weights[:, 0, None, None] * gtr + weights[:, 1, None, None] * ldrive
        return fused, weights


class ChebyKANLinear(nn.Module):
    """Chebyshev-polynomial KAN mapping used in the complex spectrum."""

    def __init__(self, input_dim: int, output_dim: int, degree: int = 3):
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
    """rFFT -> complex MKAN -> irFFT frequency-response residual."""

    def __init__(self, channels: int, d_model: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.force_band_index = None
        self.band_state_intervention = None
        self.real_mkan = ChebyKANLinear(channels, d_model, degree=3)
        self.imag_mkan = ChebyKANLinear(channels, d_model, degree=3)
        self.back = nn.Linear(d_model, channels)
        self.frequency_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        nn.init.zeros_(self.back.weight)
        nn.init.zeros_(self.back.bias)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        if self.force_band_index is not None:
            bins = spectrum.size(1)
            edges = torch.linspace(0, bins, 4, device=x.device).round().long()
            band = int(self.force_band_index)
            mask = torch.zeros(bins, device=x.device, dtype=x.dtype)
            mask[int(edges[band]) : int(edges[band + 1])] = 1.0
            spectrum = spectrum * mask.view(1, -1, 1)
        real = self.real_mkan(spectrum.real)
        imag = self.imag_mkan(spectrum.imag)
        routing_state = state
        if self.band_state_intervention == "zero":
            routing_state = torch.zeros_like(state)
        elif self.band_state_intervention == "shuffle":
            routing_state = state.roll(1, dims=0)
        gate = self.frequency_gate(routing_state).unsqueeze(1)
        corrected = torch.complex(self.back(real * gate), self.back(imag * gate))
        return torch.fft.irfft(corrected, n=self.seq_len, dim=1)


class PhysicalStateRouter(nn.Module):
    """Encode operating state from historical observations only."""

    def __init__(
        self, channels: int, d_model: int,
        physics_indices: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        indices = list(physics_indices or range(channels))
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.encoder = nn.GRU(len(indices), d_model, batch_first=True)
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = x.index_select(-1, self.indices)
        encoded, _ = self.encoder(selected)
        return self.summary(encoded[:, -1])


class TimeFrequencyFeatureRouter(nn.Module):
    """Physical-state-conditioned routing before patch extraction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.prior_logits = nn.Parameter(torch.tensor([-0.5, -1.0]))

    def forward(self, state: torch.Tensor):
        return torch.sigmoid(self.router(state) + self.prior_logits)


class StaticSemanticResidual(nn.Module):
    """Zero-safe variable-role residual without multi-head CMA."""

    def __init__(self, seq_len: int, d_model: int, channel_embeddings=None):
        super().__init__()
        self.last_head_weights = None
        self.last_gate = None
        if channel_embeddings is None:
            self.register_buffer("channel_embeddings", torch.empty(0), persistent=False)
            self.enabled = False
            return
        vectors = torch.as_tensor(channel_embeddings, dtype=torch.float32)
        self.register_buffer("channel_embeddings", vectors)
        self.enabled = True
        self.numeric_projection = nn.Linear(seq_len, d_model)
        self.semantic_projection = nn.Linear(vectors.shape[-1], d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid()
        )
        self.back = nn.Linear(d_model, seq_len)
        nn.init.zeros_(self.back.weight)
        nn.init.zeros_(self.back.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(x)
        numeric = self.numeric_projection(x.transpose(1, 2))
        semantic = self.semantic_projection(self.channel_embeddings).unsqueeze(0)
        semantic = semantic.expand(x.size(0), -1, -1)
        gate = self.gate(torch.cat([numeric, semantic], dim=-1))
        self.last_gate = gate.detach()
        return (gate * self.back(torch.tanh(numeric + semantic))).transpose(1, 2)


class PromptSemanticResidual(nn.Module):
    """Compatibility wrapper for adaptive multi-head cross-modal alignment."""

    def __init__(
        self, seq_len: int, d_model: int, prompt_dim: int,
        cma_heads: int = 4, dropout: float = 0.1,
        prompt_encoder_kind: str = "transformer",
    ):
        super().__init__()
        self.cma = AdaptiveMultiHeadCMAResidual(
            seq_len=seq_len,
            d_model=d_model,
            prompt_dim=prompt_dim,
            num_heads=cma_heads,
            dropout=dropout,
            prompt_encoder_kind=prompt_encoder_kind,
        )
        self.last_head_weights = None

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, state: torch.Tensor):
        residual, gate, strength, head_weights = self.cma(x, prompt, state)
        self.last_head_weights = head_weights
        return residual, gate, strength


def _advance_legacy_semantic_rng(seq_len: int, d_model: int, prompt_dim: int) -> None:
    """Preserve the v26 numerical-backbone initialization stream.

    The semantic block is constructed before CorPatch and the forecast head.  A
    replacement with a different parameter layout would otherwise change every
    downstream random initialization even when ``semantic_strength == 0``.  The
    temporary layers mirror the retired v26 adapter's construction order; they
    are never registered or used for prediction.
    """
    legacy_layers = (
        nn.Linear(seq_len, d_model),
        nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, d_model)),
        nn.Linear(d_model, d_model),
        nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        ),
        nn.Linear(d_model, seq_len),
    )
    del legacy_layers


class MultiScaleVariablePatch(nn.Module):
    """Patch time independently for every variable and physical scale."""

    def __init__(self, seq_len: int, d_model: int, patch_lengths: Iterable[int]):
        super().__init__()
        self.seq_len = seq_len
        self.patch_lengths = list(
            dict.fromkeys(max(1, min(seq_len, int(p))) for p in patch_lengths)
        )
        self.strides = [max(1, patch // 2) for patch in self.patch_lengths]
        self.embeddings = nn.ModuleList([nn.Linear(patch, d_model) for patch in self.patch_lengths])
        self.positions = nn.ParameterList()
        for patch, stride in zip(self.patch_lengths, self.strides):
            count = 1 + (seq_len - patch) // stride
            self.positions.append(nn.Parameter(torch.zeros(1, 1, count, d_model)))
        self.scales = list(zip(self.patch_lengths, self.strides))

    def forward(self, x: torch.Tensor):
        source = x.transpose(1, 2)
        outputs = []
        for patch, stride, embedding, position in zip(
            self.patch_lengths, self.strides, self.embeddings, self.positions
        ):
            windows = source.unfold(dimension=-1, size=patch, step=stride)
            outputs.append(embedding(windows) + position)
        return outputs


class TemporalPatchMixer(nn.Module):
    """Linear-complexity temporal mixing for one scale of variable patch tokens."""

    def __init__(self, d_model: int, dropout: float = 0.1):
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


class VariableAwareCorPatchEncoder(nn.Module):
    """Efficient temporal, variable and cross-scale correlation encoder."""

    def __init__(
        self, d_model: int, scales: int, heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by CorPatch attention heads")
        self.scales = scales

        self.temporal_encoders = nn.ModuleList(
            [TemporalPatchMixer(d_model, dropout) for _ in range(scales)]
        )
        self.variable_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            for _ in range(scales)
        ])
        self.cross_scale_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.state_router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, scales))
        self.variable_scale_router = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, patches, state: torch.Tensor, scale_context=None):
        scale_summaries = []
        for tokens, temporal_encoder, variable_attention in zip(
            patches, self.temporal_encoders, self.variable_attentions
        ):
            batch, variables, patch_count, width = tokens.shape
            temporal = temporal_encoder(tokens.reshape(batch * variables, patch_count, width))
            temporal = temporal.reshape(batch, variables, patch_count, width)
            variable_tokens = temporal.mean(dim=2)
            related, _ = variable_attention(
                variable_tokens, variable_tokens, variable_tokens, need_weights=False
            )
            scale_summaries.append(variable_tokens + related)
        summaries = torch.stack(scale_summaries, dim=2)
        batch, variables, scales, width = summaries.shape
        flat = summaries.reshape(batch * variables, scales, width)
        correlated, _ = self.cross_scale_attention(flat, flat, flat, need_weights=False)
        correlated = correlated.reshape(batch, variables, scales, width)
        logits = self.state_router(state).unsqueeze(1)
        logits = logits + self.variable_scale_router(correlated).squeeze(-1)
        if scale_context is not None:
            logits = logits + scale_context.unsqueeze(1)
        weights = torch.softmax(logits, dim=-1)
        fused = (correlated * weights.unsqueeze(-1)).sum(dim=2)
        return self.output_norm(fused), weights


class SolarResidualAnchor(nn.Module):
    """Optional validation-calibrated deterministic solar reference."""

    def __init__(self, config, d_model: int):
        super().__init__()
        self.config = config
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid()
        )

    def forward(self, x, future_x, state, reliability_bias=None):
        if self.config is None or future_x is None:
            return None, None
        config = self.config
        clear_history = (
            x[..., config["clear_hist_idx"]] * config["clear_sd"] + config["clear_mu"]
        ).clamp_min(0)
        clear_future = (
            future_x[..., config["clear_future_pos"]] * config["clear_sd"] + config["clear_mu"]
        ).clamp_min(0)
        power = x[..., -1] * config["target_sd"] + config["target_mu"]
        efficiency = (power[:, -1] / config["capacity"]) / (
            clear_history[:, -1] / config["clear_max"]
        ).clamp_min(0.02)
        reference = (
            config["capacity"] * efficiency.clamp(0, 1.5)[:, None]
            * clear_future / config["clear_max"]
        )
        reference = (reference - config["target_mu"]) / max(config["target_sd"], 1e-6)
        reliability = self.gate(state)
        if reliability_bias is not None:
            eps = reliability.new_tensor(1e-5)
            logits = torch.logit(reliability.clamp(eps, 1.0 - eps))
            reliability = torch.sigmoid(logits + reliability_bias)
        return reference, reliability


class NWPConditionedForecastDecoder(nn.Module):
    """Source-compatible decoder retained for explicit NWP ablations."""

    def __init__(self, future_channels: int, horizon: int, d_model: int):
        super().__init__()
        self.future_projection = nn.Linear(future_channels, d_model)
        self.horizon_query = nn.Parameter(torch.zeros(1, horizon, d_model))
        self.cross_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.output = nn.Linear(d_model, 1)

    def forward(self, memory, future_x, state=None):
        query = self.horizon_query.expand(future_x.size(0), -1, -1)
        query = query + self.future_projection(future_x)
        hidden, _ = self.cross_attention(query, memory, memory, need_weights=False)
        return self.output(hidden).squeeze(-1)


class Model(nn.Module):
    """Time-frequency fusion followed by semantic residual and CorPatch."""

    variant = "pcarr_v29_tf_adaptive_multihead_cma_corpatch"

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.features = getattr(configs, "features", "MS")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.cycle_len = int(getattr(configs, "cycle", 96))
        self.revin_mode = str(getattr(configs, "revin_mode", "full"))
        if self.revin_mode not in {"full", "center", "global"}:
            raise ValueError(f"Unsupported RevIN mode: {self.revin_mode}")
        self.source_forecast_interface = True
        dropout = float(getattr(configs, "dropout", 0.1))

        self.revin = RevIN(self.enc_in)
        self.physical_router = PhysicalStateRouter(
            self.enc_in, self.d_model, getattr(configs, "physics_indices", None)
        )
        self.gtr = SourceGTR(
            self.seq_len, self.enc_in, self.cycle_len,
            period_len=int(getattr(configs, "gtr_period", 24)), dropout=dropout,
        )
        self.ldrive = LDrive(self.enc_in, dropout=dropout)
        self.temporal_gate = TemporalBranchGate(self.enc_in)
        self.spectral = SpectralMKAN(self.enc_in, self.d_model, self.seq_len)
        self.tf_router = TimeFrequencyFeatureRouter(self.d_model)

        self.prompt_dim = int(getattr(configs, "semantic_prompt_dim", 768))
        self.cma_heads = int(getattr(configs, "cma_heads", 4))
        self.cma_dropout = float(getattr(configs, "cma_dropout", dropout))
        self.cma_scale_route = bool(getattr(configs, "cma_scale_route", False))
        self.cma_prompt_encoder = str(
            getattr(configs, "cma_prompt_encoder", "transformer")
        )
        # Retain the validated v26 numerical initialization, then isolate the
        # new adapter so CMA hyperparameters cannot perturb that backbone.
        _advance_legacy_semantic_rng(self.seq_len, self.d_model, 768)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2027)
            self.semantic_adapter = PromptSemanticResidual(
                self.seq_len, self.d_model, self.prompt_dim,
                cma_heads=self.cma_heads, dropout=self.cma_dropout,
                prompt_encoder_kind=self.cma_prompt_encoder,
            )
        self.semantic_enabled = True
        self.semantic_strength = 1.0
        self.fsra_adapter = StaticSemanticResidual(
            self.seq_len, self.d_model,
            getattr(configs, "channel_semantic_embeddings", None),
        )
        self.fsra_has_content = self.fsra_adapter.enabled
        self.fsra_enabled = self.fsra_has_content
        self.fsra_strength = 1.0
        self.fsra_cma_heads = self.cma_heads
        self.fsra_channel_embeddings = self.fsra_adapter.channel_embeddings

        sample_hours = float(getattr(configs, "sample_hours", 0.25))
        patch_hours = tuple(getattr(configs, "patch_hours", (1.0, 2.0, 4.0, 8.0)))
        patch_lengths = [
            max(1, min(self.seq_len, round(hours / sample_hours)))
            for hours in patch_hours
        ]
        self.patch = MultiScaleVariablePatch(self.seq_len, self.d_model, patch_lengths)
        self.encoder = VariableAwareCorPatchEncoder(
            self.d_model, len(self.patch.patch_lengths),
            heads=int(getattr(configs, "corpatch_heads", 4)), dropout=dropout,
        )
        # Keep the forecast-head initialization independent of Prompt width.
        # The temporary Linear advances the legacy 768-d route RNG stream.
        legacy_scale_route = nn.Linear(768, len(self.patch.patch_lengths))
        del legacy_scale_route
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2028)
            self.scale_prompt_router = nn.Sequential(
                nn.LayerNorm(self.prompt_dim),
                nn.Linear(self.prompt_dim, len(self.patch.patch_lengths)),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model), nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_model, self.pred_len),
        )

        self.anchor = SolarResidualAnchor(
            getattr(configs, "solar_anchor_config", None), self.d_model
        )
        self.reference_reliability_bias = nn.Parameter(torch.tensor(0.0))
        self.reference_residual_active = self.anchor.config is not None
        self.reference_strength = 1.0

        self.last_tf_weights = None
        self.last_temporal_weights = None
        self.last_scale_weights = None
        self.route_tf_weights = None
        self.route_scale_weights = None
        self.last_output_correction = None
        self.semantic_alignment_loss = None
        self.fsra_alignment_loss = None
        self.last_t3time_head_weights = None
        self.last_t3time_horizon_gate = None
        self.last_fsra_head_weights = None
        self.last_fsra_gate = None
        self.last_semantic_head_weights = None
        self.last_semantic_gate = None
        self.last_scale_prompt_head_weights = None
        self.last_scale_prompt_gate = None
        self.last_scale_prompt_match_accuracy = None
        self.last_semantic_strength = None
        self.force_tf_route = None
        self.routing_state_intervention = None
        self.tf_state_intervention = None
        self.scale_state_intervention = None
        self.output_state_intervention = None
        self.frequency_indices = None
        self.scale_prompt_adapter = None
        self.semantic_hierarchical_router = None

    @property
    def spectral_mkan(self):
        return self.spectral

    @property
    def prompt_scale_router(self):
        return self.scale_prompt_router

    def _cycle_from_mark(self, x_mark_enc, batch: int, device):
        if x_mark_enc is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if not torch.is_tensor(x_mark_enc):
            return torch.as_tensor(x_mark_enc, device=device).long().reshape(batch)
        if x_mark_enc.ndim == 1:
            cycle = x_mark_enc
        elif x_mark_enc.ndim == 2:
            cycle = x_mark_enc[:, 0]
        else:
            cycle = x_mark_enc[:, 0, -1]
        return cycle.to(device=device).long().remainder(self.cycle_len)

    def _prompt_from_mark(self, x_mark_dec):
        if not torch.is_tensor(x_mark_dec) or x_mark_dec.ndim != 3:
            return None
        variable_first = x_mark_dec.shape[1:] == (self.enc_in, self.prompt_dim)
        embedding_first = x_mark_dec.shape[1:] == (self.prompt_dim, self.enc_in)
        if variable_first:
            return x_mark_dec
        if embedding_first:
            return x_mark_dec.transpose(1, 2)
        return None

    def _future_from_decoder(self, x_dec):
        if not torch.is_tensor(x_dec) or x_dec.numel() == 0:
            return None
        if x_dec.ndim != 3:
            raise ValueError(f"x_dec must be [B,L,D], received {tuple(x_dec.shape)}")
        if x_dec.size(1) < self.pred_len:
            raise ValueError(
                f"x_dec length must be >= pred_len={self.pred_len}, received {x_dec.size(1)}"
            )
        return x_dec[:, -self.pred_len :, :]

    def _forecast_core(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        batch = x_enc.size(0)
        cycle = self._cycle_from_mark(x_mark_enc, batch, x_enc.device)
        future_x = self._future_from_decoder(x_dec)
        prompt = self._prompt_from_mark(x_mark_dec)
        self.last_output_correction = None
        if self.revin_mode == "full":
            x, mean, std = self.revin.normalize(x_enc)
        elif self.revin_mode == "center":
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            std = None
            x = x_enc - mean
        else:
            mean = None
            std = None
            x = x_enc
        physical_state = self.physical_router(x_enc)
        if self.routing_state_intervention == "zero":
            physical_state = torch.zeros_like(physical_state)
        elif self.routing_state_intervention == "shuffle":
            physical_state = physical_state.roll(1, dims=0)

        gtr_feature = self.gtr(x, cycle)
        ldrive_feature = self.ldrive(x)
        temporal_feature, temporal_weights = self.temporal_gate(gtr_feature, ldrive_feature)
        temporal_residual = temporal_feature - x
        spectral_residual = self.spectral(x, physical_state)

        tf_state = physical_state
        if self.tf_state_intervention == "zero":
            tf_state = torch.zeros_like(physical_state)
        elif self.tf_state_intervention == "shuffle":
            tf_state = physical_state.roll(1, dims=0)
        tf_weights = self.tf_router(tf_state)
        if self.force_tf_route == 0:
            tf_weights = torch.stack(
                [torch.ones_like(tf_weights[:, 0]), torch.zeros_like(tf_weights[:, 1])], dim=-1
            )
        elif self.force_tf_route == 1:
            tf_weights = torch.stack(
                [torch.zeros_like(tf_weights[:, 0]), torch.ones_like(tf_weights[:, 1])], dim=-1
            )
        elif self.force_tf_route == 2:
            tf_weights = torch.zeros_like(tf_weights)
        fused_history = (
            x + tf_weights[:, 0, None, None] * temporal_residual
            + tf_weights[:, 1, None, None] * spectral_residual
        )

        if self.fsra_enabled:
            fused_history = fused_history + float(self.fsra_strength) * self.fsra_adapter(fused_history)
        self.last_fsra_gate = self.fsra_adapter.last_gate

        semantic_gate = None
        semantic_strength = None
        prompt_active = (
            prompt is not None
            and self.semantic_enabled
            and float(self.semantic_strength) != 0.0
        )
        if prompt_active:
            semantic_residual, semantic_gate, semantic_strength = self.semantic_adapter(
                fused_history, prompt, physical_state
            )
            fused_history = fused_history + float(self.semantic_strength) * semantic_residual

        patch_tokens = self.patch(fused_history)
        scale_context = None
        if prompt_active and self.cma_scale_route:
            scale_context = float(self.semantic_strength) * self.scale_prompt_router(
                prompt.mean(dim=1).to(x.dtype)
            )
        scale_state = physical_state
        if self.scale_state_intervention == "zero":
            scale_state = torch.zeros_like(physical_state)
        elif self.scale_state_intervention == "shuffle":
            scale_state = physical_state.roll(1, dims=0)
        encoded_variables, variable_scale_weights = self.encoder(
            patch_tokens, scale_state, scale_context
        )
        target_token = encoded_variables[:, -1]
        global_token = encoded_variables.mean(dim=1)
        prediction = self.head(torch.cat([target_token, global_token], dim=-1))

        reference, reliability = self.anchor(
            x_enc, future_x, physical_state, self.reference_reliability_bias
        )
        if reference is not None and self.reference_residual_active:
            strength = prediction.new_tensor(float(self.reference_strength))
            prediction = prediction + strength * reliability * (reference - prediction)
            self.last_output_correction = reliability.detach()

        self.last_tf_weights = tf_weights.detach()
        self.last_temporal_weights = temporal_weights.detach()
        self.last_scale_weights = variable_scale_weights.detach()
        self.route_tf_weights = tf_weights
        self.route_scale_weights = variable_scale_weights.mean(dim=1)
        self.last_semantic_gate = semantic_gate.detach() if semantic_gate is not None else None
        self.last_semantic_head_weights = (
            self.semantic_adapter.last_head_weights.detach()
            if self.semantic_adapter.last_head_weights is not None and prompt_active
            else None
        )
        if semantic_strength is not None:
            self.last_semantic_strength = semantic_strength.detach()
            # The legacy evaluator expects a batch-wise [B,N,S,1] diagnostic.
            # These are the actual prompt-conditioned scale routing weights.
            self.last_scale_prompt_gate = variable_scale_weights.detach().unsqueeze(-1)
        else:
            self.last_semantic_strength = None
            self.last_scale_prompt_gate = None
        if self.revin_mode == "full":
            return self.revin.denormalize_target(prediction, mean, std)
        if self.revin_mode == "center":
            return prediction + mean[..., -1]
        return prediction

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self._forecast_core(x_enc, x_mark_enc, x_dec, x_mark_dec).unsqueeze(-1)

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            return None
        output = (
            self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.features == "M"
            else self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        )
        return output[:, -self.pred_len :, :]


RPGTR = Model
