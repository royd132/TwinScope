"""PV forecasting model: time-frequency backbone + semantic residual guidance."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from layers.corpatch import ForecastHead, MultiScaleCorPatchEncoder, MultiScaleVariablePatch
from layers.pc_fra import PcFraResidualAdapter, pack_slices, pack_width
from layers.physical_semantic import PhysicalSemanticEncoder
from layers.residual_corrector import SemanticResidualCalibration
from layers.revin import RevIN


class SourceGTR(nn.Module):
    """Cycle retrieval followed by local/global temporal fusion."""

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
        period_len = max(2, min(int(period_len), seq_len))
        self.cycle_query = nn.Parameter(torch.zeros(cycle_len, channels))
        self.mapping = nn.Linear(seq_len, seq_len)
        kernel = 1 + 2 * (period_len // 2)
        self.local_global_fusion = nn.Conv2d(
            1, 1, kernel_size=(2, kernel), padding=(0, period_len // 2), bias=False
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
    """Derivative-context dynamics with a gated GRU residual."""

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
            nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, gtr: torch.Tensor, ldrive: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([gtr.mean(1), ldrive.mean(1)], dim=-1)
        weights = torch.softmax(self.router(summary), dim=-1)
        return weights[:, 0, None, None] * gtr + weights[:, 1, None, None] * ldrive


class ChebyKANLinear(nn.Module):
    """Chebyshev-polynomial KAN mapping used by the spectral branch."""

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
    """rFFT -> complex MKAN -> irFFT frequency residual."""

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
        return torch.fft.irfft(torch.complex(real, imag), n=self.seq_len, dim=1)


class PhysicalStateEncoder(nn.Module):
    """Encode the historical physical state used by routing modules."""

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
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x.index_select(-1, self.indices))
        return self.summary(encoded[:, -1])


class TimeFrequencyRouter(nn.Module):
    """Route temporal and spectral residuals from the physical state."""

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


class Model(nn.Module):
    """TSLib-compatible forecasting model."""

    source_forecast_interface = True
    # Marks the optional keyword channel used to feed precomputed frozen
    # foundation-model window embeddings. The engine only passes
    # ``fm_context`` when an embedding cache is attached; the formal forward
    # never sees it.
    accepts_fm_context = True

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.features = getattr(configs, "features", "MS")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.cycle_len = int(getattr(configs, "cycle", 96))

        dropout = float(getattr(configs, "dropout", 0.1))
        corpatch_heads = int(getattr(configs, "corpatch_heads", 4))
        semantic_heads = int(getattr(configs, "semantic_heads", corpatch_heads))
        physical_cfg = getattr(configs, "physical_semantic", None)
        if physical_cfg is None:
            raise ValueError("PSRC requires physical_semantic training statistics")
        prompt_dim = int(physical_cfg.get("d_token", 32))
        # Pre-registered component-ablation switches. Empty/absent by
        # construction, so the formal (non-ablation) forward path is
        # numerically unchanged; every submodule is still built so checkpoint
        # parameter names stay identical across variants.
        self.ablation = frozenset(getattr(configs, "ablation", ()) or ())

        self.revin = RevIN(self.enc_in)
        self.state_encoder = PhysicalStateEncoder(
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
            heads=corpatch_heads,
            dropout=dropout,
        )
        self.forecast_head = ForecastHead(self.d_model, self.pred_len, dropout)

        self.physical_semantic_encoder = PhysicalSemanticEncoder(
            roles=physical_cfg["roles"],
            feature_mu=physical_cfg["feature_mu"],
            feature_sd=physical_cfg["feature_sd"],
            target_mu=physical_cfg["target_mu"],
            target_sd=physical_cfg["target_sd"],
            token_mean=physical_cfg["token_mean"],
            token_scale=physical_cfg["token_scale"],
            d_token=prompt_dim,
            dropout=dropout,
        )
        self.residual_calibration = SemanticResidualCalibration(
            self.d_model,
            prompt_dim,
            self.pred_len,
            heads=semantic_heads,
            max_correction=float(
                getattr(configs, "semantic_residual_max_correction", 0.5)
            ),
            max_gate=float(getattr(configs, "semantic_residual_max_gate", 1.0)),
            use_gate=bool(physical_cfg.get("use_gate", True)),
            dropout=dropout,
        )

        pc_fra_cfg = getattr(configs, "pc_fra", None)
        # PC-FRA H16: window-level H-dim residual adapter placed strictly
        # after frozen PSRC. Arm B is the PARA-style trunk; arm C adds
        # FiLM conditioning from the six physical tokens. Two shuffle
        # variants select permuted content for negative diagnostics.
        if pc_fra_cfg is not None:
            variant = str(pc_fra_cfg.get("variant", "pcfra"))
            if variant not in {"para", "pcfra", "phys_shuffle", "prior_shuffle"}:
                raise ValueError(f"unknown pc_fra variant {variant!r}")
            self.pc_fra_variant = variant
            self.pc_fra_adapter = PcFraResidualAdapter(
                horizon=int(pc_fra_cfg["horizon"]),
                hidden=int(pc_fra_cfg.get("hidden", 96)),
                dropout=float(pc_fra_cfg.get("dropout", 0.1)),
                use_film=variant != "para",
                token_dim=prompt_dim,
                film_bound=float(pc_fra_cfg.get("film_bound", 0.1)),
                epsilon=float(pc_fra_cfg["epsilon"]),
            )
            self.register_buffer(
                "pc_fra_level_mu",
                torch.tensor(float(pc_fra_cfg["level_mu"]), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "pc_fra_level_sd",
                torch.tensor(float(pc_fra_cfg["level_sd"]), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "pc_fra_sigma_mu",
                torch.tensor(float(pc_fra_cfg["sigma_mu"]), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "pc_fra_sigma_sd",
                torch.tensor(float(pc_fra_cfg["sigma_sd"]), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.pc_fra_adapter = None
            self.pc_fra_variant = None
        # (y_psrc standardized, Delta standardized) for the PC-FRA arm.
        self.last_pc_fra = None

    def _cycle_index(
        self, x_mark_enc: torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        if x_mark_enc is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        mark = x_mark_enc if torch.is_tensor(x_mark_enc) else torch.as_tensor(x_mark_enc)
        if mark.ndim == 1:
            cycle = mark
        elif mark.ndim == 2:
            cycle = mark[:, 0]
        else:
            cycle = mark[:, 0, -1]
        return cycle.to(device=device).long().reshape(batch) % self.cycle_len

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        fm_context: torch.Tensor | None = None,
        pc_partner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_dec, x_mark_dec
        ab = self.ablation
        x, mean, std = self.revin.normalize(x_enc)
        state = self.state_encoder(x_enc)
        cycle = self._cycle_index(x_mark_enc, x.size(0), x.device)
        # Compute the physics-aware semantic tokens at the same position as
        # the formal model (before the backbone branches) so dropout RNG
        # order - and therefore the full/structural-arm results - reproduces
        # the locked run exactly. The two purely numerical calibration arms
        # skip this encoder by construction.
        needs_semantic = not (
            "numerical_backbone" in ab or "numeric_residual" in ab
        )
        semantic_state = (
            self.physical_semantic_encoder(x_enc) if needs_semantic else None
        )

        # --- Group A: numerical time-frequency backbone -------------------
        gtr = self.gtr(x, cycle)
        ldrive = self.ldrive(x)
        if "no_ldrive" in ab:
            temporal = gtr
        elif "no_gtr" in ab:
            temporal = ldrive
        elif "equal_gl_fusion" in ab:
            temporal = 0.5 * gtr + 0.5 * ldrive
        else:
            temporal = self.temporal_gate(gtr, ldrive)

        if "no_spectral" in ab:
            # Temporal-only: bypass both the spectral branch and the TF router
            # (no zero vector is routed, avoiding an incidental rescaling).
            fused = temporal
        else:
            temporal_residual = temporal - x
            spectral_residual = self.spectral(x, state)
            if "equal_tf_fusion" in ab:
                fused = x + 0.5 * temporal_residual + 0.5 * spectral_residual
            else:
                tf_weights = self.tf_router(state)
                fused = (
                    x
                    + tf_weights[:, 0, None, None] * temporal_residual
                    + tf_weights[:, 1, None, None] * spectral_residual
                )

        scale_features = self.corpatch.encode_scales(self.patch(fused))
        variables, _ = self.corpatch.fuse_scales(scale_features, state)

        base_prediction = self.forecast_head(variables)

        # --- Group B: physics-aware semantic residual calibration ---------
        if "numerical_backbone" in ab:
            # B1: numerical backbone only; output the base forecast unchanged.
            prediction = base_prediction
            gate = torch.zeros_like(base_prediction)
        else:
            if "numeric_residual" in ab:
                # B2: keep the residual head/gate/supervision but remove all
                # physics-semantic conditioning (self-attend numeric queries).
                prediction, _, gate = self.residual_calibration(
                    variables,
                    state,
                    base_prediction,
                    None,
                    numeric_only=True,
                )
            else:
                prediction, _, gate = self.residual_calibration(
                    variables,
                    state,
                    base_prediction,
                    semantic_state,
                )
        self.last_psrc_gate = gate.detach()
        base_out = self.revin.denormalize_target(base_prediction, mean, std)
        psrc_out = self.revin.denormalize_target(prediction, mean, std)
        corrected_out = psrc_out
        self.last_pc_fra = None
        # --- PC-FRA residual adapter (registered H16 protocol) ----------
        # Strictly after frozen PSRC; y = y_psrc + epsilon*tanh(MLP(u)).
        # The pack carries physical-unit C_bar/L/sigma, the standardized
        # partner disagreement (negative diagnostic), and the registered
        # operating-state / intra-day one-hots. The adapter never feeds
        # GTR/LDrive/SpectralMKAN/patching/CorPatch/PSRC attention.
        if self.pc_fra_adapter is not None and "numerical_backbone" not in ab:
            if fm_context is None:
                raise ValueError(
                    "pc_fra is enabled but no precomputed pc_fra feature "
                    "pack was passed to the forward pass"
                )
            pack = fm_context.to(
                device=base_prediction.device, dtype=base_prediction.dtype
            )
            if pack.shape[-1] != pack_width(self.pred_len):
                raise ValueError(
                    "pc_fra pack width "
                    f"{pack.shape[-1]} != {pack_width(self.pred_len)}"
                )
            sl = pack_slices(self.pred_len)
            target_mu = float(self.physical_semantic_encoder.target_mu)
            target_sd = float(self.physical_semantic_encoder.target_sd)
            variant = self.pc_fra_variant
            if variant == "prior_shuffle":
                cbar_phys = pack[:, sl["c_bar_partner"]]
                disagreement = pack[:, sl["d_partner"]]
            else:
                cbar_phys = pack[:, sl["c_bar"]]
            cbar_std = (cbar_phys - target_mu) / target_sd
            if variant != "prior_shuffle":
                disagreement = cbar_std - psrc_out
            latest_std = (
                pack[:, sl["latest"] : sl["latest"] + 1] - target_mu
            ) / target_sd
            sigma_z = (
                pack[:, sl["sigma"] : sl["sigma"] + 1] - self.pc_fra_sigma_mu
            ) / self.pc_fra_sigma_sd.clamp_min(1e-8)
            level_z = (
                pack[:, sl["level"] : sl["level"] + 1] - self.pc_fra_level_mu
            ) / self.pc_fra_level_sd.clamp_min(1e-8)
            epv = torch.cat(
                [pack[:, sl["state"]], level_z, pack[:, sl["bucket"]]], dim=-1
            )
            pc_tokens = semantic_state
            if variant == "phys_shuffle":
                if pc_partner is None:
                    raise ValueError(
                        "phys_shuffle variant requires partner history"
                    )
                # Frozen encoder over the partner window: breaks physical
                # alignment while preserving the token marginals.
                pc_tokens = self.physical_semantic_encoder(
                    pc_partner.to(device=x_enc.device, dtype=x_enc.dtype)
                )
            pc_delta = self.pc_fra_adapter(
                psrc_out,
                cbar_std,
                disagreement,
                latest_std,
                epv,
                sigma_z,
                pc_tokens if variant != "para" else None,
            )
            corrected_out = psrc_out + pc_delta
            self.last_pc_fra = (psrc_out.detach(), pc_delta)
        self.last_semantic_decomposition = (base_out, psrc_out - base_out)

        return corrected_out.unsqueeze(-1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask=None,
        fm_context: torch.Tensor | None = None,
        pc_partner: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del mask
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            return None
        return self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            fm_context=fm_context,
            pc_partner=pc_partner,
        )[:, -self.pred_len :]
