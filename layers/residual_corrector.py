"""Semantic residual calibration head (PSRC components 2 + 3).

Given the numerical backbone's base forecast and the physical semantic
state tokens, this module predicts a bounded residual correction and a
per-horizon confidence gate:

    yhat = yhat0 + g * dyhat,   g in [0, 1]

The probe evidence behind this design: physical state tokens carry
residual-predictive signal (I(S_physical; e) > 0), the signal concentrates
in hard regimes (cloud transitions, ramps, high volatility), and a learned
gate that decides *when to trust* the correction beats applying it
unconditionally.  This module is deliberately independent of the TGSM
prompt machinery: PSRC consumes structured physical state, not language.
"""

from __future__ import annotations

import torch
from torch import nn


def _semantic_state_memory(
    semantic_state: torch.Tensor,
    *,
    batch: int,
    token_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if semantic_state.ndim != 3:
        raise ValueError("semantic state tokens must be [batch, tokens, token_dim]")
    if semantic_state.shape[0] != batch or semantic_state.shape[-1] != token_dim:
        raise ValueError(
            f"expected semantic state batch={batch}, width={token_dim}, "
            f"got {tuple(semantic_state.shape)}"
        )
    return semantic_state.to(device=device, dtype=dtype)


class SemanticResidualCalibration(nn.Module):
    """Confidence-gated residual correction of the base forecast.

    Horizon queries are conditioned on the backbone context and the base
    forecast itself, cross-attend to the physical state tokens, and emit a
    tanh-bounded residual dyhat plus a sigmoid confidence gate g.  The
    residual head is zero-initialized so the branch starts as an exact
    no-op (identity-safe): the PSRC model reproduces the numerical-only
    baseline at initialization and only departs from it as the calibration
    loss drives learning.
    """

    def __init__(
        self,
        d_model: int,
        token_dim: int,
        pred_len: int,
        heads: int = 4,
        max_correction: float = 0.5,
        max_gate: float = 1.0,
        use_gate: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by calibration attention heads")

        self.d_model = d_model
        self.token_dim = token_dim
        self.pred_len = pred_len
        self.max_correction = max_correction
        if not 0.0 <= max_gate <= 1.0:
            raise ValueError("max_gate must be between zero and one")
        self.max_gate = float(max_gate)
        self.use_gate = bool(use_gate)

        self.context_projection = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.prediction_projection = nn.Linear(1, d_model, bias=False)
        self.horizon_embedding = nn.Parameter(torch.empty(1, pred_len, d_model))
        nn.init.normal_(self.horizon_embedding, std=0.02)

        self.semantic_projection = nn.Linear(token_dim, d_model, bias=False)
        self.semantic_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1) if self.use_gate else None

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.confidence_head is not None:
            nn.init.zeros_(self.confidence_head.weight)
            nn.init.constant_(self.confidence_head.bias, -2.0)

    def forward(
        self,
        variables: torch.Tensor,
        state: torch.Tensor,
        base_prediction: torch.Tensor,
        semantic_state: torch.Tensor | None,
        *,
        numeric_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (corrected forecast, raw residual dyhat, confidence gate).

        ``corrected = base_prediction + gate * dyhat``; with the gate
        disabled the gate term is exactly one (unconditional correction).

        When ``numeric_only`` is set (ablation arm "numeric_residual"), the
        physics-aware semantic memory is not consumed: the calibration head
        keeps the exact same parameters and its residual/gate supervision,
        but cross-attention attends to the numerical horizon queries
        themselves instead of physical semantic tokens.

        """
        batch = variables.shape[0]
        if base_prediction.shape != (batch, self.pred_len):
            raise ValueError("base prediction must be [batch, pred_len]")

        target = variables[:, -1]
        context = variables.mean(dim=1)
        sample_context = self.context_projection(
            torch.cat([target, context, state], dim=-1)
        )
        queries = sample_context.unsqueeze(1) + self.horizon_embedding
        queries = queries + self.prediction_projection(base_prediction.unsqueeze(-1))
        query_norm = self.query_norm(queries)
        if numeric_only:
            # Self-attention over the numerical queries only; no physical
            # semantic token and no extra parameters are introduced.
            attended, _ = self.cross_attention(
                query_norm, query_norm, query_norm, need_weights=False
            )
        else:
            memory = _semantic_state_memory(
                semantic_state,
                batch=batch,
                token_dim=self.token_dim,
                device=variables.device,
                dtype=variables.dtype,
            )
            memory = self.semantic_norm(self.semantic_projection(memory))
            attended, _ = self.cross_attention(
                query_norm, memory, memory, need_weights=False
            )
        fused = self.fusion(torch.cat([queries, attended, queries * attended], dim=-1))

        residual = self.max_correction * torch.tanh(
            self.residual_head(fused).squeeze(-1)
        )
        if self.confidence_head is None:
            gate = torch.full_like(residual, self.max_gate)
        else:
            gate = self.max_gate * torch.sigmoid(
                self.confidence_head(fused).squeeze(-1)
            )
        return base_prediction + gate * residual, residual, gate
