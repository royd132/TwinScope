"""Efficient T3Time-style cross-modal alignment for the Ours model.

The frozen language model is an offline feature encoder. This module projects
its per-variable embeddings to the numerical width before applying independent
cross-modal attention heads, dynamic head aggregation and channel-wise residual
fusion. It returns a residual so the numerical time-frequency backbone remains
an explicit, auditable path.
"""
from __future__ import annotations

import torch
from torch import nn


class AdaptiveHeadAggregation(nn.Module):
    """Aggregate independent CMA heads for every sample and variable."""

    def __init__(self, num_heads: int, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, d_model)
        self.num_heads = int(num_heads)
        self.router = nn.Sequential(
            nn.LayerNorm(self.num_heads * d_model),
            nn.Linear(self.num_heads * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_heads),
        )

    def forward(self, head_outputs: list[torch.Tensor]):
        stacked = torch.stack(head_outputs, dim=2)  # [B, N, H, D]
        batch, variables, heads, width = stacked.shape
        logits = self.router(stacked.reshape(batch, variables, heads * width))
        weights = torch.softmax(logits, dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return fused, weights


class AdaptiveMultiHeadCMAResidual(nn.Module):
    """T3Time-style prompt alignment expressed as a bounded history residual.

    Numerical time-frequency tokens are the queries and frozen GPT-2 prompt
    tokens are keys/values. ``num_heads`` denotes independent CMA experts, not
    attention sub-heads. This matches T3Time while keeping memory proportional
    to ``d_model`` instead of running attention at width 768.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        prompt_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        prompt_encoder_kind: str = "transformer",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("CMA must contain at least one independent head")

        self.num_heads = int(num_heads)
        self.prompt_encoder_kind = str(prompt_encoder_kind)
        if self.prompt_encoder_kind not in {"transformer", "numeric_mlp"}:
            raise ValueError(
                "prompt_encoder_kind must be 'transformer' or 'numeric_mlp'"
            )
        self.numeric_projection = nn.Linear(seq_len, d_model)
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, d_model), nn.GELU(),
        )
        prompt_encoder_heads = next(
            heads for heads in (4, 2, 1) if d_model % heads == 0
        )
        if self.prompt_encoder_kind == "transformer":
            prompt_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=prompt_encoder_heads,
                dim_feedforward=2 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.prompt_encoder = nn.TransformerEncoder(
                prompt_layer, num_layers=1, enable_nested_tensor=False,
            )
        else:
            # Information-matched non-language control: each variable's full
            # numerical history is encoded independently before the unchanged
            # CMA path models cross-variable interactions.
            self.prompt_encoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
            )
        self.state_projection = nn.Linear(d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.prompt_norm = nn.LayerNorm(d_model)
        self.cma_heads = nn.ModuleList(
            nn.MultiheadAttention(
                d_model, num_heads=1, dropout=dropout, batch_first=True,
            )
            for _ in range(self.num_heads)
        )
        self.dynamic_heads = AdaptiveHeadAggregation(
            self.num_heads, d_model, dropout,
        )
        self.channel_mix_logit = nn.Parameter(torch.zeros(d_model))
        self.variable_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Linear(d_model, seq_len)
        self.residual_logit = nn.Parameter(torch.tensor(-1.3863))

        # Start as a small correction while letting gradients reach every CMA
        # component immediately during the prompt-only training stage.
        nn.init.normal_(self.output_projection.weight, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

        self.last_head_weights: torch.Tensor | None = None
        self.last_channel_mix: torch.Tensor | None = None

    def forward(
        self,
        history: torch.Tensor,
        prompt: torch.Tensor,
        state: torch.Tensor,
    ):
        expected = (history.size(0), history.size(2))
        if prompt.ndim != 3 or prompt.shape[:2] != expected:
            raise ValueError(
                "CMA prompt must be [batch, variables, prompt_dim], got "
                f"{tuple(prompt.shape)} for history {tuple(history.shape)}"
            )

        numeric = self.numeric_projection(history.transpose(1, 2))
        state_token = self.state_projection(state).unsqueeze(1)
        query = self.query_norm(numeric + state_token)
        prompt_token = self.prompt_projection(prompt.to(dtype=history.dtype))
        prompt_token = self.prompt_norm(self.prompt_encoder(prompt_token))

        aligned = []
        for head in self.cma_heads:
            value, _ = head(
                query, prompt_token, prompt_token, need_weights=False,
            )
            aligned.append(value)
        cross_modal, head_weights = self.dynamic_heads(aligned)

        channel_mix = torch.sigmoid(self.channel_mix_logit).view(1, 1, -1)
        # The caller already preserves the numerical identity path.  CMA must
        # therefore emit only a cross-modal correction; subtracting ``query``
        # here would remove the numerical path a second time after an unrelated
        # seq_len projection and makes the residual unnecessarily hard to fit.
        aligned_delta = channel_mix * cross_modal
        variable_gate = self.variable_gate(
            torch.cat([query, prompt_token, cross_modal], dim=-1)
        )
        internal_strength = torch.sigmoid(self.residual_logit)
        residual = self.output_projection(aligned_delta) * variable_gate
        residual = internal_strength * residual.transpose(1, 2)

        self.last_head_weights = head_weights.detach()
        self.last_channel_mix = channel_mix.detach()
        return residual, variable_gate, internal_strength, head_weights
