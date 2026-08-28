"""Cross-modal alignment between numerical history and frozen prompt features."""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveHeadAggregation(nn.Module):
    """Route independent CMA experts per sample and variable."""

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

    def forward(
        self, head_outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(head_outputs, dim=2)
        batch, variables, heads, width = stacked.shape
        logits = self.router(stacked.reshape(batch, variables, heads * width))
        weights = torch.softmax(logits, dim=-1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=2), weights


class AdaptiveMultiHeadCMAResidual(nn.Module):
    """Return a bounded history residual from adaptive cross-modal experts."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        prompt_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("CMA must contain at least one independent head")

        self.prompt_dim = int(prompt_dim)
        self.numeric_projection = nn.Linear(seq_len, d_model)
        self.prompt_projection = nn.Linear(self.prompt_dim, d_model)
        self.state_projection = nn.Linear(d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.prompt_norm = nn.LayerNorm(d_model)
        self.cma_heads = nn.ModuleList(
            nn.MultiheadAttention(
                d_model,
                num_heads=1,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(int(num_heads))
        )
        self.dynamic_heads = AdaptiveHeadAggregation(num_heads, d_model, dropout)
        self.channel_mix_logit = nn.Parameter(torch.zeros(d_model))
        self.variable_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Linear(d_model, seq_len)
        self.residual_logit = nn.Parameter(torch.tensor(-1.3863))
        nn.init.normal_(self.output_projection.weight, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

        self.last_head_weights: torch.Tensor | None = None
        self.last_channel_mix: torch.Tensor | None = None

    def forward(
        self,
        history: torch.Tensor,
        prompt: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (history.size(0), history.size(2), self.prompt_dim)
        if prompt.ndim != 3 or tuple(prompt.shape) != expected:
            raise ValueError(
                "CMA prompt must be [batch, variables, prompt_dim], got "
                f"{tuple(prompt.shape)} for history {tuple(history.shape)}"
            )

        numeric = self.numeric_projection(history.transpose(1, 2))
        state_token = self.state_projection(state).unsqueeze(1)
        query = self.query_norm(numeric + state_token)
        prompt_token = self.prompt_norm(
            self.prompt_projection(prompt.to(dtype=history.dtype))
        )

        aligned = [
            head(query, prompt_token, prompt_token, need_weights=False)[0]
            for head in self.cma_heads
        ]
        cross_modal, head_weights = self.dynamic_heads(aligned)
        channel_mix = torch.sigmoid(self.channel_mix_logit).view(1, 1, -1)
        gate = self.variable_gate(torch.cat([query, prompt_token, cross_modal], dim=-1))
        internal_strength = torch.sigmoid(self.residual_logit)
        residual = self.output_projection(channel_mix * cross_modal) * gate
        residual = internal_strength * residual.transpose(1, 2)

        self.last_head_weights = head_weights.detach()
        self.last_channel_mix = channel_mix.detach()
        return residual, gate, internal_strength, head_weights
