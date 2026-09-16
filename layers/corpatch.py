"""Multi-scale CorPatch encoder and numerical forecast head."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleVariablePatch(nn.Module):
    """Patch each variable independently at several physical time scales."""

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
    """Linear-complexity temporal mixing within one patch scale."""

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
    """Encode temporal patches, attend across variables, and route scales."""

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
        self.state_router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, scales))
        self.variable_scale_router = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)

    def encode_scales(self, patches: list[torch.Tensor]) -> torch.Tensor:
        summaries = []
        for patch_tokens, temporal_mixer, variable_attention in zip(
            patches,
            self.temporal_mixers,
            self.variable_attentions,
            strict=True,
        ):
            batch, variables, patch_count, width = patch_tokens.shape
            temporal = temporal_mixer(
                patch_tokens.reshape(batch * variables, patch_count, width)
            ).reshape(batch, variables, patch_count, width)
            variable_tokens = temporal.mean(dim=2)
            related, _ = variable_attention(
                variable_tokens,
                variable_tokens,
                variable_tokens,
                need_weights=False,
            )
            summaries.append(variable_tokens + related)
        return torch.stack(summaries, dim=2)

    def fuse_scales(
        self,
        scale_features: torch.Tensor,
        state: torch.Tensor,
        semantic_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.state_router(state).unsqueeze(1)
        logits = logits + self.variable_scale_router(scale_features).squeeze(-1)
        if semantic_bias is not None:
            if semantic_bias.shape != logits.shape:
                raise ValueError(
                    f"semantic scale bias must be {tuple(logits.shape)}, "
                    f"got {tuple(semantic_bias.shape)}"
                )
            logits = logits + semantic_bias
        weights = torch.softmax(logits, dim=-1)
        fused = (scale_features * weights.unsqueeze(-1)).sum(dim=2)
        return self.output_norm(fused), weights


class ForecastHead(nn.Module):
    """Forecast from the target-variable token and global variable context."""

    def __init__(self, d_model: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, variables: torch.Tensor) -> torch.Tensor:
        target = variables[:, -1]
        context = variables.mean(dim=1)
        return self.projection(torch.cat([target, context], dim=-1))
