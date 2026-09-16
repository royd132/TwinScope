"""Reversible instance normalization shared by forecasting models."""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
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
