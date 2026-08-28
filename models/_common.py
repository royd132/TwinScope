"""Small reusable forecasting blocks shared by local model files."""
from __future__ import annotations
import torch
from torch import nn

class LastValue(nn.Module):
    def forward(self, x, cycle=None, future_x=None):
        horizon = future_x.shape[1] if future_x is not None else self.horizon
        return x[:, -1:, -1].expand(-1, horizon)

class ResidualMLP(nn.Module):
    def __init__(self, seq_len, horizon, channels=1, hidden=128, dropout=0.1):
        super().__init__(); self.horizon = int(horizon)
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(nn.Linear(seq_len * channels, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, horizon))
    def forward(self, x, cycle=None, future_x=None):
        return self.net(self.norm(x).flatten(1))

class PatchForecaster(nn.Module):
    def __init__(self, seq_len, horizon, channels=1, patch_len=16, hidden=128):
        super().__init__(); self.patch = nn.Conv1d(channels, hidden, patch_len, stride=max(1, patch_len // 2)); self.head = nn.Linear(hidden, horizon)
    def forward(self, x, cycle=None, future_x=None):
        return self.head(self.patch(x.transpose(1, 2)).mean(-1))

class SpectralForecaster(nn.Module):
    def __init__(self, seq_len, horizon, channels=1, hidden=128):
        super().__init__(); self.temporal = nn.Sequential(nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU()); self.frequency = nn.Linear(hidden, hidden); self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizon))
    def forward(self, x, cycle=None, future_x=None):
        z = self.temporal(x.transpose(1, 2)).transpose(1, 2); f = torch.fft.rfft(z, dim=1).abs().mean(1); return self.head(z[:, -1] + torch.tanh(self.frequency(f)))
