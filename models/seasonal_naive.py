"""Seasonal naive baseline: repeat the most recent daily shape."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.horizon = int(configs.pred_len)
        self.cycle = int(getattr(configs, "cycle", 96))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        if x_enc.size(1) < self.cycle:
            out = x_enc[:, -1:, -1].expand(-1, self.horizon)
        else:
            start = x_enc.size(1) - self.cycle
            idx = start + torch.arange(self.horizon, device=x_enc.device) % self.cycle
            out = x_enc.index_select(1, idx)[..., -1]
        return out.unsqueeze(-1)
