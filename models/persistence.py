"""Persistence baseline: repeat the last observed target value."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.horizon = int(getattr(configs, "pred_len", 1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return x_enc[:, -1:, -1].expand(-1, self.horizon).unsqueeze(-1)
