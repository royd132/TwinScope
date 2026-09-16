"""LSTM baseline, public implementation from

https://github.com/laowu-code/TimeFrequency_EndogenousExogenousDecoupling_PVPowerForecast
(models/LSTM.py), exposed through the local shared forecasting interface.

Public single configuration (pre_deterministic_solar.py):
d_model=128, e_layers=1.
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Long Short Term Memory"""

    def __init__(
        self,
        enc_in,
        d_model,
        e_layers,
        pred_len,
        seq_len=None,
        bidirectional=False,
        prob=False,
    ):
        super(LSTM, self).__init__()
        self.input_size = enc_in
        self.hidden_size = d_model
        self.num_layers = e_layers
        self.output_size = pred_len
        self.bidirectional = bidirectional
        self.prob = prob
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if not self.prob:
            self.fc = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size * 7)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        if self.prob:
            out = out.view(-1, self.output_size, 7)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LSTM(
            enc_in=configs.enc_in,
            d_model=int(getattr(configs, "lstm_d_model", 128)),
            e_layers=int(getattr(configs, "lstm_layers", 1)),
            pred_len=configs.pred_len,
            seq_len=configs.seq_len,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        x = torch.cat([x_enc[..., -1:], x_enc[..., :-1]], dim=-1)
        out = self.model(x)
        return out.unsqueeze(-1)
