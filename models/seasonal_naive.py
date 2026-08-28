"""Seasonal naive baseline."""
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, configs): super().__init__(); self.horizon=configs.pred_len; self.cycle=getattr(configs, "cycle", 96)
    def forward(self, x, cycle=None, future_x=None):
        if x.size(1)<self.cycle: return x[:,-1:, -1].expand(-1,self.horizon)
        idx=x.size(1)-self.cycle+torch.arange(self.horizon,device=x.device)%self.cycle; return x.index_select(1,idx)[...,-1]
