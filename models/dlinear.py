# DLinear, NLinear, Linear, RLinear


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # Python中的//是整除的意思,即先做除法,再向下取整
        # x: [Batch, Input length, Channel]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # 一阶池化对输入(N,C,L)的最后一维进行池化
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# 基于季节趋势性分解的线性模型
# [B, L, M=1] -> [B, H]
class DLinear(nn.Module): # [Batch, Input length, Channel=1] -> [Batch, Output length]
    def __init__(self, input_size, output_size):
        super(DLinear, self).__init__()

        # Decompsition Kernel Size
        # kernel_size = 25 # 原论文中的
        kernel_size = 17 #我自己设置的
        self.decompsition = series_decomp(kernel_size)
           
        self.Linear_Seasonal = nn.Linear(input_size, output_size)
        self.Linear_Trend = nn.Linear(input_size, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x[:, :, 0].unsqueeze(-1) # [B, L, M] -> [B, L, 1]
        seasonal_init, trend_init = self.decompsition(x)
        # 展平操作
        seasonal_init = self.flatten(seasonal_init) # [B, L, M] -> [B, L*M]=[B, L]
        trend_init = self.flatten(trend_init) # [B, L, M] -> [B, L*M]=[B, L]
        # 两个全连接层/线性层分别用来学习趋势和季节性
        seasonal_output = self.Linear_Seasonal(seasonal_init) # [B, L] -> [B, H]
        trend_output = self.Linear_Trend(trend_init) # [B, L] -> [B, H]

        out = seasonal_output + trend_output
        # 输出x的维度为[B, H]
        return out


# 基于简单实例归一化的单层线性模型,目前代码还有问题
# 感觉RevIN不怎么适用于多输入单输出的情况,且如果分布漂移不明显的话,RevIN没什么提升效果
class NLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(NLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,0].detach()
        x = x - seq_last
        x = self.flatten(x)
        seq_last = self.flatten(seq_last)
        x = self.fc1(x)
        out = x + seq_last
        return out # [Batch, Output length]
    

# 单层线性层,输入为13*3,输出为1
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # flatten操作的作用是将多维输入的除了第一维后面的其它所有维度展平
        x = self.flatten(x)
        # 第一层全连接层
        out = self.fc1(x)
        return out
    

# 基于RevIN的线性模型
class RLinear(nn.Module):
    pass
class Model(DLinear):
    def __init__(self, configs):
        super().__init__(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        out = super().forward(
            torch.cat([x_enc[..., -1:], x_enc[..., :-1]], dim=-1)
        )
        return out.unsqueeze(-1)

