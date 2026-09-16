import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from TSLANet import Adaptive_Spectral_Block
except ImportError:
    Adaptive_Spectral_Block = nn.Identity


# B：batch size
# M：多变量序列的变量数
# L：过去序列的长度
# T: 预测序列的长度
# N: 分Patch后Patch的个数
# D：每个变量的通道数/特征数,这里的通道数/特征数不是指变量数
# P：kernel size of embedding layer,即patch化中的卷积核大小
# S：stride of embedding layer
# TimeMixer, UniRepLKNet, MICN, TDformer, Pathformer, SE block, GLU, ConvGLU, 周期趋势性分解
# 下面是我受到ModernTCN的启发,自己实现的一个HLLmodel模型



# RevIN,我试过了,在光伏功率预测这几个数据集上效果都不是很好,加了不如不加
class RevIN(nn.Module): # B, L, D -> B, L, D
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# embedding层,实现的是时间维度降采样+隐特征增维
class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=64, dropout=0.02):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=D, 
            kernel_size=P, 
            stride=S
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
            )  # [B*M, 1, L] -> [B*M, 1, L+P-S]
        
        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]

        x_emb = self.dropout(x_emb)
        return x_emb  # x_emb: [B, M, D, N]


# 第2种embedding层,将输入序列分成patch
class PatchEmbedding(nn.Module):
    def __init__(self, P=8, S=4):
        super(PatchEmbedding, self).__init__()
        self.P = P
        self.S = S
    
    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        # 对x的最后一个维度进行填充,重复填充P-S次
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
            ) # [B, M, L] -> [B, M, L+P-S]
        # 将x_pad沿着最后一个维度分成patch,patch的长度为P,步长为S
        # x_emb = rearrange(x_pad, 'b m l -> b m (l p) s', p=self.P, s=self.S, b=B)  # [B, M, L+P-S] -> [B, M, N, P]
        # x_emb = x_emb.permute(0, 1, 3, 2)  # [B, M, N, P] -> [B, M, P, N]
        x_pad = x_pad.unfold(dimension=-1, size=self.P, step=self.S)
        x_emb = torch.reshape(x_pad, (x_pad.shape[0], x_pad.shape[1], x_pad.shape[2], x_pad.shape[3]))
        x_emb = x_emb.permute(0, 1, 3, 2)  # [B, M, N, P] -> [B, M, P, N]
        return x_emb





# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,并将卷积的结果进行逐元素相加
class MultiDWConv(nn.Module):
    def __init__(self, M, D, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4]):
        super(MultiDWConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                dilation=d2,
                groups=M*D, 
                padding='same'
                ) for kernel_size, d2 in zip(kernel_sizes, dilat)
            ])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [conv(x) for conv in self.convs]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,并将卷积的结果进行逐元素相加,每个卷积后面接一个BN层
class MultiDWConvBN(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,9,11]):
        super(MultiDWConvBN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                padding='same'
                ) for kernel_size in kernel_sizes
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [bn(conv(x)) for conv, bn in zip(self.convs, self.bns)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,并将卷积的结果进行逐元素相加,每个卷积后面接一个BN层
# 没有自适应权重聚合
class MultiDWConvBN2(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,15], dilat=[4,3,2,1]):
        super(MultiDWConvBN2, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                dilation=d2,
                padding='same'
                ) for kernel_size, d2 in zip(kernel_sizes, dilat)
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [bn(conv(x)) for conv, bn in zip(self.convs, self.bns)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,并将卷积的结果进行逐元素相加,每个卷积后面先接一个relu激活函数,再接一个BN层
class MultiDWConvReluBN(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,9,11]):
        super(MultiDWConvReluBN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                padding='same'
                ) for kernel_size in kernel_sizes
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [bn(F.relu(conv(x))) for conv, bn in zip(self.convs, self.bns)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]  


class MultiDWConvWeightedSum2(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,15], dilat=[4,3,2,1]):
        super(MultiDWConvWeightedSum2, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                dilation=d2,
                padding='same'
                ) for kernel_size, d2 in zip(kernel_sizes, dilat)
            ])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])
        # self.weights1 = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in kernel_sizes])
        self.weights1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0/len(kernel_sizes))) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        # weights2 = torch.softmax(torch.stack([w for w in self.weights1]), dim=0)
        x = [weight*conv(x) for conv, weight in zip(self.convs, self.weights1)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,每个卷积后面接一个BN层,每个卷积得到的特征图和原始特征图的形状相同,且将卷积的结果进行有权重地逐元素相加,该权重是可学习的参数
class MultiDWConvBNWeightedSum(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,9,11]):
        super(MultiDWConvBNWeightedSum, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                padding='same'
                ) for kernel_size in kernel_sizes
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])
        # self.weights1 = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in kernel_sizes])
        self.weights1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0/len(kernel_sizes))) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        # weights2 = torch.softmax(torch.stack([w for w in self.weights1]), dim=0)
        x = [weight*bn(conv(x)) for conv, bn, weight in zip(self.convs, self.bns, self.weights1)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 并行多卷积核的逐深度卷积层DWConv,实现的是多尺度逐深度卷积,卷积核的大小分别为3,5,7,9,11,每个卷积后面接一个BN层,每个卷积得到的特征图和原始特征图的形状相同,且将卷积的结果进行有权重地逐元素相加,该权重是可学习的参数
class MultiDWConvBNWeightedSum2(nn.Module):
    def __init__(self, M, D, kernel_sizes=[3,5,7,15], dilat=[4,3,2,1]):
        super(MultiDWConvBNWeightedSum2, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                dilation=d2,
                padding='same'
                ) for kernel_size, d2 in zip(kernel_sizes, dilat)
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])
        # self.weights1 = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in kernel_sizes])
        self.weights1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0/len(kernel_sizes))) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        # weights2 = torch.softmax(torch.stack([w for w in self.weights1]), dim=0)
        x = [weight*bn(conv(x)) for conv, bn, weight in zip(self.convs, self.bns, self.weights1)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


class MultiDWConvBNWeightedSum3(nn.Module):
    def __init__(self, M, D, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4]):
        super(MultiDWConvBNWeightedSum3, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=M*D, 
                    out_channels=M*D*4, 
                    kernel_size=kernel_size, 
                    groups=M*D, 
                    dilation=d2,
                    padding='same'
                    ),
                nn.Conv1d(
                    in_channels=M*D*4, 
                    out_channels=M*D, 
                    kernel_size=kernel_size, 
                    groups=M*D,
                    dilation=d2, 
                    padding='same'
                    )
                ) for kernel_size, d2 in zip(kernel_sizes, dilat)
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])
        self.weights1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0/len(kernel_sizes))) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [weight*bn(conv(x)) for conv, bn, weight in zip(self.convs, self.bns, self.weights1)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


# 纺锤形卷积FFN,组卷积逐点卷积FFN
# nn.Conv1d的输入的shape是(Batchsize, C_in, L_in), 输出的shape是(Batchsize, C_out, L_out),对输入张量的最后一个维度(通常表示序列长度)进行卷积操作
class ConvFFN(nn.Module):
    def __init__(self, M, D, r=2, dropout=0.02, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            groups=groups_num
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=r*M*D, 
            out_channels=M*D, 
            kernel_size=1,
            groups=groups_num
            )

    def forward(self, x):
        # x: [B, M*D, N]
        # x = self.pw_con2(F.gelu(self.pw_con1(x)))
        # 上面的代码即为以下3行
        x = self.pw_con1(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.pw_con2(x)
        x = self.dropout(x)
        return x  # x: [B, M*D, N]


# 逐点组卷积GLU
class ConvGLU(nn.Module):
    def __init__(self, M, D, r=2, dropout=0.04, one=True):  # one is True: ConvGLU1, one is False: ConvGLU2
        super(ConvGLU, self).__init__()
        groups_num = M if one else D
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            groups=groups_num
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            groups=groups_num
            )
        self.pw_con3 = nn.Conv1d(
            in_channels=r*M*D, 
            out_channels=M*D, 
            kernel_size=1,
            groups=groups_num
            )

    def forward(self, x):
        # x: [B, M*D, N]
        # 上面的代码即为以下3行
        x1 = self.pw_con1(x)
        # x1 = self.dropout(x1)
        x2 = self.pw_con2(x)
        # x2 = self.dropout(x2)
        # x2经过sigmoid激活函数,然后与x1相乘
        x3 = x1 * torch.sigmoid(x2)
        x3 = self.dropout1(x3)
        x4 = self.pw_con3(x3)
        x4 = self.dropout2(x4)
        # return x+x4  # x: [B, M*D, N]
        return x4


# 未解耦的组卷积GLU, Undecoupled ConvGLU
class UndeConvGLU(nn.Module): # [B, M*D, N] -> [B, M*D, N]
    def __init__(self, M, D, r=2, dropout=0.02):
        super(UndeConvGLU, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            )
        self.pw_con3 = nn.Conv1d(
            in_channels=r*M*D, 
            out_channels=M*D, 
            kernel_size=1,
            )

    def forward(self, x):
        # x: [B, M*D, N]
        # 上面的代码即为以下3行
        x1 = self.pw_con1(x) # [B, M*D, N] -> [B, r*M*D, N]
        x2 = self.pw_con2(x) # [B, M*D, N] -> [B, r*M*D, N]
        # x2经过sigmoid激活函数,然后与x1相乘
        x3 = x1 * torch.sigmoid(x2) # [B, r*M*D, N] -> [B, r*M*D, N]
        x3 = self.dropout1(x3) # [B, r*M*D, N] -> [B, r*M*D, N]
        x4 = self.pw_con3(x3) # [B, r*M*D, N] -> [B, M*D, N]
        x4 = self.dropout2(x4) # [B, M*D, N] -> [B, M*D, N]
        # return x+x4  # x: [B, M*D, N]
        return x4



# 我自己的模型1
class ConvPVNetBlock(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPVNetBlock, self).__init__()
        # self.asb = Adaptive_Spectral_Block(D)
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        # self.dropout = nn.Dropout(0.01)
        # self.multi_dw_conv = MultiDWConv(M, D, kernel_sizes, dilat)
        # self.multi_dw_conv = MultiDWConvBN2(M, D, kernel_sizes, dilat)

        # self.bn1 = nn.BatchNorm1d(M*D)
        # self.ln1 = nn.LayerNorm(N)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=True)
        # self.conv_glu1 = ConvFFN(M, D, r, dp, one=True)
        # self.bn2 = nn.BatchNorm1d(M*D)
        # self.ln2 = nn.LayerNorm(N)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=False)
        # self.conv_glu2 = ConvFFN(M, D, r, dp, one=False)
        # self.bn3 = nn.BatchNorm1d(D*M)
        # self.ln3 = nn.LayerNorm(N)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        # M = x_emb.shape[-3]
        D = x_emb.shape[-2]
        # x = rearrange(x_emb, 'b m d n -> (b m) d n')          # [B, M, D, N] -> [B*M, D, N]
        # x = x.permute(0,2,1)                                  # [B*M, D, N] -> [B*M, N, D]
        # x = self.asb(x)                                       # [B*M, N, D] -> [B*M, N, D]
        # x1 = rearrange(x, '(b m) n d -> b (m d) n', m=M)        # [B*M, N, D] -> [B, M*D, N]

        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        # x = self.dropout(x)
        # x = x + x1
        # x = x + self.bn1(x1)
        # x = self.bn1(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln1(x)
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        # x = self.bn2(x)                                       # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln2(x)

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]
        # x = self.bn3(x)                                       # [B, D*M, N] -> [B, D*M, N]
        # x = self.ln3(x)

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        # out = x                                             # [B, M, D, N]
        out = x + x_emb                                       # [B, M, D, N]

        return out  # out: [B, M, D, N]
    

class ConvPVNetBlockASB(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPVNetBlockASB, self).__init__()
        self.asb = Adaptive_Spectral_Block(D)
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        # self.dropout = nn.Dropout(0.01)
        # self.multi_dw_conv = MultiDWConv(M, D, kernel_sizes, dilat)
        # self.multi_dw_conv = MultiDWConvBN2(M, D, kernel_sizes, dilat)

        # self.bn1 = nn.BatchNorm1d(M*D)
        # self.ln1 = nn.LayerNorm(N)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=True)
        # self.conv_glu1 = ConvFFN(M, D, r, dp, one=True)
        # self.bn2 = nn.BatchNorm1d(M*D)
        # self.ln2 = nn.LayerNorm(N)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=False)
        # self.conv_glu2 = ConvFFN(M, D, r, dp, one=False)
        # self.bn3 = nn.BatchNorm1d(D*M)
        # self.ln3 = nn.LayerNorm(N)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        M = x_emb.shape[-3]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> (b m) d n')          # [B, M, D, N] -> [B*M, D, N]
        x = x.permute(0,2,1)                                  # [B*M, D, N] -> [B*M, N, D]
        x = self.asb(x)                                       # [B*M, N, D] -> [B*M, N, D]
        x1 = rearrange(x, '(b m) n d -> b (m d) n', m=M)        # [B*M, N, D] -> [B, M*D, N]

        # x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        # x = self.dropout(x)
        # x = x + x1
        # x = x + self.bn1(x1)
        # x = self.bn1(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln1(x)
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        # x = self.bn2(x)                                       # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln2(x)

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]
        # x = self.bn3(x)                                       # [B, D*M, N] -> [B, D*M, N]
        # x = self.ln3(x)

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        # out = x                                             # [B, M, D, N]
        out = x + x_emb                                       # [B, M, D, N]

        return out  # out: [B, M, D, N]


# 我自己的模型2
# 试过了,效果不是很好
class ConvPVNetBlock2(nn.Module):
    def __init__(self, M, D, kernel_sizes, r):
        super(ConvPVNetBlock2, self).__init__()
        self.multi_dw_conv = MultiDWConvBN(M, D, kernel_sizes)
        # self.bn1 = nn.BatchNorm1d(M*D)
        # self.ln1 = nn.LayerNorm(N)
        self.conv_glu1 = ConvGLU(M, D, r, one=True)
        self.bn2 = nn.BatchNorm1d(M)
        # self.ln2 = nn.LayerNorm(N)
        self.conv_glu2 = ConvGLU(M, D, r, one=False)
        self.bn3 = nn.BatchNorm1d(D)
        # self.ln3 = nn.LayerNorm(N)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        N = x_emb.shape[-1]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        # x = x + x1
        # x = x + self.bn1(x1)
        # x = self.bn1(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln1(x)
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        # x = self.bn2(x)                                       # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln2(x)

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,3,1,2)                                # [B, M, D, N] -> [B, N, M, D]
        x = rearrange(x, 'b n m d -> (b n) m d')              # [B, D, M, N] -> [B*N, M, D]
        x = self.bn2(x)
        x = rearrange(x, '(b n) m d -> b n m d', n=N)         # [B*N, M, D] -> [B, N, M, D]
        x = x.permute(0,3,2,1)                                # [B, N, M, D] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]
        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,3,1,2)                                # [B, D, M, N] -> [B, N, D, M]
        x = rearrange(x, 'b n d m -> (b n) d m')              # [B, N, D, M] -> [B*N, D, M]
        x = self.bn3(x)
        x = rearrange(x, '(b n) d m -> b n d m', n=N)         # [B*N, D, M] -> [B, N, D, M]
        x = x.permute(0,3,2,1)                                # [B, N, D, M] -> [B, M, D, N]
        # x = self.bn3(x)                                       # [B, D*M, N] -> [B, D*M, N]
        # x = self.ln3(x)

        # x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        # x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        # out = x                                             # [B, M, D, N]
        out = x + x_emb                                       # [B, M, D, N]

        return out  # out: [B, M, D, N]


class ConvPVNet(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPVNet, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        # self.revin = RevIN(M)
        # 因为只在时间序列的尾巴填充P-S个,所以N = L // S
        N = L // S
        self.embed_layer = Embedding(P, S, D, 0)
        # self.backbone = nn.ModuleList([ConvPVNetBlock(M, D, N, kernel_sizes, r) for _ in range(num_layers)])
        # self.backbone = nn.ModuleList([ConvPVNetBlock(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.backbone = nn.ModuleList([ConvPVNetBlock(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        # x: [B, L, M]
        # x = self.revin(x, 'norm') # [B, L, M] -> [B, L, M]
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        # Flatten
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        # x = self.revin(x, 'denorm') # [B, D*N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]



# 各种子模块消融实验
# 原本模型是ConvGLU1+ConvGLU2
# ConvPV1是只有ConvGLU1
# ConvPV2是只有ConvGLU2
# ConvPV3是ConvGLU1+ConvGLU1
# ConvPV4是ConvGLU2+ConvGLU2
# ConvPV5是未解耦的单个ConvGLU
# ConvPV6是未解耦的2个ConvGLU
# ConvPV7是去掉AMSConv的模型

    
class ConvPV1Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV1Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        # self.multi_dw_conv = MultiDWConv(M, D, kernel_sizes, dilat)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=True)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x + x_emb                                       # [B, M, D, N]

        return x  # out: [B, M, D, N]

class ConvPV1(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV1, self).__init__()
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV1Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


class ConvPV2Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV2Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=False)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]
        x = x + x_emb                                         # [B, M, D, N]

        return x  # out: [B, M, D, N]


class ConvPV2(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV2, self).__init__()
        N = L // S
        self.num_layers = num_layers
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV2Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


class ConvPV3Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV3Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=True)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=True)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu2(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x + x_emb                                         # [B, M, D, N]

        return x  # out: [B, M, D, N]

class ConvPV3(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV3, self).__init__()
        N = L // S
        self.num_layers = num_layers
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV3Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


class ConvPV4Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV4Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=False)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=False)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        x = self.conv_glu1(x)                                 # [B, D*M, N] -> [B, D*M, N]
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]
        x = x + x_emb                                       # [B, M, D, N]

        return x  # out: [B, M, D, N]


class ConvPV4(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV4, self).__init__()
        N = L // S
        self.num_layers = num_layers
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV4Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


# 未解耦的ConvPVNet, 只用了一个ConvGLU
class ConvPV5Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV5Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        self.conv_glu1 = UndeConvGLU(M, D, r, dp)
        # self.conv_glu2 = UndeConvGLU(M, D, r)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        # x = self.conv_glu2(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x + x_emb                                       # [B, M, D, N]

        return x  # out: [B, M, D, N]


class ConvPV5(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV5, self).__init__()
        N = L // S
        self.num_layers = num_layers
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV5Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


# 未解耦的ConvPVNet, 用了2个ConvGLU
class ConvPV6Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV6Block, self).__init__()
        self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        self.conv_glu1 = UndeConvGLU(M, D, r, dp)
        self.conv_glu2 = UndeConvGLU(M, D, r, dp)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_glu2(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x + x_emb                                       # [B, M, D, N]

        return x  # out: [B, M, D, N]


class ConvPV6(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV6, self).__init__()
        N = L // S
        self.num_layers = num_layers
        self.embed_layer = Embedding(P, S, D, 0)
        self.backbone = nn.ModuleList([ConvPV6Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]


# 此模型无AMSConv,有Channel ConvGLU和Variate ConvGLU
class ConvPV7Block(nn.Module):
    def __init__(self, M, D, kernel_sizes, dilat, r, dp):
        super(ConvPV7Block, self).__init__()
        # self.multi_dw_conv = MultiDWConvBNWeightedSum2(M, D, kernel_sizes, dilat)
        # self.dropout = nn.Dropout(0.01)
        # self.multi_dw_conv = MultiDWConv(M, D, kernel_sizes, dilat)
        # self.multi_dw_conv = MultiDWConvBN2(M, D, kernel_sizes, dilat)

        # self.bn1 = nn.BatchNorm1d(M*D)
        # self.ln1 = nn.LayerNorm(N)
        self.conv_glu1 = ConvGLU(M, D, r, dp, one=True)
        # self.conv_glu1 = ConvFFN(M, D, r, dp, one=True)
        # self.bn2 = nn.BatchNorm1d(M*D)
        # self.ln2 = nn.LayerNorm(N)
        self.conv_glu2 = ConvGLU(M, D, r, dp, one=False)
        # self.conv_glu2 = ConvFFN(M, D, r, dp, one=False)
        # self.bn3 = nn.BatchNorm1d(D*M)
        # self.ln3 = nn.LayerNorm(N)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        x1 = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        # x = self.multi_dw_conv(x1)                             # [B, M*D, N] -> [B, M*D, N]
        x = x1
        # x = self.dropout(x)
        # x = x + x1
        # x = x + self.bn1(x1)
        # x = self.bn1(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln1(x)
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_glu1(x)                                 # [B, M*D, N] -> [B, M*D, N]
        # x = self.bn2(x)                                       # [B, M*D, N] -> [B, M*D, N]
        # x = self.ln2(x)

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_glu2(x)                                 # [B, D*M, N] -> [B, D*M, N]
        # x = self.bn3(x)                                       # [B, D*M, N] -> [B, D*M, N]
        # x = self.ln3(x)

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        # out = x                                             # [B, M, D, N]
        out = x + x_emb                                       # [B, M, D, N]

        return out  # out: [B, M, D, N]


class ConvPV7(nn.Module):
    def __init__(self, M, L, T, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=64, P=4, S=2, r=2, dp=0.02, num_layers=1):
        super(ConvPV7, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        # self.revin = RevIN(M)
        # 因为只在时间序列的尾巴填充P-S个,所以N = L // S
        N = L // S
        self.embed_layer = Embedding(P, S, D, 0)
        # self.backbone = nn.ModuleList([ConvPVNetBlock(M, D, N, kernel_sizes, r) for _ in range(num_layers)])
        self.backbone = nn.ModuleList([ConvPV7Block(M, D, kernel_sizes, dilat, r, dp) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.dropout = nn.Dropout(0.025)

    def forward(self, x):
        # x: [B, L, M]
        # x = self.revin(x, 'norm') # [B, L, M] -> [B, L, M]
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度变量维度M中取出第一个变量Power变量
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        # Flatten
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.dropout(pred)

        return pred  # out: [B, T]




# 输入为[B, M, L], 输出为[B, T]
# past_series = torch.rand(128, 4, 96)
# model = ModernTCN(4, 96, 192)
# pred_series = model(past_series)
# print(pred_series.shape)
# # torch.Size([128, 192])
# past1 = torch.rand(2, 4, 96)
# past2 = torch.rand(2, 4, 96)
# past3 = torch.rand(2, 4, 96)
# past4 = torch.rand(2, 4, 96)
# past5 = torch.rand(2, 4, 96)
# past = [past1, past2, past3, past4, past5]
# past = torch.stack(past, dim=1)
# past_sum = past.sum(dim=1)
# # past = torch.cat((past1, past2, past3, past4, past5), dim=2)
# print(past.shape)
# print(past_sum.shape)
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPVNet
# past_series = torch.rand(128, 104, 4)
# model = ConvPVNet(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])


# 输入为[B, L, M], 输出为[B, T],测试ConvPV1
# past_series = torch.rand(128, 104, 4)
# model = ConvPV1(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV2
# past_series = torch.rand(128, 104, 4)
# model = ConvPV2(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV3
# past_series = torch.rand(128, 104, 4)
# model = ConvPV3(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV4
# past_series = torch.rand(128, 104, 4)
# model = ConvPV4(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV5
# past_series = torch.rand(128, 104, 4)
# model = ConvPV5(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV6
# past_series = torch.rand(128, 104, 4)
# model = ConvPV6(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])
    

# 输入为[B, L, M], 输出为[B, T],测试ConvPV7
# past_series = torch.rand(128, 104, 4)
# model = ConvPV7(4, 104, 52)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 52])



# 测试一下ConvGLU的输入输出
# past_series = torch.rand(128, 4*64, 78)
# model = UndeConvGLU(4, 64, 8)
# pred_series = model(past_series)
# print(pred_series.shape)
# torch.Size([128, 256, 78])
    

# 测试MultiDWConvWeightedSum的输入输出
# input = torch.rand(128, 4*64, 78)
# model = MultiDWConv(4, 64)
# output = model(input)
# print(output.shape)  
# torch.Size([128, 256, 78])
    

# 测试PatchEmbedding的输入输出
# input = torch.rand(128, 4, 104)
# model = PatchEmbedding(4, 2)
# output = model(input)
# print(output.shape)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        stride = 4 if configs.pred_len >= 16 else 2
        self.model = ConvPVNet(configs.enc_in, configs.seq_len, configs.pred_len,
                               D=64, P=4, S=stride, num_layers=1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        prediction = self.model(
            torch.cat([x_enc[..., -1:], x_enc[..., :-1]], dim=-1)
        )
        return prediction.unsqueeze(-1)

