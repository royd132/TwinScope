import torch
from torch import nn
from layers.AMPD_Transformer_EncDec import Encoder, EncoderLayer
from layers.AMPD_SelfAttention_Family import FullAttention, AttentionLayer
from layers.AMPD_Embed import PatchEmbedding


# nf即为head_nf,即为d_model * (int((input_size - patch_len) / stride) + 2),即为d*N2
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, target_window)
        # self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(nf*n_vars, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num],即[B, N1, d, N2]
        x = self.flatten(x) # [B, N1, d, N2] -> [B, N1*d*N2]
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FlattenHead2(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, target_window)
        # self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, d, N2]
        x = self.flatten(x) # [B, d, N2] -> [B, d*N2]
        x = self.linear(x) # [B, d*N2] -> [B, H]
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    def __init__(self, input_size, enc_in, output_size, d_model, dropout, factor, output_attention, n_heads, d_ff, activation, e_layers, patch_len=4, stride=2):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(PatchTST, self).__init__()
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.head_nf = d_model * \
                       (int((input_size - patch_len) / stride) + 2)
        # self.head = FlattenHead(enc_in, self.head_nf, output_size,
        #                             head_dropout=dropout)
        self.head = FlattenHead2(self.head_nf, output_size,
                                    head_dropout=dropout)

    def forward(self, x_enc): # [B, L, N1] -> [B, H]

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) 
        enc_out, n_vars = self.patch_embedding(x_enc) 

        # Encoder
        enc_out, attns = self.encoder(enc_out) 
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) 
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, N1, N2, d] -> [B, N1, d, N2]
        enc_out = enc_out[:, 0, :, :] # [B, N1, d, N2] -> [B, d, N2]

        # Decoder
        dec_out = self.head(enc_out)  # [B, d, N2] -> [B, d*N2] -> [B, H]
        return dec_out




# 源代码的PatchTST的输入是(B, L, N1), N1是变量数
# PatchTST里面没有factor这个参数,只有Informer里面有这个参数
# mymodel = PatchTST(104,4,16,128,0.01,5,False,4,256,'relu',1,4,2)
# src = torch.rand(128, 104, 4)
# out = mymodel(src)
# print(out.shape)
# torch.Size([128, 16])


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = PatchTST(configs.seq_len, configs.enc_in, configs.pred_len,
                              int(getattr(configs, "patchtst_d_model", 128)),
                              float(getattr(configs, "patchtst_dropout", 0.2)),
                              5, False,
                              int(getattr(configs, "patchtst_heads", 16)),
                              int(getattr(configs, "patchtst_d_ff", 256)), nn.functional.gelu,
                              int(getattr(configs, "patchtst_layers", 3)),
                              patch_len=int(getattr(configs, "patchtst_patch_len", 16)),
                              stride=int(getattr(configs, "patchtst_stride", 8)))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        out = self.model(torch.cat([x_enc[..., -1:], x_enc[..., :-1]], dim=-1))
        return out.unsqueeze(-1)

