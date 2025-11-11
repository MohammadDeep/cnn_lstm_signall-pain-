import torch
import torch.nn as nn


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, groups=1, act='relu'):
        super().__init__()
        p = (k//2) * d  # same-ish برای S=1
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p,
                              dilation=d, groups=groups, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act=='relu' else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MODEL_CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes, hid=32, k1=7, k2=5):
        super().__init__()
        self.block1 = ConvBNAct1d(in_ch, hid, k=k1)
        self.head   = nn.Conv1d(hid, n_classes, k2, padding=(k2//2), bias=True)
    def forward(self, x):
        z = self.block1(x)        # (B,hid,T)
        z = self.head(z)          # (B,n_classes,T)
        return z.mean(-1)         # logits: (B,n_classes)

import torch
from torchinfo import summary

# مدلِ خودت
model = MODEL_CNN1D(in_ch=4, n_classes=6, hid=32, k1=7, k2=5)

# ابعاد ورودی نمونه: (Batch, C_in, T)
B, C_in, T = 2, 4, 512
summary(model, input_size=(B, C_in, T),
        col_names=("input_size","output_size","num_params","kernel_size","trainable"),
        depth=3)
