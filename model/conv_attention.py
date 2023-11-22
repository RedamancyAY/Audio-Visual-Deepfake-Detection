# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # import packages

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# + tags=["style-activity"]
from .utils import weight_init, LayerNorm


# + tags=["active-ipynb"]
# from utils import weight_init, LayerNorm
# -

# ### MLP

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)

        _dim = int(dim * mlp_ratio)
        self.cff = nn.Sequential(
            nn.Conv2d(dim, _dim, 1),
            nn.Conv2d(_dim, _dim, 3, 1, padding=1, bias=True, groups=_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(_dim, dim, 1),
            nn.Dropout(drop),
        )
        self.apply(weight_init)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.cff(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.ReLU(inplace=True)
        self.LKA = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            # nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3),
            # nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2),
            nn.Conv2d(dim, dim, 1),
        )
        self.proj_2 = nn.Conv2d(dim, dim, 1)

        self.apply(weight_init)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.LKA(x) * x
        x = self.proj_2(x)
        x = x + shorcut
        return x
