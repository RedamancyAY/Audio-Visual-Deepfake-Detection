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

# +
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

# + tags=["style-activity"]
from .conv_attention import MLP, Attention
from .utils import weight_init, LayerNorm

# + tags=["active-ipynb"]
# from conv_attention import MLP, Attention
# from utils import weight_init, LayerNorm
# -

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# # Spatial Block

# +
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


# -

# ## Spatial Pooler
#

class SpatialPooler(nn.Module):

    def __init__(self, dim, n_frames):
        super().__init__()

        # self.channel_attn = ChannelAttention(in_planes=n_frames, ratio=1)

        self.pool_layer = nn.Conv3d(n_frames, n_frames, kernel_size=1, bias=False)
        self.unpool_layer = nn.Conv3d(n_frames, n_frames, kernel_size=1, bias=False)

        # self.pool_layer = nn.Conv3d(
        #     n_frames, n_frames, kernel_size=3, padding=1, bias=False
        # )
        # self.unpool_layer = nn.Conv3d(
        #     n_frames, n_frames, kernel_size=3, padding=1, bias=False
        # )

        self.n_frames = n_frames

    def pool(self, x):
        x = x.transpose(1, 2)
        # x = self.channel_attn(x)
        x = self.pool_layer(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        return x

    def unpool(self, x):
        x = rearrange(x, "(b t) c h w -> b t c h w", t=self.n_frames)
        x = self.unpool_layer(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x, operation="pool"):
        assert operation in ["pool", "unpool"]
        if operation == "pool":
            return self.pool(x)
        else:
            return self.unpool(x)


# ## block

class TempBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()

        self.attn = Attention(dim)
        self.mlp = MLP(dim, mlp_ratio)
        alpha_0 = 1e-2
        self.alpha_1 = nn.Parameter(alpha_0 * torch.ones((dim)), requires_grad=True)
        self.alpha_2 = nn.Parameter(alpha_0 * torch.ones((dim)), requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(weight_init)

    def forward(self, x):
        x = x + self.drop_path(self.alpha_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.alpha_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class SpatialBlock(nn.Module):
    def __init__(self, n_frames, dim, mlp_ratio=4.0):
        super().__init__()
        self.block = TempBlock(dim=dim, mlp_ratio=mlp_ratio)
        self.pooler = SpatialPooler(dim=dim, n_frames=n_frames)

    def forward(self, x):
        x = self.pooler.pool(x)
        x = self.block(x)
        x = self.pooler.unpool(x)
        return x
