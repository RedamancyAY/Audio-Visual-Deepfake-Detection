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
from .utils import Multi_Head_Attention, weight_init, LayerNorm
from .block_temporal import TemporalBlock
from .block_spatial import SpatialBlock


# + tags=["active-ipynb"]
# from utils import Multi_Head_Attention, weight_init, LayerNorm
# from block_temporal import TemporalBlock
# from block_spatial import SpatialBlock
# -

class Block(nn.Module):
    def __init__(
        self,
        video_dim,
        n_frames,
        audio_dim,
        window_size=7,
        mlp_ratio=4.0,
        audio_length=None,
        video_size=None,
        attn_dropout=0.,
    ):
        super().__init__()

        self.spatial_block = SpatialBlock(n_frames, video_dim, mlp_ratio)
        self.temporal_block = TemporalBlock(
            video_dim=video_dim,
            audio_dim=audio_dim,
            window_size=7,
            n_frames=n_frames,
            audio_length=audio_length,
            video_size=video_size,
            attn_dropout=attn_dropout
        )

        # self.video_norm = LayerNorm(video_dim)
        # self.audio_norm = LayerNorm(audio_dim)
        self.video_norm1 = nn.BatchNorm3d(video_dim)
        self.video_norm2 = nn.BatchNorm3d(video_dim)
        self.audio_norm = nn.BatchNorm1d(audio_dim)
        self.apply(weight_init)

    def forward(self, data, grad_cam=False):
        video, audio = data
        B, C, T, H, W = video.shape

        x = self.spatial_block(video)
        x = self.video_norm1(x)
        
        # if grad_cam:
        #     x, y, features = self.temporal_block(x, audio, grad_cam=grad_cam)
        #     x = self.video_norm2(x)
        #     y = self.audio_norm(y)
        #     return (x, y, features)
        # else:
        x, y = self.temporal_block(x, audio)
        x = self.video_norm2(x)
        y = self.audio_norm(y)
        return (x, y)
