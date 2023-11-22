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
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# + tags=["style-activity"]
from .utils import Multi_Head_Attention, weight_init, LayerNorm


# + tags=["active-ipynb"]
# from utils import Multi_Head_Attention, weight_init, LayerNorm
# -

class TemporalBlock(nn.Module):
    def __init__(
        self,
        video_dim,
        audio_dim,
        window_size=7,
        n_frames=10,
        audio_length=None,
        video_size=None,
        drop_path=0.1,
        attn_dropout=0.,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
        self.min_dim = 1

        ## deal video
        self.conv11 = nn.Sequential(
            nn.Conv3d(
                video_dim,
                video_dim,
                groups=video_dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(video_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                video_dim, self.min_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
        )
        self.conv12 = nn.Conv3d(
            self.min_dim, video_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.window_size = window_size
        self.n_frames = n_frames
        self.video_dim = video_dim

        ## deal audio
        self.conv21 = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(audio_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(audio_dim, self.min_dim, kernel_size=33, stride=1, padding=16),
        )
        self.conv22 = nn.Conv1d(
            self.min_dim, audio_dim, kernel_size=33, stride=1, padding=16
        )
        ## self-attention
        num_embeddings = (video_size**2) // (window_size**2) * n_frames + n_frames
        # print('num_embeddings is ', num_embeddings, video_size, window_size)
        self.attention = Multi_Head_Attention(
            num_embeddings=num_embeddings,
            embed_dim=window_size**2 * self.min_dim,
            num_heads=self.min_dim,
            QKV=False,
            projection=True,
            dropout=attn_dropout
        )

        ## PS. layer scaling
        alpha_0 = 1e-2
        self.alpha_1 = nn.Parameter(
            alpha_0 * torch.ones((video_dim)), requires_grad=True
        )
        self.alpha_2 = nn.Parameter(
            alpha_0 * torch.ones((audio_dim)), requires_grad=True
        )
        self.apply(weight_init)

    def forward(self, video, audio, grad_cam=False):
        ## 1. deal video
        x = self.conv11(video)
        x = rearrange(
            x,
            "b c t (p1 h) (p2 w) -> b (t p1 p2) (c h w)",
            h=self.window_size,
            w=self.window_size,
        )
        # print("downsmaple x", x.shape)

        ## 2. deal audio
        y = self.conv21(audio)
        y = rearrange(
            y,
            "b c (n l) -> b (n c) l",
            n=self.n_frames,
            l=audio.shape[-1] // self.n_frames,
        )
        audio_length = y.shape[-1]
        y = F.adaptive_avg_pool1d(y, 49)
        y = rearrange(y, "b (n c) l -> b n (c l)", c=self.min_dim)
        # print("downsmaple y", y.shape)

        ## 3. self-attention
        # print(x.shape, y.shape)
        z = torch.concat([x, y], dim=1)
        z = z + self.attention(z)

        ## 4. recover video and audio
        x, y = z[:, : x.shape[1], :], z[:, x.shape[1] :, :]
        x = rearrange(
            x,
            "b (t p1 p2) (c h w) -> b c t (h p1) (w p2)",
            t=self.n_frames,
            c=self.min_dim,
            p1=video.shape[-1] // self.window_size,
            p2=video.shape[-1] // self.window_size,
            h=self.window_size,
            w=self.window_size,
        )
        x = self.drop_path(self.alpha_1.view(-1, 1, 1, 1) * self.conv12(x)) + video
        # print(x.shape)

        y = rearrange(y, "b n (c l) -> b (n c) l", c=self.min_dim)
        y = F.interpolate(y, size=audio_length)
        y = rearrange(y, "b (n c) l -> b c (n l)", c=self.min_dim)
        y = self.drop_path(self.alpha_2.view(-1, 1) * self.conv22(y)) + audio
        # print(y.shape)
        return x, y

# + tags=["active-ipynb", "style-student"]
# block = TemporalBlock(
#     video_dim=32, audio_dim=32, window_size=7, audio_length=12000, video_size=56
# )
# video = torch.Tensor(np.random.rand(2, 32, 10, 56, 56))
# audio = torch.Tensor(np.random.rand(2, 32, 12000))
# x, y = block(video, audio)
# print(x.shape, y.shape)
#
# print(sum(p.numel() for p in block.attention.parameters() if p.requires_grad))
# for p in block.attention.parameters():
#     print(p.shape)

# + tags=["active-ipynb", "style-activity"]
# audio = np.random.rand(2, 10, 1200)
# y = torch.Tensor(audio)
#
# y = nn.Conv1d(10, 1, kernel_size=33, stride=3, padding=1)(y)
# print(y.shape)
#
# audio = np.random.rand(2, 10, 49)
# y = torch.Tensor(audio)
# y = nn.ConvTranspose1d(
#     10, 10, kernel_size=33, stride=15, dilation=13, output_padding=11
# )(y)
# print(y.shape)
