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
from asteroid_filterbanks import Encoder, ParamSincFB
from einops import rearrange
from einops.layers.torch import Rearrange

# + tags=["style-activity"]
from .block import Block
from .style import fuse_audio_video_with_p, fuse_audio_video_with_shuffle, style_aug
from .utils import weight_init, LayerNorm, PreEmphasis

# + tags=["active-ipynb"]
# from block import Block
# from style import style_aug
# from utils import fuse_audio_video_with_p, fuse_audio_video_with_shuffle, weight_init, LayerNorm, PreEmphasis
# -

import torchvision
class Normalize(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format="NCHW"):
        self.t = torchvision.transforms.Normalize(mean=mean, std=std)
        self.data_format = data_format

    def __call__(self, x, *kargs, **kwargs):
        if self.data_format.endswith("CHW"):
            return self.t(x)
        if self.data_format == "NCTHW":
            x = torch.transpose(x, 1, 2)
            x = self.t(x)
            x = torch.transpose(x, 1, 2)
            return x
        raise ValueError(self.data_format, "wrong data format")


class Stem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self, output_channel) -> None:
        super().__init__(
            nn.Conv3d(
                3,
                output_channel,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(output_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Stage(nn.Module):
    def __init__(
        self,
        stage,
        video_dim_in,
        video_dim_out,
        n_frames,
        audio_dim_in,
        audio_dim_out,
        depth,
        mlp_ratio=4,
        audio_length=None,
        video_size=None,
        attn_dropout=0.0,
    ):
        super().__init__()

        if stage == 0:
            self.video_downsample = Stem(video_dim_out)
        else:
            self.video_downsample = nn.Sequential(
                nn.Conv3d(
                    video_dim_in,
                    video_dim_out,
                    kernel_size=(1, 7, 7) if stage == 0 else (1, 3, 3),
                    stride=(1, 4, 4) if stage == 0 else (1, 2, 2),
                    padding=(0, 3, 3) if stage == 0 else (0, 1, 1),
                ),
                nn.BatchNorm3d(video_dim_out),
            )

        self.audio_downsample = nn.Sequential(
            nn.Conv1d(
                audio_dim_in,
                audio_dim_out,
                kernel_size=7 if stage == 0 else 3,
                stride=4 if stage == 0 else 2,
                padding=3 if stage == 0 else 1,
            ),
            nn.BatchNorm1d(audio_dim_out),
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    video_dim=video_dim_out,
                    n_frames=n_frames,
                    audio_dim=audio_dim_out,
                    window_size=7,
                    mlp_ratio=mlp_ratio,
                    audio_length=audio_length,
                    video_size=video_size,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        # self.video_norm = LayerNorm(video_dim_out)
        # self.audio_norm = LayerNorm(audio_dim_out)
        self.video_norm = nn.BatchNorm3d(video_dim_out)
        self.audio_norm = nn.BatchNorm1d(audio_dim_out)
        self.apply(weight_init)

    def forward(self, video, audio):
        x = self.video_downsample(video)
        y = self.audio_downsample(audio)
        x, y = self.blocks((x, y))
        x = self.video_norm(x)
        y = self.audio_norm(y)
        return x, y


# ## Model

class Conv_Model(nn.Module):
    def __init__(
        self,
        img_size=224,  # video settings
        in_chans=3,
        n_frames=10,
        audio_freq=16000,  # audio settings
        audio_length=48000,
        num_classes=2,  # model
        out_channels=[32, 64, 128, 256],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 4, 6, 3],
        stream_head=True,
        cfg=None,
        attn_dropout=0.0,
        *kargs,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        audio_out_channels, video_out_channels = (
            out_channels.copy(),
            out_channels.copy(),
        )
        video_out_channels.insert(0, in_chans)
        audio_out_channels.insert(0, 1 if not cfg.preprocess_audio else 256)
        self.stages = nn.ModuleList(
            [
                Stage(
                    stage=i,
                    video_dim_in=video_out_channels[i],
                    video_dim_out=video_out_channels[i + 1],
                    n_frames=n_frames,
                    audio_dim_in=audio_out_channels[i],
                    audio_dim_out=audio_out_channels[i + 1],
                    depth=depths[i],
                    mlp_ratio=4,
                    audio_length=audio_length // (2 ** (i + 2)),
                    video_size=img_size // (2 ** (i + 2)),
                    attn_dropout=attn_dropout,
                )
                for i in range(len(depths))
            ]
        )

        # classificaiton head
        final_chanels = video_out_channels[-1]
        self.video_head = nn.Linear(final_chanels, num_classes, bias=False)
        self.audio_head = nn.Linear(final_chanels, num_classes, bias=False)
        self.head = nn.Linear(final_chanels * 2, num_classes, bias=False)

        if cfg.normalize:
            self.normalizer = Normalize(
                data_format="NCTHW", std=cfg.normalize_std, mean=cfg.normalize_mean
            )

        self.flag_preprocess_audio = False
        if cfg.preprocess_audio:
            self.flag_preprocess_audio = True
            self.audio_preprocess = nn.Sequential(
                PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
            )
            self.time_freq_repr = Encoder(
                ParamSincFB(n_filters=256, kernel_size=251, stride=1), padding=125
            )

        K = out_channels[-1]
        self.f_video = nn.Sequential(
            nn.Conv3d(K, K, 3, padding=1, groups=K),
            nn.AdaptiveAvgPool3d(1),
            Rearrange("b c t h w -> b (c t h w)"),
        )
        self.f_audio = nn.Sequential(
            nn.Conv1d(K, K, 31, padding=15, groups=K),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b c l -> b (c l)"),
        )
        self.style_aug = cfg.style_aug
        if self.style_aug:
            self.f_video_adv = nn.Sequential(
                nn.Conv3d(K, K, 3, padding=1, groups=K),
                nn.AdaptiveAvgPool3d(1),
                Rearrange("b c t h w -> b (c t h w)"),
            )
            self.f_audio_adv = nn.Sequential(
                nn.Conv1d(K, K, 31, padding=15, groups=K),
                nn.AdaptiveAvgPool1d(1),
                Rearrange("b c l -> b (c l)"),
            )
            self.video_head_adv = nn.Linear(video_out_channels[-1], num_classes)
            self.audio_head_adv = nn.Linear(audio_out_channels[-1], num_classes)
            self.head_adv = nn.Linear(video_out_channels[-1], num_classes)

        # Final, init all weights
        self.apply(weight_init)

    def preprocess_audio(self, x):
        x = self.audio_preprocess(x)
        x = torch.abs(self.time_freq_repr(x))
        x = torch.log(x + 1e-6)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x

    def adv_params(self):
        params = []
        for layer in self.adv_modules():
            for m in layer.modules():
                params += [p for p in m.parameters()]
        return params

    def style_params(self):
        params = []
        for m in [
            self.f_video_adv,
            self.f_audio_adv,
            self.video_head_adv,
            self.audio_head_adv,
        ]:
            params += [p for p in m.parameters()]
        return params

    def adv_modules(self):
        if self.flag_preprocess_audio:
            return [self.stages, self.audio_preprocess, self.time_freq_repr]
        else:
            return [self.stages]

    def style_modules(self):
        if self.style_aug:
            return [
                self.f_video_adv,
                self.f_audio_adv,
                self.video_head_adv,
                self.audio_head_adv,
            ]
        else:
            return []

    def forward(self, video, audio, cls_token=False, train=False, grad_cam=False):
        B, C, T, H, W = video.shape
        B, C, L = audio.shape

        x, y = video, audio
        res = {}

        if hasattr(self, "normalizer"):
            x = self.normalizer(x)
        if self.flag_preprocess_audio:
            y = self.preprocess_audio(y)

        for i, stage in enumerate(self.stages):
            x, y = stage(x, y)
            # print(x.shape)
            if grad_cam:
                res["org_x_%d" % i] = x

        if self.style_aug and train:
            x_style, x_adv = style_aug(x, dims=[-3, -2, -1])
            y_style, y_adv = style_aug(y, dims=[-1])
            res["x"] = self.f_video(x_style)
            res["y"] = self.f_audio(y_style)

            x_adv = self.f_video_adv(x_adv.detach())
            y_adv = self.f_audio_adv(y_adv.detach())
            res["video_label_adv"] = self.video_head_adv(x_adv)
            res["audio_label_adv"] = self.audio_head_adv(y_adv)
        else:
            res["x"] = self.f_video(x)
            res["y"] = self.f_audio(y)

        res["video_label"] = self.video_head(res["x"])
        res["audio_label"] = self.audio_head(res["y"])
        res["z"] = torch.concat([res["x"], res["y"]], dim=-1)

        if self.cfg.shuffle_av and train:
            res["z_adv"], res["shuffle_ids"] = fuse_audio_video_with_shuffle(
                res["x"], res["y"]
            )
            res["total_label_adv"] = self.head(res["z_adv"])

        res["total_label"] = self.head(res["z"])
        return res
