# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms


# + tags=[]
def rotate_videos(video, angles=[10, -10]):
    B = video.shape[0]
    res = []
    _video = rearrange(video, "b c t h w -> (b t) c h w")
    for _ang in angles:
        res.append(
            rearrange(transforms.functional.rotate(
                _video, _ang, interpolation=transforms.InterpolationMode.BILINEAR
            ), '(b t) c h w -> b c t h w', b=B)
        )
    return res


# + tags=[]
class Test_time_aug:
    def __init__(self, transform="ten_crop"):

        self.transform = transform

        self.Crop_Flip = transforms.TenCrop(size=200)
        self.Resize = nn.Sequential(
            Rearrange("b c t h w -> (b c) t h w"),
            transforms.Resize((224, 224)),
            Rearrange("(b c) t h w -> b c t h w", c=3),
        )

    def aggregate(self, res):
        final_res = {}
        for key in res[0].keys():
            tmp = torch.stack([_item[key] for _item in res], dim=0)
            final_res[key] = torch.mean(tmp, dim=0)
        return final_res

    def __call__(self, model, video, audio):
        res = []
        res.append(model(video, audio))

        if self.transform == "ten_crop":
            crop_flips = self.Crop_Flip(video)
            for _video in crop_flips:
                _video = self.Resize(_video)
                res.append(model(_video, audio))
        elif self.transform == 'flip':
            res.append(model(transforms.functional.hflip(video), audio))
        elif self.transform == 'flip+rotate':
            for i in range(2):
                for j in range(2):
                    if i == 0:
                        _video = video
                    else:
                        _video = transforms.functional.hflip(video)
                    if j == 0:
                        res.append(model(_video, audio))
                    else:
                        r_videos = rotate_videos(_video)
                        for _video in r_videos:
                            res.append(model(_video, audio))
                    
        return self.aggregate(res)

# + tags=["active-ipynb"]
# def model(video, audio):
#     res = {}
#     res["test"] = torch.rand(2, 2)
#     return res
#
#
# tta = Test_time_aug(transform="flip+rotate")
# video = torch.rand(2, 3, 10, 224, 224)
# audio = torch.rand(2, 1, 48000)
# tta(model, video, audio)
