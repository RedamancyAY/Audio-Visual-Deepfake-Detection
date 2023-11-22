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

# +
import random

import torch
import torch.nn as nn


# -

def get_style(t, dims):
    mu = torch.mean(t, dim=dims, keepdim=True)
    diff = torch.square(t - mu)
    diff_mean = torch.mean(diff, dim=dims, keepdim=True)
    sigma = torch.sqrt(diff_mean + 0.000001)
    return mu, sigma


def change_style(z1, z2, dims=[-1], alpha=-1):
    mu1, sigma1 = get_style(z1, dims=dims)
    mu2, sigma2 = get_style(z2, dims=dims)

    if alpha == -1:
        alpha = random.random()
        
    mu_hat = alpha * mu1 + (1.0 - alpha) * mu2
    sigma_hat = alpha * sigma1 + (1.0 - alpha) * sigma2
    z_prime = sigma_hat * ((z1 - mu1) / sigma1) + mu_hat
    return z_prime


def style_aug(z, dims=[-1], ids=None):
    if ids is None:
        ids = list(range(len(z)))
        random.shuffle(ids)
    z_styled = change_style(z, z[ids, ...], dims=dims)
    z_adv = change_style(z[ids, ...], z, dims=dims, alpha=0)
    return z_styled, z_adv


def style_tensors(*tensors, dims=[-1]):
    ids = list(range(len(tensors[0])))
    random.shuffle(ids)
    res = []
    for tensor in tensors:
        res.append(style_aug(tensor, dims=dims, ids=ids))
    return res


# + tags=["active-ipynb"]
# f_video, f_audio = torch.rand(16, 3, 1024), torch.rand(16, 3, 1024)
#
# style_aug(f_video)
#
# (f_a_style, f_a_adv), (f_v_style, f_v_adv) = style_tensors(f_audio, f_video)

# + tags=["active-ipynb"]
# for para in nn.Conv3d(256, 256, 3, padding=1, groups=256).parameters():
#     print(para.shape)
# -

def fuse_audio_video_with_p(f_video, f_audio):
    B = f_video.shape[0]
    ids = random.sample(list(range(B)), B)
    shuffle_ids = ids[: B // 2]
    shuffle_ids2 = ids[B // 2 :]
    res = torch.concat([f_video, f_audio], dim=-1)
    res[shuffle_ids, :] = torch.concat(
        [f_video[shuffle_ids, :], f_audio[shuffle_ids2, :]], dim=-1
    )
    return res, shuffle_ids


# + tags=["active-ipynb"]
# fuse_aduio_video_with_p(f_video, f_audio)
# -

def fuse_audio_video_with_shuffle(f_video, f_audio):
    B = f_video.shape[0]
    ids = random.sample(list(range(B)), B)
    shuffle_ids = [x for i, x in enumerate(ids) if x != i]
    res = torch.concat([f_video, f_audio[ids, :]], dim=-1)
    return res, shuffle_ids
