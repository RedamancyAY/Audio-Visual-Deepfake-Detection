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

# +
import argparse
import functools
import hashlib
import os
import pathlib
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
# -

from .preprocessing import Read_audio, Read_video

# +
from torchvision.transforms import Compose

from .image_compression import JPEGCompression


def get_video_aug_func(cfg):
    res = []
    format_info = "Image augumentaion: "
    if cfg.jpeg_compression:
        res.append(JPEGCompression(quality_lower=60, quality_upper=100, p=0.5))
        format_info += "jpeg compression, "

    if len(res) > 0:
        print(format_info)
        return Compose(res)
    return None


# +
import math
import os
import pathlib
import random

import numpy as np
import torch
import torchaudio
from audiomentations import TimeStretch as Org_TimeStretch


class RandomAlign:
    def __init__(self, p=0.5, max_length=18000, min_length=4500):
        self.p = p
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, audio_data):
        if random.random() > self.p:
            return audio_data

        L = audio_data.shape[-1]
        align_length = random.randint(self.min_length, self.max_length)
        align = torch.zeros(1, align_length)
        if random.random() > 0.5:
            audio_data = torch.concat([audio_data[:, align_length:], align], dim=-1)
        else:
            audio_data = torch.concat(
                [align, audio_data[:, 0 : L - align_length]], dim=-1
            )
        return audio_data


class TimeStretch:
    def __init__(
        self,
        min_rate=0.8,
        max_rate=1.25,
        leave_length_unchanged=True,
        p=1.0,
        sample_rate=16000,
    ):
        self.transform = Org_TimeStretch(
            min_rate=min_rate,
            max_rate=max_rate,
            leave_length_unchanged=leave_length_unchanged,
            p=p,
        )
        self.sample_rate = sample_rate

    def __call__(self, audio, sample_rate=-1):
        if sample_rate == -1:
            sample_rate = self.sample_rate

        if type(audio) == np.ndarray:
            return self.transform(audio, sample_rate=sample_rate)
        else:
            audio = self.transform(audio.numpy(), sample_rate=sample_rate)
            return torch.from_numpy(audio)


def get_audio_aug_func(cfg):

    res = []
    format_info = "Audio augumentaion: "
    if cfg.random_speed:
        res.append(
            TimeStretch(
                min_rate=0.8,
                max_rate=1.25,
                leave_length_unchanged=True,
                p=1.0,
                sample_rate=16000,
            )
        )
        format_info += "Random Speed"

    if cfg.random_align:
        res.append(RandomAlign(p=0.5, min_length=4000, max_length=8000))
        format_info += "Random align"

    if len(res) > 0:
        print(format_info)
        return Compose(res)
    return None


# -

class DeepFake_Dataset(Dataset):
    """Torch.utils.Dataset"""

    def __init__(
        self, data, cfg, cfg_aug=None, train=True, custom_collect_fn=None, **kwargs
    ):
        super().__init__()
        self.data = data
        self.use_audio = cfg.use_audio
        self.train_on_frame = cfg.train_on_frame
        self.train_on_sec = cfg.train_on_sec
        self.train_on_mouth = cfg.train_on_mouth
        self.cfg = cfg
        if self.train_on_mouth:

            def check_mouth(x):
                mouth_path = x["mouth_path"]
                n_frames = x["n_frames"]
                if not os.path.exists(mouth_path):
                    return 0
                # pngs = [x for x in os.listdir(mouth_path) if x.endswith('png')]
                # if len(pngs) == n_frames:
                #     return 1
                # else:
                #     return 0
                return 1

            self.data["mouth"] = self.data.apply(check_mouth, axis=1)
            len1 = len(self.data)
            self.data = self.data[self.data["mouth"] == 1].reset_index(drop=True)
            len2 = len(self.data)
            print("Filter the videos that does not have mouth: ", len1, " to ", len2)

        self.read_audio_func = Read_audio(
            freq=cfg.audio_freq,
            length=cfg.audio_freq * 3,
            features=cfg.audio_features,
        )
        self.read_video_func = Read_video(
            n_frames=cfg.video_n_frames,
            img_size=cfg.video_img_size,
            face_detect_method=cfg.face_detect_method,
            train_frame=cfg.train_on_frame,
            train_sec=cfg.train_on_sec,
        )
        print(len(data), train)
        self.video_aug_func = None if cfg_aug is None else get_video_aug_func(cfg_aug)
        self.audio_aug_func = None if cfg_aug is None else get_audio_aug_func(cfg_aug)

        self.train = train
        # from ay.torch.transforms.audio import RandomBackgroundNoise
        # self.noise_transform = RandomBackgroundNoise(
        #     sample_rate=16000, noise_dir="/home/ay/musan/noise"
        # )
        # print(self.video_aug_func, self.noise_transform)

        self.custom_collect_fn = custom_collect_fn

    def __len__(self):
        return len(self.data)

    def process_video(self, video):
        if self.video_aug_func is not None:
            video = self.video_aug_func(video)
        if type(video) is np.ndarray:
            # video = torch.tensor(video.copy())
            video = torch.from_numpy(video)
        if len(video.shape) == 4:
            video = video.transpose(0, 1)
        return video / 255

    def process_audio(self, audio):
        if self.audio_aug_func is not None:
            audio = self.audio_aug_func(audio)
        return audio

    def get_labels(self, item):
        labels = {"video_label": item["video_label"]}
        if "audio_label" in item.keys():
            labels["audio_label"] = item["audio_label"]
        if "label" in item.keys():
            labels["label"] = item["label"]
        return labels

    def __getitem__(self, ndx):
        item = self.data.iloc[ndx]
        labels = self.get_labels(item)

        data = {}

        frame_id = (
            self.data.iloc[ndx]["frame_id"]
            if self.train_on_frame and self.train
            else -1
        )
        sec_id = self.data.iloc[ndx]["sec_id"] if self.train_on_sec else -1

        if self.train_on_mouth:
            video_path = item["mouth_path"]
        else:
            video_path = item["video_path"]

        start_sec = item["start_sec"] if "start_sec" in item.keys() else 0
        end_sec = item["end_sec"] if "end_sec" in item.keys() else 3

        _data = self.process_video(
            self.read_video_func(
                video_path,
                frame_id=frame_id,
                sec_id=sec_id,
                video_fps=item["fps"],
                video_total_frames=item["n_frames"],
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )  # (T, C, H, W)
        if frame_id == -1:
            data["video"] = _data
        else:
            data["frame"] = _data
            # print(data['frame'].shape)

        if self.use_audio:
            data["audio"] = self.process_audio(
                self.read_audio_func(
                    item["audio_path"],
                    sec_id=sec_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
        # print(data['video'].shape, data['audio'].shape)
        data["video_path"] = video_path
        # for key in data.keys():
        # print(key, data[key].shape if type(data[key]) is not str else 0)
        # print(data['video'].dtype, data['audio'].dtype, data['video'].shape, data['audio'].shape)
        if self.custom_collect_fn is None:
            return data, labels
        else:
            return self.custom_collect_fn(
                data,
                labels,
                self.cfg.video_n_frames,
            )
