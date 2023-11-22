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
import os

import numpy as np
import python_speech_features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from torchvision.io import decode_png, encode_png, read_image, read_video


# -

# # Read Video

class Read_video(nn.Module):
    def __init__(
        self, n_frames, img_size=(224, 224), face_detect_method=None, **kwargs
    ):
        super().__init__()
        self.n_frames = n_frames
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.resize = T.Resize(img_size)
        self.face_detect_method = face_detect_method

    def read_on_sec(self, video_path, sec_id, video_fps=30):
        frames_sep = video_fps / self.n_frames

        frame_ids = [
            int((sec_id - 1) * video_fps + i * frames_sep) for i in range(self.n_frames)
        ]
        # print(frame_ids, sec_id, video_fps)

        x = torch.stack(
            [self.read_frame_of_video(video_path, frame_id) for frame_id in frame_ids]
        )

        if x.shape[0] < self.n_frames:
            T, C, H, W = x.shape
            x = torch.concat(
                [x, torch.zeros((self.n_frames - T, C, H, W), dtype=torch.float32)],
                dim=0,
            )
        x = self.resize(x)
        return x.contiguous()

    def read_frame_of_video(self, video_path, frame_id):
        return read_image(os.path.join(video_path, "%04d.png" % (frame_id + 1)))

    def forward(
        self,
        video_path,
        frame_id=-1,
        sec_id=-1,
        video_fps=25,
        video_total_frames=75,
        start_sec=0,
        end_sec=3,
    ):
        # read frame
        if frame_id != -1:
            frames_sep = min(video_total_frames, video_fps * 3) / self.n_frames
            frame_ids = [int(i * frames_sep) for i in range(self.n_frames)]
            image = self.read_frame_of_video(
                video_path, frame_ids[frame_id] + start_sec * video_fps
            )
            return self.resize(image)

        # print(video_fps, video_total_frames, sec_id)
        # read second
        if sec_id != -1:
            return self.read_on_sec(video_path, sec_id + start_sec, video_fps=video_fps)

        # read 10 frames from the first 3 seconds
        if video_total_frames < self.n_frames:
            x = torch.stack(
                [
                    self.read_frame_of_video(video_path, i)
                    for i in range(video_total_frames)
                ]
            )
            T, C, H, W = x.shape
            x = torch.concat(
                [x, torch.zeros(self.n_frames - video_total_frames, C, H, W)], dim=0
            )
        else:
            video_total_frames = min(video_total_frames, video_fps * 3)
            frames_sep = video_total_frames / self.n_frames
            x = torch.stack(
                [
                    self.read_frame_of_video(
                        video_path, int(i * frames_sep) + start_sec * video_fps
                    )
                    for i in range(self.n_frames)
                ],
                dim=0,
            )
            # print([int(i * frames_sep) for i in range(self.n_frames)])

        x = self.resize(x)
        return x


# + tags=["active-ipynb", "style-solution"]
# reader = Read_video(n_frames=10, img_size=224)
#
# video_path = "/usr/local/ay_data/dataset/Celeb-DF-v2/Celeb-real/id0_0003.mp4"
# reader(video_path)
# -

# # Read Audio

def get_mfcc(_audio, _sr):
    mfcc = zip(*python_speech_features.mfcc(_audio, _sr, nfft=2048))
    mfcc = np.stack([np.array(i) for i in mfcc])
    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    # print(cc.shape)
    return torch.tensor(cc, dtype=torch.float32)


class Read_audio(nn.Module):
    def __init__(self, freq, length, features=None):
        super().__init__()
        self.length = int(length)
        self.freq = freq
        self.features = features

    def read_audio(self, audio_path):
        x, sample_rate = torchaudio.load(audio_path)
        if x.shape[0] > 1:
            x = x[0:1, :]
        if sample_rate != self.freq:
            x = torchaudio.functional.resample(x, sample_rate, self.freq)
        length = x.size(1)
        if length >= self.length:
            return x[:, : self.length]
        else:
            return torch.concatenate([x, torch.zeros(1, self.length - length)], dim=1)

    def read_waveform(self, audio_path, sec_id=-1):
        x = self.read_audio(audio_path)
        if sec_id != -1:
            x = x[:, self.freq * (sec_id - 1) : self.freq * sec_id]
        return x

    def read_features(self, audio_path, sec_id=-1):
        x, sr = torchaudio.backend.sox_io_backend.load(audio_path, normalize=False)
        if x.shape[0] > 1:
            x = x[0:1, :]
        if sr != self.freq:
            x = torchaudio.functional.resample(x, sr, self.freq)
        if sec_id != -1:
            x = x[:, self.freq * (sec_id - 1) : self.freq * sec_id]
        length = x.size(1)
        if length >= self.freq:
            x = x[:, : self.freq]
        else:
            x = torch.concatenate([x, torch.zeros(1, self.freq - length)], dim=1)

        return get_mfcc(x[0].numpy(), self.freq)

    def forward(
        self,
        audio_path,
        sec_id=-1,
        start_sec=0,
        end_sec=3,
    ):
        if self.features is None:
            return self.read_waveform(audio_path, sec_id=sec_id)
        else:
            return self.read_features(audio_path, sec_id=sec_id)

# + tags=["style-solution", "active-ipynb"]
# reader = Read_audio(freq=16000, length=48000, features="mfcc")
# video_path = "/home/ay/data/DATA/dataset/0-deepfake/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/women/id00359/00053_id04376_wavtolip.mp4"
# reader(video_path)
