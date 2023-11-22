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
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy.editor import VideoFileClip
from pandarallel import pandarallel
from torchvision.io import read_video


# +
# # + tags=[]
def has_audio_streams(file_path):
    command = ["ffprobe", "-show_streams", "-print_format", "json", file_path]
    output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
    # print(output)
    parsed = json.loads(output)
    streams = parsed["streams"]
    audio_streams = list(filter((lambda x: x["codec_type"] == "audio"), streams))
    return len(audio_streams) > 0


# # + tags=[]
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # exist_audio = int(cap.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS)) > 0

    exist_audio = has_audio_streams(path)
    # exist_audio = has_audio_track(path)

    _info = {
        "height": frame_height,
        "width": frame_width,
        "fps": FPS,
        "n_frames": frame_count,
        "exist_audio": exist_audio,
    }
    _info = argparse.Namespace(**_info)
    return _info


# -

def get_video_metadata(data):
    def _video_info(path, items):
        info = get_video_info(path)
        info = vars(info)
        return [info[x] for x in items]

    pandarallel.initialize(progress_bar=True)
    video_infos = ["height", "width", "fps", "n_frames", "exist_audio"]
    data[video_infos] = data.parallel_apply(
        lambda x: tuple(_video_info(x["path"], video_infos)),
        axis=1,
        result_type="expand",
    )
    return data


def split_datasets(df, splits: list = [0.75, 0.1, 0.15]):
    """Split the video paths according to the offical splits file"""
    train = df.sample(frac=splits[0], random_state=42)
    if len(splits) == 3:
        val = df.drop(train.index).sample(
            frac=(splits[1] / (splits[1] + splits[2])), random_state=42
        )
        test = df.drop(train.index).drop(val.index)
    else:
        test = df.drop(train.index)
    res = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    return argparse.Namespace(**res)


def generate_paths(data, data_path, face_detect_method=None):
    audio_path = data_path + "/audio"
    mouth_path = data_path + "/mouth"

    # print(data_path, face_detect_method)

    if face_detect_method is None:
        video_path = data_path + "/video"
        data["video_path"] = data["name"].apply(
            lambda x: os.path.join(video_path, os.path.splitext(x)[0])
        )
    else:
        video_path = data_path + "/%s" % face_detect_method
        print(video_path)
        data["video_path"] = data["name"].apply(
            lambda x: os.path.join(video_path, os.path.splitext(x)[0])
        )
    data["audio_path"] = data["name"].apply(
        lambda x: os.path.join(audio_path, os.path.splitext(x)[0] + ".wav")
    )
    data["mouth_path"] = data["name"].apply(
        lambda x: os.path.join(mouth_path, os.path.splitext(x)[0])
    )
    return data
