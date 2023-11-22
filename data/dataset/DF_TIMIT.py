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
import shutil
import sys
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# + tags=["style-activity"]
from .utils import generate_paths, get_video_metadata, split_datasets


# + tags=["style-activity", "active-ipynb"]
# from utils import generate_paths, get_video_metadata, split_datasets
# -

# # 文件夹

def path_to_name(path):
    """
    DeepfakeTIMIT/higher_quality/fadg0/sa1-video-fram1.avi -> fadg0-sa1-video-fram1.avi
    """
    a, b = os.path.split(path)
    quality = a.split("/")[-2]
    x = "HQ" if "higher" in quality else ("LQ" if "lower" in quality else "RAW")
    return x + "-" + a.split("/")[-1] + "-" + b


def custom_splits(data, splits):
    all_items = sorted(list(set(list(data["person"]))))
    L = len(all_items)
    train = int(L * splits[0])
    val = int(L * splits[1])
    test = L - train - val
    res = {}
    res["train"] = data[data["person"].isin(all_items[0:train])].reset_index(drop=True)
    res["val"] = data[data["person"].isin(all_items[train : train + val])].reset_index(
        drop=True
    )
    res["test"] = data[data["person"].isin(all_items[train + val :])].reset_index(
        drop=True
    )
    return argparse.Namespace(**res)


class DF_TIMIT:
    def __init__(self, root_path, data_path):
        if root_path.endswith(os.sep):
            root_path = root_path[:-1]
        self.root_path = root_path
        self.data_path = data_path
        self.path_dataset_info = os.path.join(root_path, "dataset_info.csv")
        self.n_videos = 960
        self.data = generate_paths(self.read_dataset_info(), data_path=data_path)

    def read_dataset_info(self):
        if not os.path.isfile(self.path_dataset_info):
            data = self.init_dataset_info()
        else:
            data = pd.read_csv(self.path_dataset_info)
        return data

    def init_dataset_info(self):
        print("Strat generate the metadata of the DF-TIMIT dataset")

        paths = []
        used_videos = []
        for path, dir_list, file_list in os.walk(self.root_path):
            for file_name in file_list:
                if file_name.endswith("avi"):
                    paths.append(os.path.join(path, file_name))
                    if "quality" in path:
                        used_videos.append(path.split("/")[-1])
        used_videos = list(set(used_videos))
        paths = [x for x in paths if os.path.split(x)[0].split("/")[-1] in used_videos]

        assert (
            len(paths) == 960
        ), "The number of videos in DF-TIMIT should be 960, but is actually %d" % len(
            paths
        )

        ## 2. build a DataFrame from videos paths, extract labels and quality
        data = pd.DataFrame(sorted(paths), columns=["path"])
        data["video_label"] = data["path"].apply(lambda x: 0 if "quality" in x else 1)
        data["audio_label"] = 1
        data["label"] = data["video_label"]
        data["video_quality"] = data["path"].apply(
            lambda x: "HQ" if "higher" in x else ("LQ" if "lower" in x else "RAW")
        )
        data["name"] = data["path"].apply(path_to_name)

        data["person"] = data["path"].apply(lambda x: x.split("/")[-2])

        ## 3. get video info
        print("read video info from all videos:")
        data = get_video_metadata(data)

        data.to_csv(self.path_dataset_info, index=False)
        return data

    def get_splits(
        self,
        video_quality="LQ",
        splits=[0.75, 0.1, 0.15],
    ):
        """Split the video paths and generate Dataloader"""
        assert video_quality in ["LQ", "HQ"]
        data = self.data[self.data["video_quality"].isin([video_quality, "RAW"])]
        # data_splits = split_datasets(data, splits)
        data_splits = custom_splits(data, splits)
        return data_splits

# + tags=["style-activity", "active-ipynb"]
# dataset = DF_TIMIT(
#     root_path="/home/ay/data/0-原始数据集/DeepfakeTIMIT",
#     data_path="/home/ay/data/DATA/dataset/0-deepfake/DF-TIMIT",
# )
#
# dataset.get_splits()
