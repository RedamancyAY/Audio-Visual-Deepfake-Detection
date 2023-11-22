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

import cv2
import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.nn.functional as F
import torchaudio
from moviepy.editor import VideoFileClip
from pandarallel import pandarallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# + tags=["style-activity"]
from .utils import generate_paths, get_video_metadata, split_datasets


# + tags=["style-activity", "active-ipynb"]
# from utils import generate_paths, get_video_metadata, split_datasets
# -

def path_to_name(path):
    if "VoxCeleb2" in path:
        name = path.split("VoxCeleb2/")[1]
    else:
        name = "-".join(path.split("/")[-5:])
    return name


def path_to_person(path):
    if "VoxCeleb2" in path:
        name = path.split("VoxCeleb2/")[1].split("-")[0]
    else:
        name = path.split("/")[-2]
    return name


def custom_splits(total_data, splits):
    def _help(data):
        all_items = sorted(list(set(list(data["person"]))))
        L = len(all_items)
        train = int(L * splits[0])
        val = int(L * splits[1])
        test = L - train - val
        res = {}
        res["train"] = data[data["person"].isin(all_items[0:train])].reset_index(
            drop=True
        )
        res["val"] = data[
            data["person"].isin(all_items[train : train + val])
        ].reset_index(drop=True)
        res["test"] = data[data["person"].isin(all_items[train + val :])].reset_index(
            drop=True
        )
        for x in res.keys():
            print(len(res[x]))
        return argparse.Namespace(**res)

    res_org = _help(total_data.query("VoxCeleb2==0"))
    res_add = _help(total_data.query("VoxCeleb2==1"))
    return argparse.Namespace(
        train=pd.concat([res_org.train, res_add.train]),
        val=pd.concat([res_org.val, res_add.val]),
        test=pd.concat([res_org.test, res_add.test]),
    )
    return argparse.Namespace(**res)


class FakeAVCeleb:
    """deal with the dataset FakeAVCeleb"""

    def __init__(self, root_path: str, data_path):
        if root_path.endswith(os.sep):
            root_path = root_path[:-1]
        self.root_path = root_path
        self.data_path = data_path

        self.path_dataset_info = os.path.join(root_path, "dataset_info.csv")
        self.data = self.read_dataset_info()

        assert len(self.data) > 0, "There is no video in %s" % root_path
        assert len(self.data) == 21544 + 2000
        self.data = generate_paths(self.data, data_path=data_path)

    def read_dataset_info(self):
        if not os.path.exists(self.path_dataset_info):
            data = self.init_dataset_info()
        else:
            data = pd.read_csv(self.path_dataset_info)
        return data

    def init_dataset_info(self) -> pd.DataFrame:
        paths = []
        for path, dir_list, file_list in os.walk(self.root_path):
            for file_name in file_list:
                if file_name.endswith("mp4"):
                    paths.append(os.path.join(path, file_name))
        print(len(paths))
        data = pd.DataFrame(sorted(paths), columns=["path"])
        data["video_label"] = data["path"].apply(lambda x: 0 if "FakeVideo" in x else 1)
        data["audio_label"] = data["path"].apply(lambda x: 0 if "FakeAudio" in x else 1)
        data["label"] = data["path"].apply(
            lambda x: 1 if ("RealVideo-RealAudio" in x or "VoxCeleb2" in x) else 0
        )
        ## 3. get video info
        print("read video info from all videos:")
        data = get_video_metadata(data)

        data["VoxCeleb2"] = data["path"].apply(lambda x: 1 if "VoxCeleb2" in x else 0)
        data["name"] = data["path"].apply(path_to_name)
        assert len(set(list(dataset.data["name"]))) == len(data)

        data["person"] = data["path"].apply(path_to_person)
        meta_data = pd.read_excel(dataset.root_path + "/meta_data.xlsx")
        meta_data["path2"] = meta_data.apply(
            lambda x: x["path2"] + "/" + x["path"], axis=1
        )
        meta_data["name"] = meta_data["path2"].apply(path_to_name)
        meta_data = meta_data[["name", "method"]]
        meta_data = meta_data.drop_duplicates(["name"])

        data = pd.merge(data, meta_data, on="name", how="left")
        data["method"] = data["method"].fillna("real")

        data.to_csv(self.path_dataset_info, index=False)
        return data

    def get_splits(
        self,
        train_num: list,
        append_train_num=0,
        splits=[0.75, 0.1, 0.15],
        person_splits=False,
        method=None,
    ) -> list:
        """split the train and test datasets for developping a deep model

        args:
            train_num: the numbers of four types (`self.AV_types`) videos in train set
            val_num: the numbers of four types (`self.AV_types`) videos in validation set
            test_num: the numbers of four types (`self.AV_types`) videos in test set
            append_train_num: the extra number of RealVideo-RealAudio videos from VoxCeleb2
        """
        data = self.data
        assert len(train_num) == 4

        if method is not None:
            data = data[
                data["method"].isin([method, method + "-wav2lip", "rtvc", "real"])
            ]

        lens = [
            len(
                data.query(
                    "VoxCeleb2 == 0 & video_label == {} & audio_label == {}".format(
                        a, b
                    )
                )
            )
            for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]

        data_selected = pd.concat(
            [
                data.query(
                    "VoxCeleb2 == 0 & video_label == {} & audio_label == {}".format(
                        a, b
                    )
                ).sample(min(lens[i], train_num[i]), random_state=42)
                for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])
            ]
        )

        if append_train_num > 0:
            data_added = data.query("VoxCeleb2 == 1").sample(
                append_train_num, random_state=42
            )
            data_selected = pd.concat([data_selected, data_added])

        # print(len(data), lens, train_num, method, len(data_selected))
        
        
        if person_splits:
            return custom_splits(data_selected, splits=splits)
        else:
            return split_datasets(data_selected, splits=splits)

    def get_splits_by_method(
        self,
        train_num: list,
        append_train_num=0,
        splits=[0.75, 0.1, 0.15],
        method="fsgan",
    ) -> list:
        """split the train and test datasets for developping a deep model

        args:
            train_num: the numbers of four types (`self.AV_types`) videos in train set
            val_num: the numbers of four types (`self.AV_types`) videos in validation set
            test_num: the numbers of four types (`self.AV_types`) videos in test set
            append_train_num: the extra number of RealVideo-RealAudio videos from VoxCeleb2
        """
        data = self.data
        assert len(train_num) == 4

        assert method in ["fsgan", "wav2lip", "faceswap"]
        data_fsgan, data_wav2lip, data_faceswap = [
            self.get_splits(
                train_num=train_num,
                append_train_num=append_train_num,
                splits=splits,
                method=_method,
            )
            for _method in ["fsgan", "wav2lip", "faceswap"]
        ]
        # for item in [data_fsgan, data_wav2lip, data_faceswap]:
            # print(len(item.train), len(item.val), len(item.test))
        

        if method == "fsgan":
            return argparse.Namespace(
                train=data_fsgan.train,
                val=data_fsgan.val,
                test1=data_wav2lip.test,
                test2=data_faceswap.test,
            )
        elif method == "wav2lip":
            return argparse.Namespace(
                train=data_wav2lip.train,
                val=data_wav2lip.val,
                test1=data_fsgan.test,
                test2=data_faceswap.test,
            )
        else:
            return argparse.Namespace(
                train=data_faceswap.train,
                val=data_faceswap.val,
                test1=data_wav2lip.test,
                test2=data_fsgan.test,
            )
