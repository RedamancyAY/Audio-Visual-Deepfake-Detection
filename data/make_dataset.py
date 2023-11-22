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

import albumentations.augmentations.transforms as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video

from .dataset import DF_TIMIT, DFDC, FakeAVCeleb

from .utils.preprocessing import Read_audio, Read_video
from .utils.tools import DeepFake_Dataset


# + tags=["active-ipynb", "style-activity"]
# from dataset import DF_TIMIT, DFDC, FakeAVCeleb
# from utils.tools import DeepFake_Dataset
# -

# # Help function

# ## 提取视频帧、秒

def df_video_to_3sec(data):
    """
    每个视频抽N帧，组合成一个新的dataframe

    Args:
        data: 一个dataframe，长度就是数据集的长度，path指定视频的路径
        video_n_frames: 每个视频的长度
    Return:
        一个新的dataframe，frame_id列指定视频的哪一帧
    """
    data['n_sec'] = data.apply(lambda x: x['n_frames'] // x['fps'], axis=1)
    max_sec = data['n_sec'].max()
    
    datas = []
    for i in range(max_sec//3):
        _data = data.copy()
        _data["start_sec"] = i * 3
        _data['end_sec'] = i * 3 + 3
        datas.append(_data)
    datas = pd.concat(datas, ignore_index=True)
    datas = datas.query('end_sec <= n_sec ')
    print("Extract 3 sec for each video: ", len(data), ' -> ', len(datas))
    return datas.reset_index(drop=True)


def df_video_to_frame(data, video_n_frames):
    """
    每个视频抽N帧，组合成一个新的dataframe

    Args:
        data: 一个dataframe，长度就是数据集的长度，path指定视频的路径
        video_n_frames: 每个视频的长度
    Return:
        一个新的dataframe，frame_id列指定视频的哪一帧
    """
    datas = []
    for i in range(video_n_frames):
        _data = data.copy()
        _data["frame_id"] = i
        datas.append(_data)
    return pd.concat(datas, ignore_index=True)


def df_video_to_sec(data, max_sec=3):
    """
    每个视频抽N秒，组合成一个新的dataframe

    Args:
        data: 一个dataframe，长度就是数据集的长度，path指定视频的路径
        video_n_frames: 每个视频的长度
    Return:
        一个新的dataframe，frame_id列指定视频的哪一帧
    """
    datas = []
    for i in range(max_sec):
        _data = data.copy()
        _data["sec_id"] = i + 1
        datas.append(_data)
    _data = pd.concat(datas, ignore_index=True)
    print(len(_data))
    # _data['sec'] = _data.apply(lambda x: x['n_frames'] // x['fps'], axis=1)
    # _data = _data[_data['sec'] > _data['sec_id']]
    _data = _data[_data["n_frames"] >= 75]
    print("原始dataframe的长度为: ", len(data), ", 抽秒之后: ", len(_data))
    return _data


# ## panda dataframe to dataloader

def data2dataloader(data_splits, cfg, cfg_aug, custom_collect_fn=None, sec3=False):
    """
    convert dataframes of `train, val, test` into dataloaders

    Args:
        datasets: [train, val, test] or [train, test]
        cfg: total config

    Return:
        a dict for dataloaders
    """
    datasets = []
    data_splits = vars(data_splits)
    for item in ["train", "val", "test", "test1", "test2"]:
        if item in data_splits.keys():
            datasets.append(data_splits[item])
    # if "val" in data_splits:
    #     datasets = [data_splits.train, data_splits.val, data_splits.test]
    # else:
    #     datasets = [data_splits.train, data_splits.test]

    batch_size = [cfg.batch_size] + [cfg.test_batch_size] * (len(datasets) - 1)
    res = []
    for i, (dataset, _batch_size) in enumerate(zip(datasets, batch_size)):
        
        if sec3:
            dataset = df_video_to_3sec(dataset)
        
        
        if cfg.train_on_frame and i == 0:
            dataset = df_video_to_frame(dataset, cfg.video_n_frames)

        if cfg.train_on_sec:
            dataset = df_video_to_sec(dataset, max_sec=3)

        res.append(
            DataLoader(
                DeepFake_Dataset(
                    dataset,
                    cfg,
                    cfg_aug=cfg_aug if i == 0 else None,
                    train=(i == 0),
                    custom_collect_fn=custom_collect_fn,
                ),
                batch_size=_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                shuffle=True if i == 0 else False,
                prefetch_factor=2,
                collate_fn=None,
            )
        )
    return res


def my_collate_fn(batch):
    # your batching code here
    data = {}
    data["video"] = torch.stack([b[0]["video"] for b in batch])
    data["audio"] = torch.stack([b[0]["audio"] for b in batch])
    data["video_path"] = [b[0]["video_path"] for b in batch]
    label = {}
    label["label"] = torch.stack([torch.tensor(b[1]["label"]) for b in batch])
    label["video_label"] = torch.stack(
        [torch.tensor(b[1]["video_label"]) for b in batch]
    )
    label["audio_label"] = torch.stack(
        [torch.tensor(b[1]["audio_label"]) for b in batch]
    )
    # for b in batch:
    # print(b[0]['video'].shape)
    return data, label


# # Datasets

# ## FakeAVCeleb

def get_FakeAVCeleb(cfg, cfg_aug, custom_collect_fn=None):
    dataset = FakeAVCeleb(
        root_path=cfg.FakeAVCeleb.root_path, data_path=cfg.FakeAVCeleb.data_path
    )

    if cfg.FakeAVCeleb.quality != 0:
        print(
            "Compress video using H264 with quantiztion rate %d"
            % cfg.FakeAVCeleb.quality
        )
        dataset.data["video_path"] == dataset.data["video_path"].apply(
            lambda x: x.replace("/video/", "/video%d/" % cfg.FakeAVCeleb.quality)
        )

    if cfg.FakeAVCeleb.method is None:
        data_splits = dataset.get_splits(
            train_num=cfg.FakeAVCeleb.train_num,
            append_train_num=cfg.FakeAVCeleb.append_train_num,
            splits=cfg.FakeAVCeleb.splits,
            person_splits=False,
        )
    else:
        data_splits = dataset.get_splits_by_method(
            train_num=cfg.FakeAVCeleb.train_num,
            append_train_num=cfg.FakeAVCeleb.append_train_num,
            splits=cfg.FakeAVCeleb.splits,
            method=cfg.FakeAVCeleb.method,
        )
    return data2dataloader(
        data_splits, cfg, cfg_aug, custom_collect_fn=custom_collect_fn, sec3=False
    )


# ## DF TIMIT

def get_DF_TIMIT(cfg, cfg_aug, custom_collect_fn=None):
    data_splits = DF_TIMIT(
        root_path=cfg.DF_TIMIT.root_path,
        data_path=cfg.DF_TIMIT.data_path,
    ).get_splits(
        splits=cfg.DF_TIMIT.splits,
        video_quality=cfg.DF_TIMIT.video_quality,
    )
    return data2dataloader(
        data_splits, cfg, cfg_aug, custom_collect_fn=custom_collect_fn
    )


# ## DFDC

def get_DFDC(cfg, cfg_aug, custom_collect_fn=None):
    data_splits = DFDC(
        root_path=cfg.DFDC.root_path,
        data_path=cfg.DFDC.data_path,
        face_detect_method=cfg.face_detect_method,
    ).get_splits(
        splits=cfg.DFDC.train_splits,
    )
    return data2dataloader(
        data_splits, cfg, cfg_aug, custom_collect_fn=custom_collect_fn
    )


# # Main Function

def make_dataset(cfg, cfg_aug, custom_collect_fn=None):
    # read_func, aug_func = build_read_aug_func(cfg)

    if "FakeAVCeleb" in cfg.name:
        return get_FakeAVCeleb(cfg, cfg_aug, custom_collect_fn=custom_collect_fn)
    elif cfg.name == "DF_TIMIT":
        return get_DF_TIMIT(cfg, cfg_aug, custom_collect_fn=custom_collect_fn)
    elif cfg.name == "DFDC":
        return get_DFDC(cfg, cfg_aug, custom_collect_fn=custom_collect_fn)
