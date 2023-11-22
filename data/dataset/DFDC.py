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
import os
import shutil
from typing import Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# + tags=["style-activity"]
from .utils import generate_paths, get_video_metadata, split_datasets

# + tags=["style-activity", "active-ipynb"]
# from utils import generate_paths, get_video_metadata, split_datasets
# -

# ## DFDC

cur_path = os.path.split(os.path.abspath(__file__))[0]


class DFDC(object):
    def __init__(self, root_path, data_path, face_detect_method="s3fd"):
        if root_path.endswith(os.sep):
            root_path = root_path[:-1]
        self.root_path = root_path
        self.face_detect_method = face_detect_method
        
        data = pd.read_csv(os.path.join(cur_path, "splits/DFDC-18000.csv"))
        data['name'] = data['filename'].apply(lambda x: os.path.splitext(x)[0])
        data['path'] = data.apply(lambda x: os.path.join(root_path, 'dfdc_train_part_%s'%x['part'], x['filename']), axis=1)
        self.data = generate_paths(
            data, data_path=data_path, face_detect_method=face_detect_method
        )

    def get_splits(
        self,
        splits=[0.75, 0.1, 0.15],
    ):
        """Split the video paths and generate Dataloader"""
        data_splits = split_datasets(self.data, splits)
        return data_splits
