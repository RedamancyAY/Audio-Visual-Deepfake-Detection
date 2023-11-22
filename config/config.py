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

from yacs.config import CfgNode as ConfigurationNode

# + tags=["style-activity"]
from .dataset import ALL_DATASETS, get_dataset_cfg

# + tags=["active-ipynb"]
# from dataset import ALL_DATASETS, get_dataset_cfg
# -

# # 默认配置

__C = ConfigurationNode()
__C.DATASET = get_dataset_cfg()

# +
__C.DataAug = ConfigurationNode()
# 1. sample based
__C.DataAug.jpeg_compression = True
__C.DataAug.gaussian_noise = False
__C.DataAug.color_jitter = False
__C.DataAug.gaussian_filter = False
__C.DataAug.cv_pil_compression = False

# 1.2 audio
__C.DataAug.random_align = False
__C.DataAug.random_speed = False


# batch based
__C.DataAug.flip = True
__C.DataAug.rotate = True
__C.DataAug.gaussianBlur = False
__C.DataAug.gaussianBlur_kernel = 5
__C.DataAug.gaussianBlur_p = 0.5
__C.DataAug.ISONoise = False
__C.DataAug.colorJitter = False
__C.DataAug.normalize = False
__C.DataAug.crop_resize = False

# +
__C.MODEL = ConfigurationNode()
__C.MODEL.epochs = 200
__C.MODEL.optimizer = "AdamW"
__C.MODEL.weight_decay = 0.01
__C.MODEL.lr = 0.0001
__C.MODEL.lr_decay_factor = 0.5
__C.MODEL.lr_scheduler = "linear"
__C.MODEL.warmup_epochs = 10
__C.MODEL.label_smoothing = 0.1

__C.MODEL.normalize = False
__C.MODEL.normalize_mean = [0.5, 0.5, 0.5]
__C.MODEL.normalize_std = [0.5, 0.5, 0.5]
__C.MODEL.preprocess_audio = True
__C.MODEL.out_channels = [32, 64, 128, 256]
__C.MODEL.mlp_ratios = [4, 4, 4, 4]
__C.MODEL.depths = [3, 3, 9, 3]
__C.MODEL.contrast_loss_weight = 0.1
__C.MODEL.contrast_loss_alpha = 0.4
__C.MODEL.ce_loss = "normal"
__C.MODEL.stream_head = True
__C.MODEL.CS = False
__C.MODEL.loss_total_contrast = True
__C.MODEL.attn_dropout = 0.0
__C.MODEL.style_aug = False
__C.MODEL.shuffle_av = False
__C.MODEL.adv_loss = True

__C.MODEL.pretrained = True
__C.MODEL.ensemble = False


# -

def get_cfg_defaults(cfg_file=None, ablation=''):
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    res = __C.clone()

    if cfg_file is not None:

        aug_file_path = os.path.join(os.path.split(cfg_file)[0], "data_aug.yaml")
        if os.path.exists(aug_file_path):
            res.merge_from_file(aug_file_path)
            print("load aug yaml in ", aug_file_path)
        
        model_file_path = os.path.join(os.path.split(cfg_file)[0], "0-model.yaml")
        if os.path.exists(model_file_path):
            res.merge_from_file(model_file_path)
            print("load model yaml in ", model_file_path)
        
        if ablation != '':
            ablation_path = os.path.join(os.path.split(cfg_file)[0], "%s.yaml"%ablation)
            res.merge_from_file(ablation_path)
            print("load ablation yaml in ", ablation_path)
            
        res.merge_from_file(cfg_file)
        

    for _ds in ALL_DATASETS:
        if _ds != res.DATASET.name:
            res.DATASET.pop(_ds)

    return res
