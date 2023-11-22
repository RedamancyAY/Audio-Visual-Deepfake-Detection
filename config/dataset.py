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

from yacs.config import CfgNode as ConfigurationNode

# # 默认配置

ALL_DATASETS = ["FakeAVCeleb", "DF_TIMIT", "DFDC"]


def DF_TIMIT():
    C = ConfigurationNode()
    C.root_path = "...../DeepfakeTIMIT"
    C.data_path = "...../DF-TIMIT"
    C.video_quality = 'LQ' # LQ ○r HQ
    C.splits = [0.75, 0.1, 0.15] # LQ ○r HQ
    return C


def FakeAVCeleb():
    __C = ConfigurationNode()
    __C.root_path = "...../FakeAVCeleb_v1.2"
    __C.data_path = "...../FakeAVCeleb+"
    __C.train_num=[1500, 1500, 500, 500]
    __C.append_train_num=2000
    __C.splits = [0.75, 0.1, 0.15]
    __C.quality = 0 # 0 for raw, 23 for c23, 40 for c40
    __C.method = None  # ["fsgan", "wav2lip", "faceswap"]
    return __C


def DFDC():
    __C = ConfigurationNode()
    __C.root_path = "...../dfdc"
    __C.data_path = "...../DFDC"
    __C.train_splits = [0.75, 0.1, 0.15]
    __C.subset_splits = [4500, 4500, 9000]
    return __C


def get_dataset_cfg():
    __C = ConfigurationNode()
    __C.video_n_frames = 10
    __C.video_img_size = 224
    __C.video_in_chans = 3
    __C.face_detect_method = None
    __C.use_audio = True
    __C.audio_freq = 16000
    __C.audio_length = 48000
    __C.audio_features = None


    __C.name = 'DFDC'
    __C.batch_size = 32
    __C.test_batch_size = 1
    __C.num_workers = 10
    __C.train_on_frame = False
    __C.train_on_sec = False
    __C.train_on_mouth = False
    
    __C.DF_TIMIT = DF_TIMIT()
    __C.FakeAVCeleb = FakeAVCeleb()
    __C.DFDC = DFDC()
    return __C


