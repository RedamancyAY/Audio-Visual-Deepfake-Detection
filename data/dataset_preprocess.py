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
import multiprocessing as mp
import os

import torch
import torchaudio
import argparse
# from pandarallel import pandarallel
import torchvision
from torchvision.io import read_image, read_video, write_jpeg, write_png
from utils import check_dir
# -

from dataset.DF_TIMIT import DF_TIMIT
from dataset.fakeAVCeleb import FakeAVCeleb
from dataset.DFDC import DFDC


# +
def extract_frames(video, video_path):
    for i in range(video.shape[0]):
        file_path = os.path.join(video_path, "%04d.png" % (i + 1))
        check_dir(file_path)
        write_png(video[i], file_path)


def resample_and_store_audio(audio, audio_path, old_freq, new_freq=16000):
    new_audio = torchaudio.functional.resample(audio, old_freq, new_freq)
    check_dir(audio_path)
    torchaudio.save(audio_path, new_audio, 16000)


# +
def preprocess(data):
    path = data["path"]
    video_path = data["video_path"]
    audio_path = data["audio_path"]
    n_frames = data["n_frames"]
    # print(path)

#     # 1. save video frames
#     if os.path.exists(video_path):
#         file_paths = [x for x in os.listdir(video_path) if x.endswith("png")]
#         if len(file_paths) != n_frames:
#             video, audio, metadata = read_video(path, output_format="TCHW")
#             extract_frames(video, video_path)
#     else:
#         video, audio, metadata = read_video(path, output_format="TCHW")
#         extract_frames(video, video_path)

#     # 2. save resampled audio
#     if not os.path.exists(audio_path):
#         if "audio_fps" in metadata.keys():
#             resample_and_store_audio(
#                 audio, audio_path, old_freq=metadata["audio_fps"], new_freq=16000
#             )
#         else:
#             ext = os.path.splitext(path)
#             org_audio_path = path.replace(ext[-1], ".wav")
#             audio, old_frep = torchaudio.load(org_audio_path)
#             resample_and_store_audio(audio, audio_path, old_frep, new_freq=16000)

            
    
    # 1. save video frames
    if not os.path.exists(video_path):
        try:
            video, audio, metadata = read_video(path, output_format="TCHW")
            extract_frames(video, video_path)
        except OSError:
            print(path)

    # 2. save resampled audio
    if not os.path.exists(audio_path):
        ext = os.path.splitext(path)
        org_audio_path = path.replace(ext[-1], ".wav")
        audio, old_frep = torchaudio.load(org_audio_path)
        resample_and_store_audio(audio, audio_path, old_frep, new_freq=16000)
            
    return 1


# -

def process_df_chunk(chunk, process_id):
    for i in range(len(chunk)):
        preprocess(chunk.iloc[i])
        if i > 0 and i % 10 == 0:
            print("Process id {}: now {}, total {}".format(process_id, i, len(chunk)))
    return 1


# +
def strat_preprocessing_dataset(dataset):

    num_chunks = 8
    num_per_chunk = len(dataset.data) // num_chunks + num_chunks
    dataset.data = dataset.data.sample(frac=1).reset_index(drop=True)
    chunks = [
        dataset.data[i * num_per_chunk : min((i + 1) * num_per_chunk, len(dataset.data))]
        for i in range(num_chunks)
    ]

    for chunk in chunks:
        print(len(chunk))

    pool = mp.Pool(processes=num_chunks)
    for  i, chunk in enumerate(chunks):
        pool.apply_async(process_df_chunk, args=(chunk, i))
    pool.close()
    pool.join()
    
    
#     results = [
#         pool.apply_async(process_df_chunk, args=(chunk, i))
#         for i, chunk in enumerate(chunks)
#     ]

#     results[0].get()
    return 1
# -

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FakeAVCeleb")
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    assert args.dataset in ['FakeAVCeleb', 'DF-TIMIT', 'DFDC']
    
    
    if args.dataset == "DF-TIMIT":
        dataset = DF_TIMIT(root_path=args.root_path, data_path=args.data_path)
    elif args.dataset == 'FakeAVCeleb':
        dataset = FakeAVCeleb(root_path=args.root_path, data_path=args.data_path)
    elif args.dataset == 'DFDC':
        dataset = DFDC(root_path=args.root_path, data_path=args.data_path)
    
    strat_preprocessing_dataset(dataset)
