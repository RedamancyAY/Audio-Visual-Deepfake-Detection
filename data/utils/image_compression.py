
import albumentations.augmentations.functional as F
from enum import IntEnum
import torch
import torchvision.io as io
import random

import torch
import torchvision.io as io

import numpy as np
import torch



MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
    torch.uint8: 255,
    torch.float32: 1.0
}


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.to(torch.float32) / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).to(dtype)

def jpg_compression(img, quality):
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == torch.float32:
        img = from_float(img, dtype=torch.uint8)
        needs_float = True
    elif input_dtype not in (torch.uint8, torch.float32):
        raise ValueError("Unexpected dtype {} for jpg_compression".format(input_dtype))

    img = io.encode_jpeg(img, quality=quality)
    img = io.decode_jpeg(img)

    if needs_float:
        img = to_float(img, max_value=255)
    return img



# https://github.com/albumentations-team/albumentations/blob/9b0525f479509195a7a7b7c19311d8e63bbc6494/albumentations/augmentations/transforms.py#L219

class JPEGCompression(object):
    """Decreases image quality by Jpeg compression of an image.
    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg. 
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        quality_lower=60,
        quality_upper=100,
        consistent_quality=True,
        p=0.5,
    ):
        super().__init__()

        assert 0 <= quality_lower <= quality_upper <= 100
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.consistent_quality = consistent_quality
        self.p = p
        self.debug = 0
        
    def compress_img(self, x, quality=-1):
        assert x.ndim == 3 and x.shape[0] in [1, 3]
        if quality == -1:
            quality = random.randint(self.quality_lower, self.quality_upper)
        y = jpg_compression(x, quality)
        # print(quality, torch.sum(y-x))
        return y
    
    def __call__(self, x, **kwargs):
        '''
        Args:
            x: (T, C, H, W) or (C, H, W)
        '''
        # if not self.debug:
        #     print("Use jpeg compression augmentation")
        #     self.debug = True
            
        if random.random() > self.p:
            return x
        
        assert x.ndim in [3, 4]
        if self.consistent_quality:
            quality = random.randint(self.quality_lower, self.quality_upper)
        else:
            quality = -1
            
        if x.ndim == 3:
            return self.compress_img(x, quality)
        else:
            return torch.stack([self.compress_img(x[i], quality) for i in range(x.shape[0])])
