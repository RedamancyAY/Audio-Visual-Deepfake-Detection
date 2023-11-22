

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from typing import Union


class BatchGaussianBlur(nn.Module):
    '''
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    '''
    def __init__(self, kernel_size: int, sigma=(0.1, 2), p=0.5, cuda=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma if not isinstance(sigma, numbers.Number) else (sigma, sigma)
        self.cuda = cuda
        self.p = p
        
    def _get_batch_gaussian_kernel2d(self, kernel_size: int, sigma=Union[float, list]):
        """generate multiple 2d gaussian kernel

        Args:
            kernel_size: the kernel size of the 2d gaussian kernel
            sigma: one or multiple sigma

        Returns:
            N 2d gaussian kernels, (N, kernel_size, kernel_size), where N is the
                length of sigma.
        """
        if isinstance(sigma, numbers.Number):
            sigma = [sigma]
        sigma = torch.Tensor(sigma)
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x[None, ...] / sigma[..., None]).pow(2))
        kernel1d = pdf / pdf.sum(dim=1)[:, None]
        kernel2d = torch.matmul(kernel1d[..., None], kernel1d[:, None, :])
        return kernel2d
    

    def gene_kernels(self, N):
        sigma = torch.empty(N).uniform_(self.sigma[0], self.sigma[1])
        kernels = self._get_batch_gaussian_kernel2d(self.kernel_size, sigma)
        # print("kernels size is ", kernels.shape)
        return kernels
    
    def gaussian_conv(self, x):
        B, C, H, W = x.shape
        if B < 1:
            return x
        assert B > 0, x.shape
        kernels = self.gene_kernels(B).unsqueeze(1)
        if self.cuda:
            kernels = kernels.cuda()
        x = x.transpose(0, 1)
        x = F.conv2d(x, kernels, padding="same", groups=B)
        x = x.transpose(0, 1)
        return x
    
    def conv_4D(self, x, label=None):
        B, C, H, W = x.shape
        p = torch.rand(B)
        if label is None:
            index1 = torch.where(p <= self.p)
        else:
            index1 = torch.where((p <= self.p) & (label == 1))
        index2 = torch.where(p > self.p)
        x_gaussian = self.gaussian_conv(x[index1])
        # x = torch.concat([x_gaussian, x[index2]], dim=0)
        x[index1] = x_gaussian
        return torch.clip(x.contiguous(), min=0., max=1.)
    
    def forward(self, x, label=None):
        if label is not None and label.is_cuda:
            label = label.cpu()
            
        if x.ndim == 4:
            return self.conv_4D(x, label)
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = self.conv_4D(x, label)
            x = rearrange(x, 'b (c t) h w -> b c t h w', c=C, t=T)
            return x
        return x



# + tags=[]
import numbers
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F2
from einops import rearrange


# + tags=[]
class BatchRandomRotation(nn.Module):
    """
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    """

    def __init__(self, angles=(-10, 10), p=0.5):
        super().__init__()
        self.angles = angles
        self.p = p

    def rotate(self, x):
        B = x.shape[0]
        _angles = list(
            torch.empty(B).uniform_(self.angles[0], self.angles[1]).numpy()
        )
        x = torch.stack(
            [
                F2.rotate(
                    x[i],
                    float(_angles[i]),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                )
                for i in range(B)
            ],
            dim=0,
        )
        return x

    def forward(self, x, label=None):
        B = x.shape[0]
        p = torch.rand(B)
        index1 = torch.where(p <= self.p)
        index2 = torch.where(p > self.p)
        if len(index1[0]) == 0:
            return x
        else:
            x_rotated = self.rotate(x[index1])
            # x = torch.concat([x_rotated, x[index2]], dim=0)
            x[index1] = x_rotated
            return x.contiguous()



# + tags=[]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from typing import Union


# -

# https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py

# ```python
# def hflip(img: Tensor) -> Tensor:
#     _assert_image_tensor(img)
#
#     return img.flip(-1)
# ```

# + tags=[]
class BatchRandomHorizontalFlip(nn.Module):
    '''
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    '''
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    
    def forward(self, x, label=None):
        B = x.shape[0]
        p = torch.rand(B)
        index1 = torch.where(p <= self.p)
        if len(index1[0]) == 0:
            return x
        else:
            index2 = torch.where(p > self.p)
            x_flipped = x[index1].flip(-1)
            # x = torch.concat([x_flipped, x[index2]], dim=0)
            x[index1] = x_flipped
            return x.contiguous()
