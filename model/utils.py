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

import torch.nn as nn
from einops import rearrange

# # import packages

# +
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn

# -

# # Method

# ## 参数初始化方法

def weight_init(m):
    # if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
    # nn.init.xavier_normal_(m.weight, gain=math.sqrt(2.0))
    # nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.Conv3d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ## Multi_Head_Attention

def _multi_head_attention(q, k, v, heads=1, dropout=None):
    q, k, v = map(
        lambda mat: rearrange(mat, "b n (h d) -> (b h) n d", h=heads), (q, k, v)
    )
    scale = q.shape[-1] ** -0.5
    qkT = einsum("b n d, b m d->b n m", q, k) * scale
    attention = dropout(qkT.softmax(dim=-1))
    attention = einsum("b n m, b m d->b n d", attention, v)
    attention = rearrange(attention, "(b h) n d -> b n (h d)", h=heads)
    return attention


class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embed_dim,
        num_heads=1,
        QKV=False,
        projection=False,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = LayerNorm(num_embeddings)
        self.PE = PositionEmbedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim
        )
        self.num_heads = num_heads
        self.QKV = QKV
        self.projection = projection
        if QKV:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        if projection:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False), nn.Dropout(dropout)
            )
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.PE(x)
        x = self.norm(x)
        if self.QKV:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        else:
            q, k, v = x, x, x
        x = _multi_head_attention(q, k, v, heads=self.num_heads, dropout=self.dropout)
        if self.projection:
            x = self.proj(x)
        return x


import torch.nn as nn
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = self._norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        elif x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self._norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        elif x.ndim == 3:
            x = rearrange(x, "b c l -> b l c")
            x = self._norm(x)
            x = rearrange(x, "b l c -> b c l")
        return x
class PositionEmbedding(nn.Module):

    MODE_EXPAND = "MODE_EXPAND"
    MODE_ADD = "MODE_ADD"
    MODE_CONCAT = "MODE_CONCAT"

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(
                torch.Tensor(num_embeddings * 2 + 1, embedding_dim)
            )
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()
        # print("PositionEmbedding, weight shape is ", self.weight.shape)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = (
                torch.clamp(x, -self.num_embeddings, self.num_embeddings)
                + self.num_embeddings
            )
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        # print(x.shape, seq_len, self.num_embeddings, self.embedding_dim)
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError("Unknown mode: %s" % self.mode)

    def extra_repr(self):
        return "num_embeddings={}, embedding_dim={}, mode={}".format(
            self.num_embeddings,
            self.embedding_dim,
            self.mode,
        )


# +
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.ndim in [2, 3]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # reflect padding to match lengths of in/out
        x = F.pad(x, (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter)
