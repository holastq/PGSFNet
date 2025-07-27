import math
from abc import ABC
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.common.mixed_attn_block import (
    AnchorProjection,
    CAB,
    CPB_MLP,
    QKVProjection,
)
from basicsr.archs.common.ops import (
    window_partition,
    window_reverse,
)
from basicsr.archs.common.swin_v1_block import Mlp
from timm.models.layers import DropPath

class AffineTransform(nn.Module):
    r"""Affine transformation of the attention map.
    The window could be a square window or a stripe window. Supports attention between different window sizes
    """

    def __init__(self, num_heads):
        super(AffineTransform, self).__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        B_, H, N1, N2 = attn.shape
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H, N1, N2) + mask
            attn = attn.view(-1, H, N1, N2)

        return attn


def _get_stripe_info(stripe_size_in, stripe_groups_in, stripe_shift, input_resolution):
    stripe_size, shift_size = [], []
    for s, g, d in zip(stripe_size_in, stripe_groups_in, input_resolution):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)

        else:
            stripe_size.append(d // g)
            shift_size.append(0 if g == 1 else d // (g * 2))

    return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, attn_transform, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        # if self.euclidean_dist is not None:
            # print("use euclidean distance")
            # attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        # else:
            # attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class MultiHeadAttention_query(Attention):
    def __init__(
        self,
        in_dim,
        window_size=4,
        num_heads=4,
        attn_drop=0.0,
        args=None,
    ):

        super(MultiHeadAttention_query, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        # self.euclidean_dist = args.euclidean_dist

    def forward(self, q, k, v, H, W , mask, table=None, index=None):

        L, B, C = k.shape # L B C 
        Q, _, _ = q.shape # Q B C 
        # print(q.shape)#200 1 96
        q = q.view(B, Q, C)
        k = k.view(B, L, C).view(B, H, W, C)
        v = v.view(B, L, C).view(B, H, W, C)
        # print(q.shape)

        # partition windows
        k = window_partition(k, self.window_size)  # nW*B, wh, ww, C
        v = window_partition(v, self.window_size)  # nW*B, wh, ww, C

        k = k.view(-1, self.window_size*self.window_size, C)  # nW*B, wh*ww, C
        v = v.view(-1, self.window_size*self.window_size, C)  # nW*B, wh*ww, C

        B_, N_, _ = k.shape
        # q = q.reshape(B, Q, self.num_heads, -1).repeat(B_ // B, 1, 1, 1).permute(0, 2, 1, 3)# cost too much
        q = q.reshape(B, Q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, N_, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, N_, self.num_heads, -1).permute(0, 2, 1, 3) #b- numheads n c/n  # nW*B, Head, wh*ww, dim
        k = torch.mean(k, dim=0).view(B, self.num_heads, N_, -1)
        v = torch.mean(v, dim=0).view(B, self.num_heads, N_, -1)


        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)
        x = x.view(Q, B, C)

        return x

    def flops(self, N):
        pass

class MultiHeadAttention_window(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        in_dim,
        window_size=4,
        num_heads=4,
        attn_drop=0.0,
        args=None,
    ):

        super(MultiHeadAttention_window, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        # self.euclidean_dist = args.euclidean_dist

    def forward(self, q, k, v, H, W , mask, table=None, index=None):

        L, B, C = q.shape # L B C 
        q = q.view(B, L, C)
        # print(q.shape)
        q = q.view(B, H, W, C)
        k = k.view(B, L, C).view(B, H, W, C)
        v = v.view(B, L, C).view(B, H, W, C)
        # print(q.shape)

        # partition windows
        q = window_partition(q, self.window_size)  # nW*B, wh, ww, C
        k = window_partition(k, self.window_size)  # nW*B, wh, ww, C
        v = window_partition(v, self.window_size)  # nW*B, wh, ww, C
        q = q.view(-1, self.window_size*self.window_size, C)  # nW*B, wh*ww, C
        k = k.view(-1, self.window_size*self.window_size, C)  # nW*B, wh*ww, C
        v = v.view(-1, self.window_size*self.window_size, C)  # nW*B, wh*ww, C

        B_, N, _ = q.shape
        q = q.reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3) #b- numheads n c/n  # nW*B, Head, wh*ww, dim

        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)

        # merge windows
        x = x.view(-1, self.window_size*self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # B, H, W, C

        x = x.view(L, B, C)

        return x

    def flops(self, N):
        pass

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # print(window_size)
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    # H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x