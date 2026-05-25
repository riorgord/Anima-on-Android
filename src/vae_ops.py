import torch
import torch.nn as nn
import torch.nn.functional as F


class Ops:
    Linear = nn.Linear
    Conv2d = nn.Conv2d
    Conv3d = nn.Conv3d
    GroupNorm = nn.GroupNorm


def pytorch_attention(q, k, v):
    B, C, H, W = q.shape
    q = q.view(B, 1, C, -1).transpose(2, 3).contiguous()
    k = k.view(B, 1, C, -1).transpose(2, 3).contiguous()
    v = v.view(B, 1, C, -1).transpose(2, 3).contiguous()
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(2, 3).reshape(B, C, H, W)


def vae_attention():
    return pytorch_attention
