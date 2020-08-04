# -*- coding: utf-8 -*-
# !@time: 2020/7/10 下午5:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: other_tool.py
import torch

from experiments.config import opt


def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = torch.zeros((count,), dtype=data.dtype).cuda() + fill
        ret[index] = data
    else:
        ret = torch.zeros((count,) + data.shape[1:], dtype=data.dtype).cuda() + fill
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    inside_index = torch.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= H) & (anchor[:, 3] <= W))[0]
    return inside_index
