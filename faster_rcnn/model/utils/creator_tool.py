# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午2:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: creator_tool.py

import numpy as np
import torch
from torchvision.ops import nms
from .bbox_tools import bbox2loc, bbox_iou, loc2bbox


class ProposalTargetCreator():

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def __call__(self, roi, bbox, label):
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox))


def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret




