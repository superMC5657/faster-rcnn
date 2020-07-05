# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午2:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: creator_tool.py

import numpy as np
import torch
from torchvision.ops import nms, box_iou
from .bbox_tools import bbox2loc, loc2bbox


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
        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = box_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))


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


def _get_inside_index(anchor, H, W):
    index_inside = np.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= H) & (anchor[:, 3] <= W))[0]
    return index_inside


class ProposalCreator:
    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000,
                 n_test_post_nms=300, min_size=300):
        self.min_size = min_size
        self.n_test_post_nms = n_test_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
