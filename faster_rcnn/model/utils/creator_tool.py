# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午2:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: creator_tool.py

import numpy as np
import torch
from torchvision.ops import nms, box_iou

from experiments.config import opt
from .bbox_tool import bbox2loc, loc2bbox


class ProposalTargetCreator:

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
                 loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def __call__(self, roi, bbox, label):
        n_bbox, _ = bbox.shape
        roi = torch.cat((roi, bbox), dim=0)  # trick 一定有个bbox 适配 方便之后cls
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = box_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1

        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size()))

        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]

        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image

        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(self.loc_normalize_mean, np.float32) /
                       np.array(self.loc_normalize_std, np.float32)))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator:
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._creat_label(inside_index, anchor, bbox)

        loc = bbox2loc(anchor, bbox[argmax_ious])
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        return loc, label

    def _creat_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1

        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)

        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = box_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


class ProposalCreator:
    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000,
                 n_test_post_nms=300, min_size=16):
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
        # roi  dst_bbox
        roi = loc2bbox(anchor, loc)

        roi[:, 0] = torch.clamp(roi[:, 0], 0, img_size[0])
        roi[:, 2] = torch.clamp(roi[:, 2], 0, img_size[0])
        roi[:, 1] = torch.clamp(roi[:, 1], 0, img_size[1])
        roi[:, 3] = torch.clamp(roi[:, 3], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        # 过滤出界的
        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]

        roi = roi[keep, :]

        score = score[keep]
        # 过滤,只选取score高的
        order = score.argsort(descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        score = score[order]

        keep = nms(roi, score, self.nms_thresh)
        # 最后输出的时候 最大只选取n_post_nms
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


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
    inside_index = np.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= H) & (anchor[:, 3] <= W))[0]
    return inside_index
