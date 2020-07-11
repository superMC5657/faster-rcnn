# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午2:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: AnchorTargetCreator.py

import numpy as np
import torch
from torchvision.ops import box_iou
from experiments.config import opt
from faster_rcnn.model.utils.bbox_tool import bbox2loc
from faster_rcnn.utils import array_tool as at
from faster_rcnn.model.rpn.tool import _get_inside_index, _unmap


class AnchorTargetCreator:
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def forward(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index].to(opt.device)

        argmax_ious, label = self._creat_label(inside_index, anchor, bbox)

        loc = bbox2loc(anchor, bbox[argmax_ious])
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        return loc.contiguous(), label.contiguous()

    def __call__(self, bbox, anchor, img_size):
        return self.forward(bbox, anchor, img_size)

    # bugs : disable_index 生成问题
    def _creat_label(self, inside_index, anchor, bbox):
        label = torch.zeros(inside_index.shape[0], dtype=torch.int32) - 1

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)

        label = at.tonumpy(label)

        pos_index = np.where(label == 1)[0]

        if pos_index.shape[0] > n_pos:
            # disable_index = torch.randperm(pos_index.shape[0])[:pos_index.shape[0] - n_pos]
            # label[disable_index] = -1

            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

            # label = -1 不参与训练
        # n_neg = self.n_sample - torch.sum(label == 1).item()
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]

        if neg_index.shape[0] > n_neg:
            # disable_index = torch.randperm(neg_index.shape[0])[:neg_index.shape[0] - n_neg]
            # label[disable_index] = -1
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        label = at.totensor(label)
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = box_iou(anchor, bbox)
        argmax_ious = ious.argmax(dim=1)
        max_ious = ious[torch.arange(inside_index.shape[0]), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, torch.arange(ious.shape[1])]

        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]  # [anchor,oneimagelabels] find [oneimagelables] -> []

        return argmax_ious, max_ious, gt_argmax_ious
