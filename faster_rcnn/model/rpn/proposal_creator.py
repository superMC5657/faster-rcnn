# -*- coding: utf-8 -*-
# !@time: 2020/7/10 下午5:56
# !@author: superMC @email: 18758266469@163.com
# !@fileName: proposal_creator.py
import torch
from torchvision.ops import nms

from faster_rcnn.model.utils.bbox_tool import loc2bbox


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

    def __call__(self, loc, score, anchor, img_size):
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

        min_size = self.min_size
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
