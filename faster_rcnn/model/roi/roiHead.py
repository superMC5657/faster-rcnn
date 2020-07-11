# -*- coding: utf-8 -*-
# !@time: 2020/7/10 下午6:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: roiHead.py
import torch
from torch import nn
from torchvision.ops import RoIPool

from experiments.config import opt
from faster_rcnn.model.utils.normalize_tool import normal_init


class RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois):
        roi_indices = torch.zeros(rois.shape[0]).to(opt.device).float()

        indices_and_rois = torch.cat((roi_indices[:, None], rois), dim=1)

        # yx->xy
        indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]].contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1).contiguous()
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
