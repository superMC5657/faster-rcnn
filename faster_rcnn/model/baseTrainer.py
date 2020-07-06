# -*- coding: utf-8 -*-
# !@time: 2020/7/6 下午9:09
# !@author: superMC @email: 18758266469@163.com
# !@fileName: baseTrainer.py

import os
from collections import namedtuple
import time
from torch.nn import functional as F
from faster_rcnn.model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch
from faster_rcnn.utils import array_tool
from faster_rcnn.utils.vis_tool import Visualizer

from experiments.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTupe', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, rcnn):
        super(FasterRCNNTrainer, self).__init()
        self.rcnn = rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = rcnn.loc_normalize_mean()
        self.loc_normalize_std = rcnn.loc_normalize_std()

        self.optimizer = self.rcnn.get_optimizer()
        self.vis = Visualizer(env=opt.env)

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, imgs, bboxes, labels, scale):
        batch_size = bboxes.shape[0]
        if batch_size != 1:
            raise ValueError('Currently only batch_size of 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rcnn.rpn(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
