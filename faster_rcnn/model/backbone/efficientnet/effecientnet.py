# -*- coding: utf-8 -*-
# !@time: 2020/7/3 下午2:08
# !@author: superMC @email: 18758266469@163.com
# !@fileName: effecientnet.py

import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from faster_rcnn.model.rcnn.faster_rcnn import _fasterRCNN
from faster_rcnn.model.utils.config import cfg
from faster_rcnn.model.backbone.efficientnet.model import EfficientNet


class Effecientnet(_fasterRCNN):
    def __init__(self, classes, efficient_name, pretrained=False, class_agnostic=False):
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.efficient_name = efficient_name

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        if self.pretrained:
            efficientnet = EfficientNet.from_pretrained(self.efficient_name)
        else:
            efficientnet = EfficientNet.from_name(self.efficient_name)
            # Build resnet.
        self.RCNN_base = nn.Sequential(efficientnet.conv1, efficientnet.bn1, efficientnet.relu,
                                       efficientnet.maxpool, efficientnet.layer1, efficientnet.layer2,
                                       efficientnet.layer3)

        self.RCNN_top = nn.Sequential(efficientnet.layer4)

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_base[0].parameters(): p.requires_grad = False
        for p in self.RCNN_base[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
