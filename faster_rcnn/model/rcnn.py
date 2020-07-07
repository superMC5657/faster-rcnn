# -*- coding: utf-8 -*-
# !@time: 2020-07-03 23:07
# !@author: superMC @email: 18758266469@163.com
# !@fileName: rcnn.py
import torch
from torch import nn
from torchvision.ops.roi_pool import RoIPool

from experiments.config import opt
from faster_rcnn.model.backbone.resnet import resnet
from faster_rcnn.model.backbone.vgg import vgg
from faster_rcnn.model.net_structure.base_faster_rcnn import baseFasterRCNN
from faster_rcnn.model.rpn.region_proposal_network import RegionProposalNetwork
from faster_rcnn.model.utils.normalize_tool import normal_init

backbone = {'vgg': vgg, 'resnet': resnet}


class FasterRCNN(baseFasterRCNN):
    def __init__(self, model_name, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], pretrained=True):
        self.feat_stride = 16
        model_name = model_name.split('-')
        backbone_name = model_name[0]
        backbone_num = int(model_name[1])
        extractor, classifier = backbone[backbone_name](backbone_num, pretrained)
        rpn = RegionProposalNetwork(512, 512, ratios, anchor_scales, feat_stride=self.feat_stride)
        head = RoIHead(n_class=n_fg_class + 1, roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=classifier)

        super(FasterRCNN, self).__init__(extractor, rpn, head)


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
        roi_indices = torch.zeros(rois.shape[0]).to(opt.device)

        indices_and_rois = torch.cat((roi_indices[:, None], rois), dim=1)

        # yx->xy
        indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]].contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1).contiguous()
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
