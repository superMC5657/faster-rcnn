# -*- coding: utf-8 -*-
# !@time: 2020-07-03 23:07
# !@author: superMC @email: 18758266469@163.com
# !@fileName: FasterRCNN.py

from faster_rcnn.model.backbone.resnet import resnet
from faster_rcnn.model.backbone.vgg import vgg
from faster_rcnn.model.baseNets.fasterRCNN.baseFasterRCNN import BaseFasterRCNN
from faster_rcnn.model.roi.roiHead import RoIHead
from faster_rcnn.model.rpn.region_proposal_network import RegionProposalNetwork

backbone = {'vgg': vgg, 'resnet': resnet}


class FasterRCNN(BaseFasterRCNN):
    def __init__(self, model_name, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], pretrained=True):
        self.feat_stride = 16
        model_name = model_name.split('-')
        backbone_name = model_name[0]
        backbone_num = int(model_name[1])
        extractor, classifier = backbone[backbone_name](backbone_num, pretrained)
        rpn = RegionProposalNetwork(512, 512, ratios, anchor_scales, feat_stride=self.feat_stride)
        head = RoIHead(n_class=n_fg_class + 1, roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=classifier)

        super(FasterRCNN, self).__init__(extractor, rpn, head)
