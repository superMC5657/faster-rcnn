# -*- coding: utf-8 -*-
# !@time: 2020-07-03 22:15
# !@author: superMC @email: 18758266469@163.com
# !@fileName: _faster_rcnn.py
from torch import nn


class baseFasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(baseFasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.2, 0.2, 0.2)
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        image_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, image_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7

        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05

        else:
            raise ValueError
