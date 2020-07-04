# -*- coding: utf-8 -*-
# !@time: 2020-07-03 23:07
# !@author: superMC @email: 18758266469@163.com
# !@fileName: rcnn.py
from .net_structure._faster_rcnn import baseFasterRCNN
from .rpn


class FasterRCNN(baseFasterRCNN):
    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        extractor,classifier =
        rpn = RegionProposalNetwork(5)
