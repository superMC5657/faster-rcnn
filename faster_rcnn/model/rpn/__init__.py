# -*- coding: utf-8 -*-
# !@time: 2020-07-03 23:21
# !@author: superMC @email: 18758266469@163.com
# !@fileName: __init__.py

from torch import nn


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor=[8, 16, 32], feat_stride=16,
                 proposal_creator_params=dict(), ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base =
