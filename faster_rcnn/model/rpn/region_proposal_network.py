import numpy as np
import torch

from torch.nn import functional as F
from torch import nn

from faster_rcnn.model.utils.normalize_tool import normal_init
from faster_rcnn.model.utils.bbox_tool import generate_anchor_base
from faster_rcnn.model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_creator_params=None, anchor_base_size=16):
        super(RegionProposalNetwork, self).__init__()
        if proposal_creator_params is None:
            proposal_creator_params = dict()
        self.feat_stride = feat_stride
        self.anchor_base = generate_anchor_base(base_size=anchor_base_size, anchor_scales=anchor_scales, ratios=ratios)
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shift_anchor(torch.tensor(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        features = F.relu(self.conv1(x))
        rpn_locs = self.loc(features)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(features)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)

        rpn_fg_scores = rpn_softmax_scores[..., 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor,
                                      img_size, scale)

            bacth_index = i * torch.ones((len(roi),), dtype=torch.int32)
            rois.append(roi)
            roi_indices.append(bacth_index)
        rois = torch.stack(rois, dim=0)
        roi_indices = torch.stack(roi_indices, dim=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shift_anchor(anchor_base, feat_stride, height, width):
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    shift = torch.stack((shift_y.flatten(), shift_x.flatten(), shift_y.flatten(), shift_x.flatten()), dim=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).permute((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).type(torch.float32)
    return anchor


if __name__ == '__main__':
    pass
