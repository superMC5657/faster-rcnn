import torch
from torch import nn
from torch.nn import functional as F

from faster_rcnn.model.rpn.proposal_creator import ProposalCreator
from faster_rcnn.model.utils.bbox_tool import generate_anchor_base, enumerate_shift_anchor
from faster_rcnn.model.utils.normalize_tool import normal_init


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
        anchor = enumerate_shift_anchor(self.anchor_base, self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        features = F.relu(self.conv1(x))
        rpn_locs = self.loc(features)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().reshape(n, -1, 4)

        rpn_scores = self.score(features)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)

        rpn_fg_scores = rpn_softmax_scores[..., 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()

        for i in range(n):
            # 需要分离 roi不需要梯度下降
            roi = self.proposal_layer(rpn_locs[i].detach(), rpn_fg_scores[i].detach(), anchor,
                                      img_size, scale)

            rois.append(roi)

        rois = torch.stack(rois, dim=0)

        return rpn_locs, rpn_scores, rois, anchor



if __name__ == '__main__':
    pass
