# -*- coding: utf-8 -*-
# !@time: 2020/7/6 下午9:09
# !@author: superMC @email: 18758266469@163.com
# !@fileName: baseFasterTrainer.py

import os
from collections import namedtuple
import time
from torch.nn import functional as F
from faster_rcnn.model.rpn.anchorTarget_creator import AnchorTargetCreator

from torch import nn
import torch

from faster_rcnn.model.rpn.proposalTarget_creator import ProposalTargetCreator
from faster_rcnn.utils.vis_tool import Visualizer
from faster_rcnn.utils import array_tool as at
from experiments.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])


def _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (pred_loc - gt_loc)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


class FasterRCNNTrainer(nn.Module):
    def __init__(self, rcnn):
        super().__init__()
        self.rcnn = rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator(loc_normalize_mean=rcnn.loc_normalize_mean,
                                                             loc_normalize_std=rcnn.loc_normalize_std)

        self.optimizer = self.rcnn.get_optimizer()
        self.vis = Visualizer(env=opt.env)

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, imgs, bboxes, labels, scale):
        batch_size = bboxes.shape[0]
        if batch_size != 1:
            raise ValueError('Currently only batch_size of 1 is supported')

        img_size = list(imgs.shape)[2:4]

        features = self.rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, anchor = self.rcnn.rpn.forward(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        rois = rois[0]

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator.forward(rois, bbox, label)

        roi_cls_loc, roi_label = self.rcnn.head.forward(features, sample_roi)
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator.forward(bbox, anchor, img_size)

        gt_rpn_label = gt_rpn_label.long()
        gt_roi_label = gt_roi_label.long()
        rpn_loc_loss = self._rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        self.rpn_cm.add(_rpn_score.detach(), _gt_rpn_label.detach())

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), gt_roi_label]

        roi_loc_loss = self._rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = self.CrossEntropyLoss(roi_label, gt_roi_label)
        self.roi_cm.add(roi_label.detach(), gt_roi_label.detach())
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def _rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).to(opt.device)
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(opt.device)] = 1
        loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        loc_loss /= ((gt_label >= 0).sum().float())
        return loc_loss

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.
        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.rcnn.state_dict()
        save_dict['config'] = opt.state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt.parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}
