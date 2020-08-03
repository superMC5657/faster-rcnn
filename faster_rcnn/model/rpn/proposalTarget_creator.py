# -*- coding: utf-8 -*-
# !@time: 2020/7/10 下午5:54
# !@author: superMC @email: 18758266469@163.com
# !@fileName: proposalTarget_creator.py
import torch
from torchvision.ops import box_iou

from experiments.config import opt
from faster_rcnn.model.utils.bbox_tool import bbox2loc
import numpy as np
from faster_rcnn.utils import array_tool as at


class ProposalTargetCreator:

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
                 loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def single_forward(self, roi, bbox, label):
        n_bbox, _ = bbox.shape
        roi = torch.cat((bbox, roi), dim=0)  # trick 一定有个bbox 适配 方便之后cls
        pos_roi_per_image = int(self.n_sample * self.pos_ratio)  # hard negative mining
        iou = box_iou(roi, bbox)
        gt_assignment = iou.argmax(dim=1)
        max_iou = iou[torch.arange(gt_assignment.shape[0]), gt_assignment]
        # max_iou = iou.max(dim=1, keepdim=False)[0 ]  # keep_dim
        gt_roi_label = label[gt_assignment] + 1  # 背景为0 所以加1

        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.shape[0]))

        if pos_index.shape[0] > 0:
            indices = torch.randperm(pos_index.shape[0])[:pos_roi_per_this_image]
            pos_index = pos_index[indices]

        neg_index = torch.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.shape[0]))

        if neg_index.shape[0] > 0:
            indices = torch.randperm(neg_index.shape[0])[:neg_roi_per_this_image]
            neg_index = neg_index[indices]

        keep_index = torch.cat((pos_index, neg_index), dim=0)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - torch.tensor(self.loc_normalize_mean, dtype=torch.float32).to(
            opt.device)) / torch.tensor(self.loc_normalize_std, dtype=torch.float32).to(opt.device)

        return sample_roi, gt_roi_loc, gt_roi_label

    def __call__(self, rois, bboxs, labels):
        batch_size = rois.shape[0]
        sample_rois = []
        gt_roi_locs = []
        gt_roi_labels = []
        for i in range(batch_size):
            roi = rois[i]
            bbox = bboxs[i]
            label = labels[i]
            arg = torch.where(label == -1.)[1]
            len = label.shape[0] - arg.shape[0]
            bbox = bbox[:len]
            label = label[:len]
            sample_roi, gt_roi_loc, gt_roi_label = self.single_forward(roi, bbox, label)
            sample_rois.append(sample_roi)
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label)
        sample_rois = torch.stack(sample_rois, dim=0)
        gt_roi_locs = torch.stack(gt_roi_locs, dim=0)
        gt_roi_labels = torch.stack(gt_roi_labels, dim=0)
        return sample_rois, gt_roi_locs, gt_roi_labels
