# -*- coding: utf-8 -*-
# !@time: 2020-07-03 22:15
# !@author: superMC @email: 18758266469@163.com
# !@fileName: _faster_rcnn.py
import torch
from torch import nn
from torchvision.ops import nms
import numpy as np
from faster_rcnn.data.transforms.image_utils import preprocess
from faster_rcnn.utils import array_tool


def nograd(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


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
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7

        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05

        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((raw_cls_bbox, raw_prob))
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    # maybe batch inference
    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(array_tool.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = array_tool.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self.forward(img, scale)
