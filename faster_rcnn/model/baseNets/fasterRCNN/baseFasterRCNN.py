# -*- coding: utf-8 -*-
# !@time: 2020-07-03 22:15
# !@author: superMC @email: 18758266469@163.com
# !@fileName: _faster_rcnn.py
import torch
from torch import nn
from torchvision.ops import nms
import numpy as np
from faster_rcnn.data.transforms.image_utils import preprocess
from faster_rcnn.model.utils.bbox_tool import loc2bbox
from faster_rcnn.utils import array_tool as at
from torch.nn import functional as F
from experiments.config import opt


class BaseFasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(BaseFasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.optimizer = None
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

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
        raw_cls_bbox = raw_cls_bbox.reshape((-1, self.n_class, 4))
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox[:, l, :]
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

    def forward(self, x):
        image_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois = self.rpn(h, image_size)
        roi_cls_locs, roi_scores = self.head(h, rois)
        return roi_cls_locs, roi_scores, rois

    # maybe batch inference

    @torch.no_grad()
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois = self.forward(img)
            roi_scores = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            rois = at.totensor(rois) / scale

            mean = torch.tensor(self.loc_normalize_mean).to(opt.device).repeat(self.n_class)[None]
            std = torch.tensor(self.loc_normalize_std).to(opt.device).repeat(self.n_class)[None]

            roi_cls_loc = roi_cls_loc * std + mean
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

            rois = rois.view(-1, 1, 4).expand_as(roi_cls_loc)

            cls_bbox = loc2bbox(rois.reshape((-1, 4)),
                                roi_cls_loc.reshape((-1, 4)))

            cls_bbox = cls_bbox.to(opt.device)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            roi_scores = roi_scores.view(-1, self.n_class)

            # clip bbox
            cls_bbox[:, 0::2] = cls_bbox[:, 0::2].clamp(min=0, max=size[0])  # x
            cls_bbox[:, 1::2] = cls_bbox[:, 1::2].clamp(min=0, max=size[1])  # y

            prob = F.softmax(at.totensor(roi_scores), dim=1)
            bbox, label, score = self._suppress(cls_bbox, prob)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        lr = opt.lr
        weight_decay = opt.weight_decay
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=opt.sgd_momentum)
        return self.optimizer

    def scale_lr(self, decay=opt.lr_decay):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
