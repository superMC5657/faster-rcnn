# -*- coding: utf-8 -*-
# !@time: 2020/7/3 下午7:35
# !@author: superMC @email: 18758266469@163.com
# !@fileName: voc.py
import torch

from .voc_dataset import VOCBboxDataset
from ..transforms.image_utils import Transform, preprocess
from ...model.utils.bbox_tool import generate_anchor_base, enumerate_shift_anchor


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.train_voc_data_dir)
        self.tsf = Transform(opt.size)


    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.

        img = torch.from_numpy(img)
        bbox = torch.from_numpy(bbox)
        label = torch.from_numpy(label)
        scale = torch.tensor(scale)

        return img

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.test_voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
