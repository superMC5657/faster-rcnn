# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午2:16
# !@author: superMC @email: 18758266469@163.com
# !@fileName: bbox_tool.py

import numpy as np
import torch
from six import moves

#  loc 可能为bbox的偏移
from experiments.config import opt


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.type(src_bbox.dtype).to(opt.device)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, None] + src_ctr_y[:, None]
    ctr_x = dx * src_width[:, None] + src_ctr_x[:, None]
    h = torch.exp(dh) * src_height[:, None]
    w = torch.exp(dw) * src_width[:, None]

    dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype).to(opt.device)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    loc = torch.stack((dy, dx, dh, dw), dim=1)
    return loc


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = torch.zeros((len(ratios) * len(anchor_scales), 4), dtype=torch.float32)
    for i in moves.range(len(ratios)):
        for j in moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def enumerate_shift_anchor(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = torch.from_numpy(shift_x)
    shift_y = torch.from_numpy(shift_y)

    # shift_y = torch.arange(0, height * feat_stride, feat_stride)
    # shift_x = torch.arange(0, width * feat_stride, feat_stride)
    # shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    shift = torch.stack((shift_y.flatten(), shift_x.flatten(), shift_y.flatten(), shift_x.flatten()), dim=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).permute((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).type(torch.float32)
    return anchor


def __test():
    src_bbox = [[1, 1, 2, 2]]
    loc = [[0.1, 0.1, 0.2, 0.2]]
    dst_bbox = bbox2loc(torch.tensor(src_bbox), torch.tensor(loc))
    print(dst_bbox)


if __name__ == '__main__':
    __test()
