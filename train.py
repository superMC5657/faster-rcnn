# -*- coding: utf-8 -*-
# !@time: 2020/7/6 下午9:00
# !@author: superMC @email: 18758266469@163.com
# !@fileName: train.py

import os
import ipdb
import matplotlib
import torch
from tqdm import tqdm

from experiments.config import opt
from faster_rcnn.data.datasets.dataset import Dataset, TestDataset
from faster_rcnn.data.transforms.image_utils import inverse_normalize
from faster_rcnn.model.baseTrainer import FasterRCNNTrainer
from faster_rcnn.model.rcnn import FasterRCNN
from torch.utils.data import DataLoader

import resource

from faster_rcnn.utils import array_tool
from faster_rcnn.utils.eval_tool import eval_detection_voc
from faster_rcnn.utils.vis_tool import visdom_bbox

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for index, (imgs_, sizes_, gt_bboxes_, gt_labels_, gt_difficult_) in enumerate(tqdm(dataloader)):
        sizes_ = [sizes_[0][0].item(), sizes_[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = rcnn.predict(imgs_, [sizes_])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficult_.numpy())

        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if index == test_num:
            break
    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
                                use_07_metric=True)
    return result


def train(**kwargs):
    opt.parse(kwargs)
    dataset = Dataset(opt)
    print('load dataset')
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataset = TestDataset(opt)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)

    rcnn = FasterRCNN(opt.pretrained_model)
    print('construct completed')

    trainer = FasterRCNNTrainer(rcnn).to(opt.device).train()
    if opt.load_path:
        trainer.load(opt.load_path)
    trainer.vis.log(dataset.db.label_names, win='labels')
    best_map = 0

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for index, (imgs, gt_bboxes, gt_labels, scales) in enumerate(tqdm(dataloader)):
            scales = array_tool.scalar(scales)
            imgs = imgs.to(opt.device).float()
            gt_bboxes = gt_bboxes.to(opt.device)
            gt_labels = gt_labels.to(opt.device)

            trainer.train_step(imgs, gt_bboxes, gt_labels, scales)

            if (index + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                trainer.vis.plot_many(trainer.get_meter_data())
                ori_img_ = inverse_normalize(array_tool.tonumpy(imgs[0]))
                gt_img = visdom_bbox(ori_img_, array_tool.tonumpy(gt_bboxes[0]), array_tool.tonumpy(gt_labels[0]))
                trainer.vis.img('gt_img', gt_img)

                pred_bboxes, pred_labels, pred_scores = trainer.rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_, array_tool.tonumpy(pred_bboxes[0]),
                                       array_tool.tonumpy(pred_labels[0]).reshape(-1),
                                       array_tool.tonumpy(pred_scores[0]))

                trainer.vis.img('pred_img', pred_img)

                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                trainer.vis.img('roi_cm', array_tool.totensor(trainer.rpn_cm.conf).cpu().float())

        eval_result = eval(test_dataloader, rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}\n map:{}\nloss:{}'.format(str(lr_), str(eval_result['map']), str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] >= best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.rcnn.scale_lr(opt.lr_decay)
        if epoch == 13:
            break


if __name__ == '__main__':
    train()
