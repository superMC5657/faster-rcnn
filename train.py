# -*- coding: utf-8 -*-
# !@time: 2020/7/6 下午9:00
# !@author: superMC @email: 18758266469@163.com
# !@fileName: train.py

import resource

import ipdb
import matplotlib
from tqdm import tqdm

from faster_rcnn.data.datasets.coco_dataset import *
from faster_rcnn.data.transforms.image_utils import inverse_normalize
from faster_rcnn.model.baseNets.fasterRCNN.fasterRCNN import FasterRCNN
from faster_rcnn.model.trainer.baseFasterTrainer import FasterRCNNTrainer
from faster_rcnn.model.utils.bbox_tool import generate_anchor_base, enumerate_shift_anchor
from faster_rcnn.utils import array_tool
from faster_rcnn.utils.eval_tool import eval_detection_voc
from faster_rcnn.utils.vis_tool import visdom_bbox

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for index, (imgs_, sizes_, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(tqdm(dataloader)):
        sizes_ = [sizes_[0][0].item(), sizes_[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = rcnn.predict(imgs_, [sizes_])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())

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

    anchor_base = generate_anchor_base(base_size=opt.anchor_base_size, anchor_scales=opt.anchor_scales,
                                       ratios=opt.ratios)
    opt.diff_anchor_num = len(anchor_base)
    anchor = enumerate_shift_anchor(anchor_base, opt.feat_stride, opt.feat_height,
                                    opt.feat_width)
    opt.anchor = anchor

    print('load dataset')

    params = Params(f'projects/{opt.project}.yml')
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name, params.train_set),
                               set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(opt.size[1])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name, params.val_set), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(opt.size[1])]))
    val_generator = DataLoader(val_set, **val_params)

    rcnn = FasterRCNN(opt.pretrained_model)
    print('construct completed')

    trainer = FasterRCNNTrainer(rcnn).to(opt.device).train()
    if opt.load_path:
        trainer.load(opt.load_path)
    best_map = 0

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for index, sample in enumerate(tqdm(training_generator)):
            if index == opt.train_num:
                break
            img = sample['img']
            annot = sample['annot']
            img = img.to(opt.device).float()
            annot = annot.to(opt.device)

            trainer.train_step(img, annot)

        # eval_result = eval(test_dataloader, rcnn, test_num=opt.test_num)
        # trainer.vis.plot('test_map', eval_result['map'])
        # lr_ = trainer.rcnn.optimizer.param_groups[0]['lr']
        # log_info = 'lr:{}\n map:{}\nloss:{}'.format(str(lr_), str(eval_result['map']), str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        #
        # if eval_result['map'] >= best_map:
        #     best_map = eval_result['map']
        #     best_path = trainer.save(best_map=best_map)
        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.rcnn.scale_lr(opt.lr_decay)
        # if epoch == 13:
        #     break


if __name__ == '__main__':
    train()
