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
from faster_rcnn.model.rpn.anchorTarget_creator import AnchorTargetCreator
from faster_rcnn.model.trainer.baseFasterTrainer import FasterRCNNTrainer
from faster_rcnn.model.utils.bbox_tool import generate_anchor_base, enumerate_shift_anchor, loc2bbox
from faster_rcnn.utils import array_tool
from faster_rcnn.utils.eval_tool import eval_detection_voc
from faster_rcnn.utils.vis_tool import visdom_bbox

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for index, sample in enumerate(tqdm(dataloader)):
        img = sample['img']
        annot = sample['annot']
        img = img.to(opt.device).float()
        annot = annot.to(opt.device)
        bbox = annot[0, :, :4]
        label = annot[0, :, 4:5].int()
        arg = torch.where(label == -1.)[1]
        _len = label.shape[0] - arg.shape[0]
        bbox = bbox[:_len]
        label = label[:_len]
        pred_bboxes_, pred_labels_, pred_scores_ = rcnn.predict(img)

        gt_bboxes += [bbox.cpu().numpy()]
        gt_labels += [label.squeeze(dim=-1).cpu().numpy()]
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if index == test_num:
            break
    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
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

    val_params = {'batch_size': 1,
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
            if (index + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                trainer.vis.plot_many(trainer.get_meter_data())
                ori_img_ = inverse_normalize(array_tool.tonumpy(img[0]))
                bbox = annot[0, :, :4]
                label = annot[0, :, 4:5].int()
                arg = torch.where(label == -1.)[1]
                _len = label.shape[0] - arg.shape[0]
                bbox = bbox[:_len]
                label = label[:_len]
                label = label.squeeze(dim=-1)
                rpn_img = display(ori_img_, bbox)
                trainer.vis.img('rpn_img', rpn_img)

                bbox = xy2yx(bbox)
                gt_img = visdom_bbox(ori_img_, array_tool.tonumpy(bbox),
                                     array_tool.tonumpy(label))

                trainer.vis.img('gt_img', gt_img)

                pred_bboxes, pred_labels, pred_scores = trainer.rcnn.predict([ori_img_], visualize=True)
                pred_bbox = xy2yx(pred_bboxes[0])
                pred_img = visdom_bbox(ori_img_, array_tool.tonumpy(pred_bbox),
                                       array_tool.tonumpy(pred_labels[0]).reshape(-1),
                                       array_tool.tonumpy(pred_scores[0]))

                trainer.vis.img('pred_img', pred_img)

                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                trainer.vis.img('roi_cm', array_tool.totensor(trainer.rpn_cm.conf).cpu().float())

        eval_result = eval(val_generator, rcnn, test_num=opt.test_num)
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


def display(ori_img_, bbox):
    loc, label = AnchorTargetCreator().single_forward(bbox)
    label_1 = torch.where(label == 1)[0]
    loc = loc[label_1]
    label = label[label_1]
    anchor = opt.anchor[label_1]
    rpn_bbox = loc2bbox(anchor, loc)
    rpn_bbox = xy2yx(rpn_bbox)
    rpn_img = visdom_bbox(ori_img_, array_tool.tonumpy(rpn_bbox), array_tool.tonumpy(label))
    return rpn_img


def xy2yx(bbox):
    return bbox[:, [1, 0, 3, 2]]


if __name__ == '__main__':
    train()
