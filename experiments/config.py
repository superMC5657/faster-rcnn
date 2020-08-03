from pprint import pprint

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'
import torch


class Config:
    """
    anchor
    size = (3,height,width)
    """
    # size
    size = (3, 640, 640)
    feat_stride = 16
    feat_width = size[2] // feat_stride
    feat_height = size[1] // feat_stride
    anchor_base_size = 16
    anchor_scales = [8, 16, 32]
    ratios = [0.5, 1.0, 2.0]

    train_voc_data_dir = '/home/supermc/Datasets/VOCdevkit/VOC2012'
    test_voc_data_dir = '/home/supermc/Datasets/VOCdevkit/VOC2007'

    data_path = "/home/supermc/Datasets"

    project = "voc"
    batch_size = 3
    num_workers = 1
    test_num_workers = 1

    device = torch.device("cuda:0")

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'fasterRcnn'  # visdom env
    url = '10.20.216.190'
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg-16'

    # training
    epoch = 14

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead

    # debug
    debug_file = '/tmp/debugf'
    train_num = None
    test_num = 1000
    # model
    load_path = None
    sgd_momentum = 0.9
    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def parse(self, kwargs):
        state_dict = self.state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self.state_dict())
        print('==========end============')

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
