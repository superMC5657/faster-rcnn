# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午1:46
# !@author: superMC @email: 18758266469@163.com
# !@fileName: vgg.py
from torchvision.models import vgg16
from torch import nn
from experiments.config import opt

layers = {16: vgg16}


def vgg(num=16, pretrained=False):
    # the 30th layer of features is relu of conv5_3

    model = layers[num](pretrained)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:  # dropout
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    if pretrained:
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

    return nn.Sequential(*features), classifier


if __name__ == '__main__':
    vgg(16, True)
