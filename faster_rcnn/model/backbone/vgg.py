# -*- coding: utf-8 -*-
# !@time: 2020/7/4 下午1:46
# !@author: superMC @email: 18758266469@163.com
# !@fileName: vgg.py
from torchvision.models import vgg16


def decom_vgg16(model_path):
    # the 30th layer of features is relu of conv5_3

    model = vgg16(model_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier
