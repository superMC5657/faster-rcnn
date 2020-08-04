# This is a Faster-RCNN practice
Author: 张研,陈先客

batch_size = 1,dataloader 返回 img,bbox,label,scale,因为bbox , label维度不一致所以 不能组装
batch_size = n, reszie好图片大小 在dataloader时提供所有的每个anchor bbox偏移和对应的label