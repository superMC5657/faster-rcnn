# -*- coding: utf-8 -*-
# !@time: 2020-07-06 05:42
# !@author: superMC @email: 18758266469@163.com
# !@fileName: normalize_tool.py

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
