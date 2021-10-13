import os
import sys
import logging
import random
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for m in feature.sublayers():
        if isinstance(m, (nn.Conv2D, nn.Conv3D)):
            conv_init(m.weight)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            value_init = nn.initializer.Constant(1)
            value_init(m.weight)
            bias_init = nn.initializer.Constant(0)
            bias_init(m.bias)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)

        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.shape)
        p = 1
        for j in range(i):
            p *= parameter.shape[j]
        total_paramters += p

    return total_paramters


def init_logger(log_file=None, name='root', log_level=logging.DEBUG):
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        dir_name = os.path.dirname(log_file)
        if len(dir_name) > 0 and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(log_level)
    return logger