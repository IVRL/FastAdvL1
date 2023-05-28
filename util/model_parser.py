import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import pdb

from arch.preprocess import DataNormalizeLayer
from arch.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from arch.wrn import WideResNet34_10
from arch.preact_resnet import PreActResNet18
from arch import fast_models

# mnist_normalize = {'bias': [0.5, 0.5, 0.5], 'scale': [0.5, 0.5, 0.5]}
# svhn_normalize = {'bias': [0.4380, 0.4440, 0.4730], 'scale': [0.1751, 0.1771, 0.1744]}
# cifar10_normalize = {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]}
# cifar100_normalize = {'bias': [0.5071, 0.4867, 0.4408], 'scale': [0.2675, 0.2565, 0.2761]}

normalize_dict = {
    'mnist': {'bias': [0.5, 0.5, 0.5], 'scale': [0.5, 0.5, 0.5]},
    'svhn': {'bias': [0.4380, 0.4440, 0.4730], 'scale': [0.1751, 0.1771, 0.1744]},
    'cifar10': {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'bias': [0.5071, 0.4867, 0.4408], 'scale': [0.2675, 0.2565, 0.2761]},
    'imagenet1000': {'bias': [0.485, 0.456, 0.406], 'scale': [0.229, 0.224, 0.225]},
    'imagenet100': {'bias': [0.485, 0.456, 0.406], 'scale': [0.229, 0.224, 0.225]},
    'tiny_imagenet': {'bias': [0.485, 0.456, 0.406], 'scale': [0.229, 0.224, 0.225]}
}

num_classes_dict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'imagenet100': 100,
    'tiny_imagenet': 200,
    'imagenet1000': 1000
}

def parse_model(dataset, model_type, normalize = None, **kwargs):

    # if isinstance(normalize, str):
    #     if normalize.lower() in ['cifar10', 'cifar10_normalize',]:
    #         normalize_layer = DataNormalizeLayer(bias = cifar10_normalize['bias'], scale = cifar10_normalize['scale'])
    #     else:
    #         raise ValueError('Unrecognized normalizer: %s' % normalize)
    # elif normalize is not None:
    #     normalize_layer = DataNormalizeLayer(bias = normalize['bias'], scale = normalize['scale'])
    # else:
    #     normalize_layer = DataNormalizeLayer(bias = 0., scale = 1.)
    assert dataset in ['cifar10', 'cifar100', 'mnist', 'svhn', 'imagenet100', 'tiny_imagenet', 'imagenet1000'], 'Dataset not included!'

    if normalize is not None:
        normalize_layer = DataNormalizeLayer(bias = normalize_dict[dataset]['bias'], scale = normalize_dict[dataset]['scale'])
    else:
        normalize_layer = DataNormalizeLayer(bias = 0., scale = 1.)

    num_classes = num_classes_dict[dataset]

    if model_type.lower() in ['resnet', 'resnet18']:
        net = ResNet18(num_classes=num_classes)
    elif model_type.lower() in ['wide_resnet', 'wrn', 'wideresnet']:
        net = WideResNet34_10(num_classes=num_classes)
    elif model_type.lower() in ['preact_resnet', 'preact_resnet18']:
        net = PreActResNet18(num_classes=num_classes)
    elif model_type.lower() in ['preact_resnet_softplus']:
        net = fast_models.PreActResNet18(n_cls=num_classes, activation='softplus1')
    elif model_type.lower() in ['resnet34']:
        net = ResNet34(num_classes=num_classes)
    elif model_type.lower() in ['resnet50']:
        net = ResNet50(num_classes=num_classes)
    elif model_type.lower() in ['resnet101']:
        net = ResNet101(num_classes=num_classes)

    else:
        raise ValueError('Unrecognized architecture: %s' % model_type)

    return nn.Sequential(normalize_layer, net)

