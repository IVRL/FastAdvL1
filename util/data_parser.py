import sys
sys.path.insert(0, './')
import numpy as np

import json

from dataset.cifar10 import cifar10, cifar10_plus
from dataset.svhn import svhn, svhn_plus
from dataset.cifar100 import cifar100
from dataset.imagenet100 import imagenet100

def parse_data(name, batch_size, valid_ratio = None, shuffle = True, augmentation = True, **kwargs):

    if name.lower() in ['cifar10',]:
        train_loader, valid_loader, test_loader, classes = cifar10(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    elif name.lower() in ['cifar100']:
        train_loader, valid_loader, test_loader, classes = cifar100(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    elif name.lower() in ['svhn',]:
        train_loader, valid_loader, test_loader, classes = svhn(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    elif name.lower() in ['imagenet100', 'imagenet']:
        train_loader, valid_loader, test_loader, classes = imagenet100(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    else:
        raise ValueError('Unrecognized name of the dataset: %s' % name)

    return train_loader, valid_loader, test_loader, classes

def parse_plus_data(name, batch_size, plus_prop, valid_ratio = None, augmentation = True, train_subset = None):

    if name.lower() in ['cifar10',]:
        train_loader, valid_loader, test_loader, classes = cifar10_plus(batch_size = batch_size, plus_prop = plus_prop, valid_ratio = valid_ratio, augmentation = augmentation, train_subset = train_subset)
    elif name.lower() in ['svhn',]:
        train_loader, valid_loader, test_loader, classes = svhn_plus(batch_size = batch_size, plus_prop = plus_prop, valid_ratio = valid_ratio, train_subset = train_subset)
    else:
        raise ValueError('Unrecognized name of the dataset: %s' % name)

    return train_loader, valid_loader, test_loader, classes

def parse_subset(per_file, subset):

    mode = subset['mode']
    instance_num = int(subset['num'])

    meta_data = json.load(open(per_file, 'r'))
    train_report = meta_data['train_per_report']
    class_num = len(train_report)
    instance_per_class = int(instance_num / class_num)

    chosen_idx = []
    for label in train_report:
        sorted_this_label = list(sorted(train_report[label], key = lambda x: x['per'][0]))
        idx_this_label = list(map(lambda x: x['idx'], sorted_this_label))
        if mode.lower() in ['easy',]:
            chosen_idx = chosen_idx + idx_this_label[-instance_per_class:]
        elif mode.lower() in ['hard',]:
            chosen_idx = chosen_idx + idx_this_label[:instance_per_class]
        elif mode.lower() in ['random',]:
            chosen_idx = chosen_idx + list(np.random.permutation(idx_this_label)[:instance_per_class])
        else:
            raise ValueError('Invalid mode: %s' % mode)

    print('>>> Pick %d instances from the training set' % len(chosen_idx))

    return chosen_idx

def parse_groups(per_file, group_num):

    meta_data = json.load(open(per_file, 'r'))
    idx2group = {'train': {}, 'valid': {}, 'test': {}}

    for subset in ['train', 'valid', 'test']:
        report = meta_data['%s_per_report' % subset]
        for label in report:
            for item in report[label]:
                index = int(item['idx'])
                per = item['per'][0]
                group_idx = min(int(per * group_num), group_num - 1)
                idx2group[subset][index] = group_idx

    return idx2group
