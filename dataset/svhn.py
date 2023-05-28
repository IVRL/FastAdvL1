import os
import sys
sys.path.insert(0, './')
import numpy as np
import scipy.io as sio

from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .utility import SubsetRandomSampler, SubsetSampler, HybridBatchSampler

class IndexedSVHN(datasets.SVHN):

    def __init__(self, root, split, download, transform):

        super(IndexedSVHN, self).__init__(root = root, split = split, download = download, transform = transform)

    def __getitem__(self, index):

        img, target = super(IndexedSVHN, self).__getitem__(index)

        return img, target, index

def svhn(batch_size, valid_ratio = None, shuffle = True,  augmentation=True, train_subset = None):

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.ToTensor()
        ]) if augmentation == True else transforms.Compose([
        transforms.ToTensor()
        ])
    transform_valid = transforms.Compose([
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])
        
    trainset = IndexedSVHN(root = './data/svhn', split = 'train', download = True, transform = transform_train)
    validset = IndexedSVHN(root = './data/svhn', split = 'train', download = True, transform = transform_valid)
    testset = IndexedSVHN(root = './data/svhn', split = 'test', download = True, transform = transform_test)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if train_subset is None:
        instance_num = len(trainset)
        indices = list(range(instance_num))
    else:
        indices = np.random.permutation(train_subset)
        instance_num = len(indices)
    print('%d instances from the training set are picked' % instance_num)

    if valid_ratio is not None and valid_ratio > 0.:
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        if shuffle == True:
            train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = SubsetSampler(train_idx), SubsetSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 0, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    else:
        if shuffle == True:
            train_sampler = SubsetRandomSampler(indices)
        else:
            train_sampler = SubsetSampler(indices)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = True)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    return train_loader, valid_loader, test_loader, classes

class IndexedSVHN_Plus(torch.utils.data.Dataset):

    def __init__(self, root4ori, root4plus, split, download, transform):

        super(IndexedSVHN_Plus, self).__init__()

        self.set_ori = IndexedSVHN(root = root4ori, split = split, download = download, transform = transform)
        self.ori_num = len(self.set_ori)
        if split.lower() in ['test',]:
            self.data_list_plus = []
            self.label_list_plus = []
            self.plus_num = 0
        else:
            data_file = os.path.join(root4plus, 'extra_32x32.mat')
            data = sio.loadmat(data_file)
            self.data_list_plus = np.transpose(data['X'].astype(np.uint8), (3, 2, 0, 1))
            self.label_list_plus = data['y'].astype(np.int64).squeeze()
            np.place(self.label_list_plus, self.label_list_plus == 10, 0)
            self.plus_num = len(self.label_list_plus)

        self.transform = transform

    def __len__(self,):

        return self.ori_num + self.plus_num

    def __getitem__(self, index):

        assert index >= 0 and index < self.__len__(), 'invalid index %d, the dataset size is %d' % (index, self.__len__())

        if index < self.ori_num:
            img, target, index = self.set_ori.__getitem__(index)
        else:
            index_in_plus = index - self.ori_num
            img, target = self.data_list_plus[index_in_plus], self.label_list_plus[index_in_plus]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                img = self.transform(img)

        return img, target, index

def svhn_plus(batch_size, plus_prop, valid_ratio, train_subset = None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    trainset = IndexedSVHN_Plus(root4ori = './data/svhn', root4plus = '/ivrldata1/data/cliu/svhn', split = 'train', download = True, transform = transform)
    validset = IndexedSVHN_Plus(root4ori = './data/svhn', root4plus = '/ivrldata1/data/cliu/svhn', split = 'train', download = True, transform = transform)
    testset = IndexedSVHN(root = './data/svhn', split = 'test', download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if train_subset is None:
        total_indices = list(range(trainset.__len__()))
        ori_indices = list(range(trainset.ori_num))
        plus_indices = list(range(trainset.ori_num, trainset.ori_num + trainset.plus_num))
    else:
        ori_indices = list(filter(lambda x: x < trainset.ori_num, train_subset))
        plus_indices = list(range(trainset.ori_num, trainset.ori_num + trainset.plus_num))
        total_indices = len(ori_indices) + len(plus_indices)
    print('%d instances from the original dataset are picked and %d instances from the additional dataset are picked' % (len(ori_indices), len(plus_indices)))

    if valid_ratio is not None and valid_ratio > 0.:
        split_pt = int(len(ori_indices) * valid_ratio)
        ori_train_idx, plus_train_idx, valid_idx = ori_indices[split_pt:], plus_indices, ori_indices[:split_pt]

        train_sampler = HybridBatchSampler(idx4ori = ori_train_idx, idx4plus = plus_train_idx, plus_prop = plus_prop, batch_size = batch_size, permutation = True)
        valid_sampler = HybridBatchSampler(idx4ori = valid_idx, idx4plus = [], plus_prop = -1, batch_size = batch_size, permutation = True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler = train_sampler, num_workers = 4, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_sampler = valid_sampler, num_workers = 4, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    else:
        train_sampler = HybridBatchSampler(idx4ori = ori_indices, idx4plus = plus_indices, plus_prop = plus_prop, batch_size = batch_size, permutation = True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler = train_sampler, num_workers = 4, pin_memory = True)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, valid_loader, test_loader, classes
