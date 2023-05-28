import os
import sys
sys.path.insert(0, './')

import time
import numpy as np
from copy import deepcopy
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utility import project
from .autopgd_train import apgd_train
import pdb

h_message = '''
>>> PGD(step_size, threshold, iter_num, order = np.inf)
>>> APGD(threshold, iter_num, rho, loss_type, alpha = 0.75, order = np.inf)
>>> Square(threshold, window_size_factor, iter_num, order = np.inf)
'''

def parse_attacker(name, **kwargs):

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['pgd',]:
        return PGD(**kwargs)
    elif name.lower() in ['fastegl1']:
        return FastEGL1(**kwargs)
    elif name.lower() in ['nuat']:
        return NuAT(**kwargs)
    elif name.lower() in ['nfgsm']:
        return NFSGM(**kwargs)
    elif name.lower() in ['gradalign']:
        return Gradalign(**kwargs)
    elif name.lower() in ['adat']:
        return AdAT(**kwargs)
    elif name.lower() in ['apgd']:
        return APGD(**kwargs)
    else:
        raise ValueError('Unrecognized name of the attacker: %s' % name)

def init_delta(x, order, threshold):
    ori_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    N, ndim = x.shape

    if order == np.inf:
        delta = np.random.uniform(-threshold, threshold, ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    elif order == 2:
        r = np.random.normal(size=(N, ndim))
        norm = np.linalg.norm(r, axis=-1, keepdims=True)
        delta = (r / norm) * threshold * np.random.uniform(size=(N, ndim)) ** (1/ndim)
        delta = delta.reshape(ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    elif order == 1:
        r = np.random.laplace(size=(N, ndim))
        norm = np.linalg.norm(r, axis=-1, ord=1, keepdims=True)
        delta = (r / norm) * threshold * np.random.uniform(size=(N, ndim)) ** (1/ndim)
        delta = delta.reshape(ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    else:
        raise ValueError("Unknown norm {}".format(order))


class PGD(object):
    '''
    Methods for FGSM and PGD
        - order : order of the attack, in [-1 or np.inf, 1, 2],
                  when order=1, it refers to Sparse L1 Descent (SLIDE, https://arxiv.org/pdf/1904.13000.pdf)
        - eps_train: training on larger budget when eps_train > 1
    '''
    def __init__(self, step_size, threshold, iter_num, eps_train=1, order = np.inf, **kwargs):

        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf

        self.step_size = step_size
        self.threshold = threshold 
        if self.order == np.inf and threshold >= 1:
            self.threshold = threshold / 255
        self.eps_train = eps_train 

        if self.order == 1:
            # k coordinate descent for SLIDE algorithm
            self.k = kwargs.get('k', [5, 20])

        self.random_start = kwargs.get('random_start', True)

        print('Create a PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.order))

    def attack(self, model, data_batch, label_batch, criterion=None, is_train=False, random_start=True, ori_batch=None, **kwargs):
        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold, order = self.step_size, self.threshold, self.order

        criterion = criterion.cuda(device) if criterion else nn.CrossEntropyLoss().cuda()

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        if ori_batch is None:
            ori_batch = data_batch.detach()

        random_start = random_start and self.random_start
        # Initial perturbation
        noise = init_delta(data_batch, threshold=threshold, order=order) if random_start else 0.

        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        model.eval()
        for iter_idx in range(self.iter_num):
            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            elif self.order == 1:
                ## method of SLIDE
                ori_shape = data_batch.size()
                data_size = data_batch.numel() / batch_size
                k = random.randint(self.k[0], self.k[1]) if type(self.k) is list else self.k
                # perb = l1_dir_topk(grad, data_batch, gap=step_size, k=k) * step_size * 20 / k
                perb = l1_dir_topk(grad, data_batch, gap=step_size / k, k=k) * step_size / k 
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            proj_order = self.order if self.order != 21 else 1
            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold*self.eps_train, order = proj_order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()
            model.zero_grad()

        if is_train:
            model.train()

        logits = model(data_batch)
        loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
        loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
        # print(int(loss_update_bits.sum()))
        mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
        loss_update_bits = loss_update_bits.view(*mask_shape)
        adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())

        return adv_data_batch, label_batch

class FastEGL1(object):
    '''
    Our proposed method Fast-EG-L1
        - order: order of the attack
        - update_order: update direction, 2 in our paper
        - eps_train: training on larger budget when eps_train > 1, set to 2 in our paper
    '''
    def __init__(self, step_size=None, threshold=12, iter_num=1, order = 1, eps_train=2, update_order=2, **kwargs):

        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.update_order = self.order if update_order is None else update_order

        self.step_size = step_size if step_size is not None else np.sqrt(threshold)
        self.threshold = threshold 
        if self.order == np.inf and threshold >= 1:
            self.threshold = threshold / 255

        self.eps_train = eps_train
        if self.order == 1:
            # k coordinate descent for SLIDE algorithm
            self.k = kwargs.get('k', [5, 20])

        print('Create a Fast-EG-L1 attacker')

    def attack(self, model, data_batch, label_batch, criterion=None, is_train=False, random_start=True, ori_batch=None, **kwargs):
        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold, order = self.step_size, self.threshold, self.order

        criterion = criterion.cuda(device) if criterion else nn.CrossEntropyLoss().cuda()

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        # Initial perturbation
        noise = init_delta(data_batch, threshold=threshold, order=order) if random_start else 0.
        # noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold, threshold = threshold, order = self.order)

        # noise as data augmentation
        data_batch_noise = torch.clamp(data_batch + noise, min = 0., max = 1.)
        ori_batch = data_batch.clone()

        data_batch = data_batch_noise.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        model.eval()

        for iter_idx in range(self.iter_num):
            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.update_order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.update_order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            elif self.update_order == 1:
                ori_shape = data_batch.size()
                data_size = data_batch.numel() / batch_size
                k = random.randint(self.k[0], self.k[1]) if type(self.k) is list else self.k
                # perb = l1_dir_topk(grad, data_batch, gap=step_size, k=k) * step_size * 20 / k
                perb = l1_dir_topk(grad, data_batch, gap=step_size / k, k=k) * step_size / k 
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold*self.eps_train, order = self.order)            
            next_point = torch.clamp(next_point, min = 0., max = 1.)
            data_batch = next_point.detach().requires_grad_()

        if is_train:
            model.train()

        return data_batch.detach(), label_batch

class NuAT(object):
    '''
    [NeurIPS'21] Towards efficient and effective adversarial training (https://openreview.net/forum?id=kuK2VARZGnI)
        - lmbd: weight parameter of nuclear norm regularization
    '''
    def __init__(self, step_size, threshold, iter_num=1, order=1, update_order=None, eps_train=1, lmbd=3, **kwargs):

        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.update_order = self.order if update_order is None else update_order

        self.step_size = step_size
        self.threshold = threshold 
        if self.order == np.inf and threshold >= 1:
            self.threshold = threshold / 255
        self.eps_train = eps_train

        if self.order == 1:
            self.k = kwargs.get('k', [5, 20])

        self.lmbd = lmbd
        print('Create a NuAT attacker')


    def attack(self, model, data_batch, label_batch, criterion=None, is_train=False, random_start=True, ori_batch=None, **kwargs):
        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold, order = self.step_size, self.threshold, self.order

        criterion = criterion.cuda(device) if criterion else nn.CrossEntropyLoss().cuda()

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        if ori_batch is None:
            ori_batch = data_batch.detach()

        ori_logit = model(ori_batch)
        # Initial perturbation
        noise = init_delta(data_batch, threshold=threshold, order=order) if random_start else 0.
        # noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold, threshold = threshold, order = self.order)

        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        model.eval()

        for iter_idx in range(self.iter_num):
            logits = model(data_batch)
            loss = criterion(logits, label_batch) + self.lmbd * torch.norm(ori_logit - logits, 'nuc') / batch_size
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.update_order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.update_order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            elif self.update_order == 1:
                ori_shape = data_batch.size()
                data_size = data_batch.numel() / batch_size
                k = random.randint(self.k[0], self.k[1]) if type(self.k) is list else self.k
                # perb = l1_dir_topk(grad, data_batch, gap=step_size, k=k) * step_size * 20 / k
                perb = l1_dir_topk(grad, data_batch, gap=step_size / k, k=k) * step_size / k 
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold*self.eps_train, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()
            model.zero_grad()

        if is_train:
            model.train()

        return data_batch, label_batch

class NFSGM(object):
    '''
    [Neurips'22] Make Some Noise: Reliable and Efficient Single-Step Adversarial Training
        - k_epsilon: radius of the noise for data augmentation
    '''
    def __init__(self, step_size, threshold=12, iter_num=1, k_epsilon=2, eps_train=1, order = 1, update_order=None, projection=True, **kwargs):

        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.update_order = self.order if update_order is None else update_order

        self.step_size = step_size
        self.threshold = threshold 
        if self.order == np.inf and threshold >= 1:
            self.threshold = threshold / 255

        if self.order == 1:
            self.k = kwargs.get('k', [5, 20])

        self.k_epsilon = k_epsilon
        self.eps_train = eps_train

        self.projection = projection
 
        print('Create a NPGD attacker')
        
    def attack(self, model, data_batch, label_batch, criterion=None, is_train=False, random_start=True, ori_batch=None, **kwargs):
        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold, order = self.step_size, self.threshold, self.order

        criterion = criterion.cuda(device) if criterion else nn.CrossEntropyLoss().cuda()

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        # Initial perturbation
        noise = init_delta(data_batch, threshold=threshold*self.k_epsilon, order=order) if random_start else 0.
        # noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold, threshold = threshold, order = self.order)

        # noise as data augmentation
        data_batch_noise = torch.clamp(data_batch + noise, min = 0., max = 1.)
        ori_batch = data_batch_noise.clone() 

        data_batch = data_batch_noise.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        model.eval()

        for iter_idx in range(self.iter_num):
            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.update_order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.update_order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            elif self.update_order == 1:
                ori_shape = data_batch.size()
                data_size = data_batch.numel() / batch_size
                k = random.randint(self.k[0], self.k[1]) if type(self.k) is list else self.k
                # perb = l1_dir_topk(grad, data_batch, gap=step_size, k=k) * step_size * 20 / k
                perb = l1_dir_topk(grad, data_batch, gap=step_size / k, k=k) * step_size / k 
                # raise Exception('The line above need reconsidering, the step size defined here is not consistent with the ones in l2/linf cases')
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            if self.projection:
                next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold*self.eps_train, order = self.order)
            
            next_point = torch.clamp(next_point, min = 0., max = 1.)
            data_batch = next_point.detach().requires_grad_()

        if is_train:
            model.train()

        return data_batch.detach(), label_batch

class AdAT(object):
    '''
    [AAAI'21] Understanding Catastrophic Overfitting in Single-step Adversarial Training
    [https://arxiv.org/abs/2010.01799]
    '''
    def __init__(self, step_size, threshold=12, iter_num=1, eps_train=1, order = np.inf, update_order=None, c=3, log = 0, projection=True, **kwargs):

        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.update_order = self.order if update_order is None else update_order

        self.step_size = step_size
        self.threshold = threshold 
        if self.order == np.inf and threshold >= 1:
            self.threshold = threshold / 255
        self.step_size_threshold_ratio = self.step_size / (self.threshold + 1e-6)

        if self.order == 1:
            self.k = kwargs.get('k', [5, 20])

        self.random_start = kwargs.get('random_start', True)
        self.eps_train = eps_train
        self.c = c

        self.log = False if int(log) == 0 else True

        print('Create a AdAT attacker')

    # attack without restart
    def attack(self, model, data_batch, label_batch, criterion=None, 
                    is_train=False, random_start=True, ori_batch=None, **kwargs):
        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold, order = self.step_size, self.threshold, self.order

        criterion = criterion.cuda(device) if criterion else nn.CrossEntropyLoss().cuda()

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        if ori_batch is None:
            ori_batch = data_batch.detach()

        random_start = random_start and self.random_start
        # Initial perturbation
        noise = init_delta(data_batch, threshold=threshold, order=order) if random_start else 0.
        # noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold, threshold = threshold, order = self.order)

        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        model.eval()

        for iter_idx in range(self.iter_num):
            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.update_order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.update_order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            elif self.update_order == 1:
                ## method of SLIDE
                ori_shape = data_batch.size()
                data_size = data_batch.numel() / batch_size
                k = random.randint(self.k[0], self.k[1]) if type(self.k) is list else self.k
                # perb = l1_dir_topk(grad, data_batch, gap=step_size, k=k) * step_size * 20 / k
                perb = l1_dir_topk(grad, data_batch, gap=step_size / k, k=k) * step_size / k 
                # raise Exception('The line above need reconsidering, the step size defined here is not consistent with the ones in l2/linf cases')
                next_point = data_batch + perb.view(ori_shape)

            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold*self.eps_train, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()

            model.zero_grad()

        if is_train:
            model.train()

        correctness = []
        with torch.no_grad():
            last_correct = torch.ones(batch_size).to(device) == 1
            perb = data_batch - ori_batch
            mini_perb = torch.zeros(batch_size).to(device)
            for idx_c in range(1, self.c+1):
                delta = perb * idx_c / self.c
                logit = model(ori_batch + delta)
                correct = logit.argmax(dim=1) == label_batch
                correctness.append(correct.detach().cpu().numpy())
                
                mini_perb[correct] += 1
                mini_perb[(~correct) & last_correct] += 1

                last_correct = correct

            data_batch = ori_batch + perb * mini_perb.view(batch_size, 1, 1, 1) / self.c

        return data_batch.detach(), label_batch

class Gradalign(object):
    '''
    [NeurIPS'20] Understanding and Improving Fast Adversarial Training
    '''
    def __init__(self, **kwargs):
        self.pgd = PGD(**kwargs)

        self.order = self.pgd.order
        self.threshold = self.pgd.threshold
        self.iter_num = self.pgd.iter_num
        self.step_size = self.pgd.step_size

        self.lmbd = kwargs.get('lmbd', 5)

    def forward(self, **kwargs):
        return self.pgd(**kwargs)


class APGD(object):
    '''
    Adversarial Training with AutoPGD
    '''
    # code from https://github.com/fra31/robust-finetuning/blob/master/autopgd_train.py
    def __init__(self, threshold, iter_num, rho = .75, loss_type = 'ce', alpha = 0.75, order = np.inf, **kwargs):

        self.threshold = threshold 
        self.step_size = kwargs.get('step_size', threshold)
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.rho = rho
        self.alpha = alpha
        self.loss_type = loss_type

        self.status = 'normal'
        self.log = False
        self.random_start = kwargs.get('random_start', True)
        print('use random start', self.random_start)

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        print('Create a Auto-PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, rho = %.4f, alpha = %.4f, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.rho, self.alpha, self.order))
        print('loss type = %s' % self.loss_type)

    def adjust_threshold(self, threshold, log = True):

        threshold = float(threshold) # if threshold < 1. else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        if self.iter_num > 1:
            self.threshold = threshold

        if log == True:
            print('Attacker adjusted, threshold = %1.4e, step_size = %1.4e' % (self.threshold, self.step_size))

    def attack(self, model, data_batch, label_batch, criterion, is_train=False, verbose=False, restarts=1, **kwargs):
        norm = {np.inf: 'Linf', 1:'L1', 2: 'L2', 21: 'L21', 221: 'L221'}[self.order]
        p = 0.05 if is_train else 0.2
        
        # x_init = data_batch + init_delta(data_batch, order=self.order, threshold=self.threshold)

        model.eval()
        # attacker = APGDAttack(model, n_restarts = restarts, n_iter = self.iter_num, p=p, verbose=verbose, eps = self.threshold,
        #     norm = norm, eot_iter = 1, rho = self.rho, seed = time.time(), loss = self.loss_type, device = data_batch.device)
        
        ## adv_data_batch = attacker.perturb(data_batch, label_batch)
        # attacker.init_hyperparam(data_batch)
        # adv_data_batch, _, _, _ = attacker.attack_single_run(data_batch, label_batch, x_init='no_init')

        adv_data_batch, _, _, _ = apgd_train(model, data_batch, label_batch, norm=norm, step_size=self.step_size,
                                eps=self.threshold, n_iter=self.iter_num, random_start=self.random_start)
        if is_train:
            model.train()
            
        return adv_data_batch.detach(), label_batch

# from https://github.com/locuslab/robust_union
def l1_dir_topk(grad, X_curr, gap, k = 20) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    # X_curr = X + delta
    batch_size = X_curr.shape[0]
    channels = X_curr.shape[1]
    pix = X_curr.shape[2]
    # print (batch_size)
    # neg1 = (grad < 0)*(X_curr <= gap)
    # # neg1 = (grad < 0)*(X_curr == 0)
    # neg2 = (grad > 0)*(X_curr >= 1-gap)
    # # neg2 = (grad > 0)*(X_curr == 1)
    # neg3 = X_curr <= 0
    # neg4 = X_curr >= 1
    # neg = neg1 + neg2 + neg3 + neg4
    if type(gap) is not float:
        gap = gap.reshape(-1, 1, 1, 1)
    neg = (grad < 0)*(X_curr <= 0) + (grad > 0)*(X_curr >= 1-gap)
    u = neg.view(batch_size, 1, -1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval.half()).half() * grad_check.sign()

    # g1 = grad_check.clone()
    # g1[grad_check.abs() >= kval.half()] == 0
    # v1, k1 = g1.abs().reshape(128, -1).max(dim=1)
    # v2, k2 = grad_check.abs().reshape(128, -1).max(dim=1)
    # pdb.set_trace()
    return k_hot.view(batch_size, channels, pix, pix)

def kthlargest(tensor, k, dim=-1):
    if type(k) is int or k.dim==1:
        val, idx = tensor.topk(int(k), dim = dim)
        return val[:,:,-1], idx[:,:,-1]
    else:
        k = k.view(-1)
        assert k.shape[0] == tensor.shape[0]
        nfts = tensor.shape[-1]
        val, idx = tensor.abs().view(tensor.shape[0], -1).sort(-1)
        topk_curr = torch.clamp(nfts - k, min=0, max=nfts-1).long()

        val = val[torch.arange(tensor.shape[0], device=tensor.device), topk_curr]
        idx = idx[torch.arange(tensor.shape[0], device=tensor.device), topk_curr]
        return val.unsqueeze(-1), idx.unsqueeze(-1)
