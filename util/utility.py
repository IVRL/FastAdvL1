import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

import pdb

## Tensor Operation
def group_add(x1_list, mul1, x2_list, mul2):
    '''
    >>> group summation: x1 * mul1 + x2 * mul2
    '''

    return [x1 * mul1 + x2 * mul2 for x1, x2 in zip(x1_list, x2_list)]

def group_product(x1_list, x2_list):
    '''
    >>> x1_list, x2_list: the list of tensors to be multiplied

    >>> group dot product
    '''

    return sum([torch.sum(x1 * x2) for x1, x2 in zip(x1_list, x2_list)])

def group_normalize(v_list):
    '''
    >>> normalize the tensor list to make them joint l2 norm be 1
    '''

    summation = group_product(v_list, v_list)
    summation = summation ** 0.5
    v_list = [v / (summation + 1e-6) for v in v_list]

    return v_list

def get_param(model):
    '''
    >>> return the parameter list
    '''
    param_list = []
    for param_name, param in model.named_parameters():
        param_list.append(param.data)
    return param_list

def get_param_grad(model, with_grad_fn=False):
    '''
    >>> return the parameter and gradient list
    '''
    param_list = []
    grad_list = []
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if with_grad_fn:
                param_list.append(param)
                grad_list.append(param.grad)
            else:
                param_list.append(param.data)
                grad_list.append(param.grad.data)
    return param_list, grad_list

def calc_gap(value_batch, label_batch):
    '''
    >>> calculate the gap between the true outputs and biggest false outputs
    '''
    value, idx = value_batch.topk(dim = 1, k = 2)
    correct_bit = (label_batch == idx[:, 0]).float()
    true_logits = value_batch.gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
    false_logits = value[:, 0] * (1. - correct_bit) + value[:, 1] * correct_bit
    return true_logits - false_logits

## Model Operation
def distance_between_ckpts(ckpt1, ckpt2):
    '''
    >>> Calculate the distance ckpt2 - ckpt1
    '''

    assert len(ckpt1) == len(ckpt2), 'The length of ckpt1 should be the same as ckpt2'
    key_list = ckpt1.keys()

    distance_dict = OrderedDict()
    for key in key_list:
        param1 = ckpt1[key]
        param2 = ckpt2[key]
        distance_dict[key] = param2.data - param1.data

    return distance_dict

## Project to an adversarial budget
def project_(ori_pt, threshold, order = np.inf):

    if order in [np.inf,]:
        prj_pt = torch.clamp(ori_pt, min = - threshold, max = threshold) 
    elif order in [2,]:
        ori_shape = ori_pt.size()
        pt_norm = torch.norm(ori_pt.view(ori_shape[0], -1), dim = 1, p = 2)
        pt_norm_clip = torch.clamp(pt_norm, max = threshold)
        prj_pt = ori_pt.view(ori_shape[0], -1) / (pt_norm.view(-1, 1) + 1e-8) * (pt_norm_clip.view(-1, 1) + 1e-8)
        prj_pt = prj_pt.view(ori_shape)
    elif order in [1,]:
        ori_shape = ori_pt.size()
        # ori_shape_2d = ori_shape.view(ori_shape[0], -1)
        abs_pt = ori_pt.view(ori_shape[0], -1).abs()
        if (abs_pt.sum(dim = 1) <= threshold).all():
            prj_pt = ori_pt
        else:
            prj_pt = proj_simplex(abs_pt, threshold)
            prj_pt = prj_pt.view(ori_shape)
            prj_pt *= ori_pt.sign()
    else:
        raise ValueError('Invalid norms: %s' % order)

    return prj_pt


def proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    assert v.dim() == 2, "input should be resized into 2-d vector"
    batch_size = v.shape[0]
    device = v.device
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    # vec = u * torch.arange(1, n+1).half().to(device)
    # comp = (vec > (cssv - s)).half()
    vec = u * torch.arange(1, n+1).to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    # c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = torch.Tensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    # theta = torch.div(c,(rho.half() + 1))
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def project(ori_pt, threshold, order = np.inf):
    '''
    Project the data into a norm ball

    >>> ori_pt: the original point
    >>> threshold: maximum norms allowed, can be float or list of float
    >>> order: norm used
    '''
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.reshape(-1).tolist()

    if isinstance(threshold, (int, float)):
        prj_pt = project_(ori_pt, threshold, order)    
    elif isinstance(threshold, (list, tuple)):
        if list(set(threshold)).__len__() == 1:
            prj_pt = project_(ori_pt, threshold[0], order)
        else:
            assert len(threshold) == ori_pt.size(0)
            prj_pt = torch.zeros_like(ori_pt)
            for idx, threshold_ in enumerate(threshold):
                prj_pt[idx: idx+1] = project_(ori_pt[idx: idx+1], threshold_, order)
    else:
        raise ValueError('Invalid value of threshold: %s' % threshold)

    return prj_pt


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log, verbose=False):
        if verbose:
            print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
