import os
import sys
sys.path.insert(0, './')

import time
import numpy as np
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utility import project

# def init_delta(x, order, threshold):
#     eps = threshold

#     if order in [-1, np.inf]:
#         delta = np.random.uniform(-eps, eps, x.shape)
#         return torch.tensor(delta, dtype=torch.float32).to(x.device)
#     elif order == 2:
#         r = np.random.randn(*x.shape)
#         norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
#         delta = (r / norm) * eps
#         return torch.tensor(delta, dtype=torch.float32).to(x.device)
#     elif order == 1:
#         r = np.random.laplace(size=x.shape)
#         norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1, ord=1).reshape(-1, 1, 1, 1)
#         delta = (r / norm) * eps
#         return torch.tensor(delta, dtype=torch.float32).to(x.device)
#     else:
#         raise ValueError("Unknown norm {}".format(order))

def init_delta(x, order, threshold):
    ori_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    N, ndim = x.shape

    if order in [-1, np.inf]:
        delta = np.random.uniform(-threshold, threshold, ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    elif order == 2:
        r = np.random.randn(*x.shape)
        norm = np.linalg.norm(r, axis=-1, keepdims=True)
        delta = (r / norm) * threshold * np.random.uniform(size=x.shape) ** (1/ndim)
        delta = delta.reshape(ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    elif order == 1:
        r = np.random.laplace(size=x.shape)
        norm = np.linalg.norm(r, axis=-1, ord=1, keepdims=True)
        delta = (r / norm) * threshold * np.random.uniform(size=x.shape) ** (1/ndim)
        delta = delta.reshape(ori_shape)
        return torch.tensor(delta, dtype=torch.float32).to(x.device)
    else:
        raise ValueError("Unknown norm {}".format(order))


def get_input_grad(model, X, y, order, threshold, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    else:
        delta = init_delta(X, order, threshold)
        delta.requires_grad_()

    output = model(X + delta)
    loss = F.cross_entropy(output, y)

    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad

def grad_align_loss(model, X, y, order, threshold, grad_align_lambda=0.2):
    grad1 = get_input_grad(model, X, y, order, threshold, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y, order, threshold, backprop=True)

    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
    reg = grad_align_lambda * (1.0 - cos.mean())

    return reg

def get_reg_loss(model, X, y, order, threshold, grad_align_lambda=0.2, theme='random'):
    if type(order) is not list:
        return grad_align_loss(model, X, y, order, threshold, grad_align_lambda)
    elif theme == 'random':
        K = len(threshold)
        k = random.randint(0, K)
        return grad_align_loss(model, X, y, order[k], threshold[k], grad_align_lambda)
    elif theme == 'average':
        K = len(threshold)
        reg = 0
        for k in range(K):
            reg += grad_align_loss(model, X, y, order[k], threshold[k], grad_align_lambda)

        return reg / K
    else:
        return 0