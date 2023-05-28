
import os
import sys
sys.path.insert(0, './')

import time
import json
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .evaluation import *
from .grad_align import grad_align_loss
from util.attack import Gradalign, NuAT

def epoch_pass(model, criterion, attacker, optimizer, loader, is_train, epoch_idx, label, device, lr_func, tosave, 
                logger=None, n_eval=None):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()
    batch_size = loader.batch_size

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    if is_train == True:
        model.train()
    else:
        model.eval()

    for idx, (data_batch, label_batch, idx_batch) in enumerate(loader, 0):
        if is_train == True and lr_func is not None:
            epoch_batch_idx = epoch_idx
            epoch_batch_idx += idx / len(loader)
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learing rate = %1.2e' % lr_this_batch)

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch
        idx_batch = idx_batch.int().data.cpu().numpy()

        if not attacker:
            adv_data_batch, adv_label_batch = data_batch, label_batch
        else:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion, is_train=is_train, idx_batch=idx_batch)
        
        if is_train:
            optimizer.zero_grad()
            
            logits = model(adv_data_batch)
            loss = criterion(logits, adv_label_batch)

            if isinstance(attacker, Gradalign):
                loss += grad_align_loss(model, data_batch, label_batch, 
                            order=attacker.order, threshold=attacker.threshold, grad_align_lambda=attacker.lmbd)

            if isinstance(attacker, NuAT):
                ori_logits = model(data_batch)
                loss += attacker.lmbd * torch.norm(ori_logits - logits, 'nuc') / batch_size

            loss.backward()
            optimizer.step()
        else:
            logits = model(adv_data_batch)
            loss = criterion(logits, adv_label_batch)

        acc = accuracy(logits.data, adv_label_batch)
        _, prediction_this_batch = logits.max(dim = 1)

        loss_calculator.update(loss.item(), data_batch.size(0))
        acc_calculator.update(acc.item(), data_batch.size(0))

        sys.stdout.write('%s - Instance Idx: %d - %.2f%% \r' % (label, idx, acc_calculator.average * 100.))

        if n_eval and (not is_train) and (idx >= n_eval/batch_size - 1):
            break

    loss_this_epoch = loss_calculator.average
    acc_this_epoch = acc_calculator.average

    if logger:
        logger.log('%5s loss / acc after epoch %d: %.4f / %.2f%%' % (label, epoch_idx, loss_this_epoch, acc_this_epoch * 100.))

    print('%s loss / acc after epoch %d: %.4f / %.2f%%' % (label, epoch_idx, loss_this_epoch, acc_this_epoch * 100.))

    tosave['%s_loss' % label][epoch_idx] = loss_this_epoch
    tosave['%s_acc' % label][epoch_idx] = acc_this_epoch

    return model, tosave, loss_this_epoch, acc_this_epoch


def train(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, optimizer,
    lr_func, out_folder, model_name, device, criterion, tosave, logger=None, n_eval=None,  **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    for epoch_idx in range(epoch_num):
        # Training phase
     
        t0 = time.time()
        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = train_attacker, optimizer = optimizer, loader = train_loader,
            is_train = True, epoch_idx = epoch_idx, label = 'train', device = device, lr_func = lr_func, tosave = tosave, logger=logger,)
        t1 = time.time()

        tosave['runtime'].append(round((t1 - t0) / 60, 3))

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave, logger=logger, n_eval=n_eval)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
                # save_model(model, os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                save_model(model, os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
                best_valid_acc = acc_this_epoch

        # Test phase
        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = test_loader,
            is_train = False, epoch_idx = epoch_idx, label = 'test', device = device, lr_func = None, tosave = tosave, logger=logger, n_eval=n_eval)

        if (epoch_idx + 1) in epoch_ckpts:
            # torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
            save_model(model, os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                        
        # json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))
        if (epoch_idx + 1) % 5 == 0 or (epoch_idx + 1) == epoch_num:
              pickle.dump(tosave, open(os.path.join(out_folder, '%s_info_train.pickle' % model_name), 'wb'))
    
    # torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))
    plot_curve(tosave, epoch_num, out_folder)
    return model, tosave

def save_model(model, dir, optimizer=None):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), dir)
    elif optimizer:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    dir)
    else:
        torch.save(model.state_dict(), dir)


def plot_curve(tosave, epoch_num, out_folder):
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(np.arange(epoch_num), tosave['train_loss'].values())
    plt.plot(np.arange(epoch_num), tosave['valid_loss'].values())
    plt.plot(np.arange(epoch_num), tosave['test_loss'].values())
    if 'test_ema_loss' in tosave.keys():
        plt.plot(np.arange(epoch_num), tosave['test_ema_loss'].values(), '--')
        plt.legend(['train_loss', 'valid_loss', 'test_loss', 'test_ema_loss'])
    else:
        plt.legend(['train_loss', 'valid_loss', 'test_loss'])
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(np.arange(epoch_num), [100*v for v in tosave['train_acc'].values()])
    plt.plot(np.arange(epoch_num), [100*v for v in tosave['valid_acc'].values()])
    plt.plot(np.arange(epoch_num), [100*v for v in tosave['test_acc'].values()])
    if 'test_ema_acc' in tosave.keys():
        plt.plot(np.arange(epoch_num), [100*v for v in tosave['test_ema_acc'].values()], '--')
        plt.legend(['train_acc', 'valid_acc', 'test_acc', 'test_ema_acc'])
    else:
        plt.legend(['train_acc', 'valid_acc', 'test_acc'])
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(out_folder, 'training_curve.png'))
  