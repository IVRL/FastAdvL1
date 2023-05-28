import os
import sys
sys.path.insert(0, './')
import json
import argparse
import time
import copy
import numpy as np
import pickle

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker, init_delta
from util.train import train, epoch_pass, plot_curve
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.data_parser import parse_data
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser
from util.utility import Logger

import pdb

def train_with_minibatch_repaly(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, optimizer,
    lr_func, out_folder, model_name, device, criterion, tosave, logger=None, n_eval=None, minibatch_replays=3, **tricks):
    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    delta_record = torch.zeros(train_loader.batch_size, 3, 32, 32).to(device)
    order, threshold = train_attacker.order, train_attacker.threshold
    for epoch_idx in range(epoch_num):
        ##--- Training phase ----###
        model.train()
        train_loss = []
        train_acc = []
        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):
            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            
            # adjust learning rate
            if lr_func is not None:
                epoch_batch_idx = epoch_idx
                epoch_batch_idx += idx / len(train_loader)
                lr_this_batch = lr_func(epoch_batch_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_batch
            
            # delta = init_delta(data_batch, threshold=threshold, order=order)
            delta = delta_record[:data_batch.size(0)].detach()
            delta.requires_grad = True
            for replay_idx in range(minibatch_replays):
                # forward
                logits = model(data_batch + delta)
                loss = criterion(logits, label_batch)
                acc = logits.argmax(dim=1) == label_batch

                optimizer.zero_grad()
                loss.backward()
                grad = delta.grad.data

                # update delta
                adv_batch = train_attacker.single_step(data_batch, grad)
                delta.data = adv_batch - data_batch

                # updata model
                optimizer.step()
                delta.grad.zero_()
            
            sys.stdout.write('Train - Instance Idx: %d - %.2f%% \r' % (idx, acc.detach().cpu().numpy().mean()))

            train_loss.append(float(loss))
            train_acc.append(float(torch.mean((logits.argmax(dim=1) == label_batch).float())))
            delta_record[:data_batch.size(0)] = delta.data            
        
        if logger:
            logger.log('train loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, np.mean(train_loss), np.mean(train_acc) * 100.), verbose=True)

        tosave['train_loss'][epoch_idx] = np.mean(train_loss)
        tosave['train_acc'][epoch_idx] = np.mean(train_acc)
        
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
                torch.save(model.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
                best_valid_acc = acc_this_epoch

        # Test phase
        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = test_loader,
            is_train = False, epoch_idx = epoch_idx, label = 'test', device = device, lr_func = None, tosave = tosave, logger=logger, n_eval=n_eval)

        if (epoch_idx + 1) in epoch_ckpts:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                        
        # json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))
        if (epoch_idx + 1) % 5 == 0 or (epoch_idx + 1) == epoch_num:
            pickle.dump(tosave, open(os.path.join(out_folder, '%s_info_train.pickle' % model_name), 'wb'))

     
    # torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))
    plot_curve(tosave, epoch_num, out_folder)
    return model, tosave


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The nomralization mode, default is None.')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default is 128.')
    parser.add_argument('--epoch_num', type = int, default = 20,
        help = 'The number of epochs, default is 20.')
    parser.add_argument('--minibatch_replays', type = int, default = 4,
        help = 'The number of replay times in each minibatch, default is 4.')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The checkpoint epoch, default is [].')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default is None.')

    parser.add_argument('--out_folder', type = str, default = 'current',
        help = 'The output folder.')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model.')

    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 5e-4},
        help = 'The optimizer, default is name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4.')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'jump', 'start_v': 1e-1, 'min_jump_pt': 100, 'jump_freq': 50, 'power': 0.1},
        help = 'The learning rate schedule, default is name=jump,min_jump_pt=100,jump_freq=50,start_v=0.1,power=0.1.')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None, use name=h to obtain help messages.')
    parser.add_argument('--test_attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None, use name=h to obtain help messages.')
    parser.add_argument('--n_eval', type=int, default=2000,
        help = 'Number of samples for evaluation')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default is None.')

    args = parser.parse_args()

    t1 = time.localtime()
    print('{}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec))
    
    # Config the GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model and dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, valid_ratio = args.valid_ratio)

    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    # if use multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.to(device)
    criterion = criterion.cuda() if use_gpu else criterion

    if args.model2load is not None:
        ckpt2load = torch.load(args.model2load)
        model.load_state_dict(ckpt2load)

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.named_parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None

    # Parse the attacker
    train_attacker = None if args.attack == None else parse_attacker(**args.attack)
    test_attacker = parse_attacker(**args.attack) if args.test_attack == None else parse_attacker(**args.test_attack)
    # test_attacker= None

    # logger
    logger = Logger(log_path='{}/logger.txt'.format(args.out_folder))

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'label':{}, 'weight':[],
        'train_loss': {}, 'train_acc': {}, 'train_acc_per_instance': {}, 'train_entropy_per_instance': {}, 'train_loss_per_instance': {}, 'train_logits_per_instance':{},
        'valid_loss': {}, 'valid_acc': {}, 'valid_acc_per_instance': {}, 'valid_entropy_per_instance': {}, 'valid_loss_per_instance': {}, 'valid_logits_per_instance': {}, 
        'test_loss': {}, 'test_acc': {}, 'test_acc_per_instance': {}, 'test_entropy_per_instance': {}, 'test_loss_per_instance': {}, 'test_logits_per_instance': {},
        'total_batch_num': {}, 'lr_per_epoch': {}, 'runtime': [], 'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        logger.log('%s\t=>%s' % (param, tosave['setup_config'][param]), verbose=True)

    ##----training----##
    train_with_minibatch_repaly(model = model, train_loader = train_loader, valid_loader = valid_loader, test_loader = test_loader, 
                                train_attacker = train_attacker, test_attacker = test_attacker,
                                epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, minibatch_replays = args.minibatch_replays,
                                optimizer = optimizer, lr_func = lr_func, out_folder = args.out_folder, model_name = args.model_name, 
                                device = device, criterion = criterion, tosave = tosave, logger=logger, n_eval = args.n_eval,
            )

    logger.log('Start time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec), verbose=True)
    t2 = time.localtime()
    logger.log('End time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t2.tm_year, t2.tm_mon, t2.tm_mday, t2.tm_hour, t2.tm_min, t2.tm_sec), verbose=True)