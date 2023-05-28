import os
import sys
sys.path.insert(0, './')
import json
import argparse
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker
from util.train import train
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.data_parser import parse_data
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser
from util.utility import Logger

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
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'The number of epochs, default is 200.')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The checkpoint epoch, default is [].')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--ckpt2load', type = str, default = None,
        help = 'The model to be loaded, default is None.')

    parser.add_argument('--out_folder', type = str, default = 'current',
        help = 'The output folder.')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model.')

    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 5e-4},
        help = 'The optimizer, default is name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4.')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'jump', 'start_v': 1e-1, 'min_jump_pt': 100, 'jump_freq': 50, 'power': 0.1},
        help = 'The learning rate schedule, default is name=jump,min_jump_pt=100,jump_freq=50,start_v=0.1,power=0.1.')
    parser.add_argument('--eps_schedule', action = DictParser, default = None,
        help = 'The scheduler of the adversarial budget, default is None')

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

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.named_parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None
    eps_func = continuous_seq(**args.eps_schedule) if args.eps_schedule != None else None

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


    train(model = model, train_loader = train_loader, valid_loader = valid_loader, test_loader = test_loader, train_attacker = train_attacker, test_attacker = test_attacker,
            epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, optimizer = optimizer, lr_func = lr_func, out_folder = args.out_folder, model_name = args.model_name, 
            device = device, criterion = criterion, tosave = tosave, logger=logger, n_eval = args.n_eval,
            )

    logger.log('Start time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec), verbose=True)
    t2 = time.localtime()
    logger.log('End time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t2.tm_year, t2.tm_mon, t2.tm_mday, t2.tm_hour, t2.tm_min, t2.tm_sec), verbose=True)

    logger.log('Runtime per epoch in minutes \n')
    logger.log(str(tosave['runtime']))