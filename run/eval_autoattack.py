import os
import sys
sys.path.insert(0, './')

import json
import pickle
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime
import pdb

from util.attack import parse_attacker
from util.data_parser import parse_data
from util.model_parser import parse_model
from util.param_parser import DictParser, BooleanParser
from util.device_parser import config_visible_gpu
from util.autopgd_train import apgd_train
from arch.fast_models import PreActResNet18

from external.autoattack import AutoAttack

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The normalization mode, default is None.')
    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'The batch size, default = 100.')
    parser.add_argument('--n_eval', type=int, default=10000,
        help = 'Number of samples for evaluation')

    parser.add_argument('--subset', type = str, default = 'test',
        help = 'Specify which set is used, default = "test".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default = None.')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output direaction, default is None, meaning no file to save.')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'The adversarial attack, default is None.')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--partial_eval', action='store_true', 
        help = 'Run partial_eval evaluation to save time during training')

    parser.add_argument('--use_train', action = BooleanParser, default = False,
        help = 'Whether or not to use the training mode, default is False.')
    
    parser.add_argument('--save_adv_dataset', action='store_true')
    parser.add_argument('--result_log_path', type=str, default=None,
        help = 'Write the result into txt file')
    parser.add_argument('--model_name', type=str, default=None,
        help = 'Model name to log')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default is None.')

    args = parser.parse_args()

    # Config the GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    print(out_dir)
    # Parse model and dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, shuffle = False)
    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize)

    loader = {'train': train_loader, 'test': test_loader}[args.subset]
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    if args.model2load is not None:
        print('loading model')
        ckpt2load = torch.load(args.model2load)
        if 'state_dict' in ckpt2load.keys():
            ckpt2load = ckpt2load['state_dict']
        try:
            model.load_state_dict(ckpt2load)
        except:
            ckpt = {}
            for k,v in ckpt2load.items():
                ckpt['1.' + k] = v
            model.load_state_dict(ckpt)

    model.eval()

    n_eval = min(args.n_eval, 10000)
    # create auto attack
    order_to_norm = {'-1': 'Linf', -1.: 'Linf', '1':'L1', 1.:'L1', 2.:'L2', '2':'L2'}
    print(args.attack)
    if args.attack['name'] in ['apgd', 'pgd', 'single', 'apgd_single']:
        # adversary = AutoAttack(model, norm=order_to_norm[args.attack['order']], eps=args.attack['threshold'], log_path=args.out_file,
        #     version=args.version)
        if args.partial_eval:
            adversary = AutoAttack(model, norm=order_to_norm[args.attack['order']], eps=args.attack['threshold'], log_path=args.out_file,
                version='custom', attacks_to_run=['apgd-ce', 'apgd-t'])
        else:
            # adversary = AutoAttack(model, norm=order_to_norm[args.attack['order']], eps=args.attack['threshold'], log_path=args.out_file,
            #     version='custom', attacks_to_run=['apgd-ce', 'apgd-t', 'fab-t'])
            adversary = AutoAttack(model, norm=order_to_norm[args.attack['order']], eps=args.attack['threshold'], log_path=args.out_file,
            version=args.version)
                    
        ### Evaluate AutoAttack ###
        l = [x for (x, y, idx) in test_loader]
        x_test = torch.cat(l, 0)[:n_eval]
        l = [y for (x, y, idx) in test_loader]
        y_test = torch.cat(l, 0)[:n_eval]

        # adversary.apgd.n_iter = 20
        # adversary.apgd_targeted.n_iter = 20
        X_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)

        result = y_adv == y_test
        np.save(args.out_file[:args.out_file.rfind('/')]+'/autoattack.npy', result.numpy())

        if args.result_log_path is not None:
            clean_acc = adversary.clean_accuracy(x_test, y_test)
            adv_acc = float(result.numpy().mean())

            with open(args.result_log_path, 'a') as f:
                t = time.localtime()
                # curr_time = '{}-{}-{}, {:02d}:{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
                f.write('Model %s, clean_acc %.4f, adv_acc %.4f' % (args.model_name, clean_acc, adv_acc) + ' \n')
                f.flush()

    elif args.attack['name'] in ['pgd_list', 'apgd_list']:
        l = [x for (x, y, idx) in test_loader]
        x_test = torch.cat(l, 0)[:n_eval]
        l = [y for (x, y, idx) in test_loader]
        y_test = torch.cat(l, 0)[:n_eval]

        results = []
        num_attacks = len(args.attack['order'])
        for i in range(num_attacks):
            order = order_to_norm[args.attack['order'][i]]
            threshold = args.attack['threshold'][i]
            adversary = AutoAttack(model, norm=order, eps=threshold, log_path=args.out_file,
                version='custom', attacks_to_run=['apgd-ce', 'apgd-t'])
            adversary.logger.log('start to evaluate attack order: {}, threshold: {}'.format(order, threshold))

            X_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)
            
            result = y_adv == y_test
            results.append(result.numpy())

        results = np.array(results)
        np.save(args.out_file[:args.out_file.rfind('/')]+'/autoattack.npy', result.numpy())
        adversary.logger.log('Final results:')
        for i in range(num_attacks):
            order = order_to_norm[args.attack['order'][i]]
            threshold = args.attack['threshold'][i]
            adversary.logger.log('Order: {}, threshold: {}, accuracy: {}'.format(order, threshold, results[i].mean()))
        adversary.logger.log('Union accuracy: {}'.format(results.min(axis=0).mean()))

    else:
        raise 'invalid attack type!'

    