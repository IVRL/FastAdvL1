"""
This file trains ATTA models on CIFAR10 dataset.
"""
import os
import sys
sys.path.insert(0, './')
import argparse
import time
import json
import pdb
import copy
import random

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# from adaptive_data_aug import atta_aug, atta_aug_trans, inverse_atta_aug
from util.attack import PGD, parse_attacker
from util.param_parser import DictParser
from arch.fast_models import PreActResNet18
from util.utility import Logger

# import cifar_dataloader
# import adv_attack


###----- code in adaptive_data_aug.py ----######
def atta_aug(input_tensor, rst):
  batch_size = input_tensor.shape[0]
  x = torch.zeros(batch_size)
  y = torch.zeros(batch_size)
  flip = [False] * batch_size

  for i in range(batch_size):
    flip_t = bool(random.getrandbits(1))
    x_t = random.randint(0,8)
    y_t = random.randint(0,8)

    rst[i,:,:,:] = input_tensor[i,:,x_t:x_t+32,y_t:y_t+32]
    if flip_t:
      rst[i] = torch.flip(rst[i], [2])
    flip[i] = flip_t
    x[i] = x_t
    y[i] = y_t

  return rst, {"crop":{'x':x, 'y':y}, "flipped":flip}

def atta_aug_trans(input_tensor, transform_info, rst):
  batch_size = input_tensor.shape[0]
  x = transform_info['crop']['x']
  y = transform_info['crop']['y']
  flip = transform_info['flipped']
  for i in range(batch_size):
    flip_t = int(flip[i])
    x_t = int(x[i])
    y_t = int(y[i])
    rst[i,:,:,:] = input_tensor[i,:,x_t:x_t+32,y_t:y_t+32]
    if flip_t:
      rst[i] = torch.flip(rst[i], [2])
  return rst

#Apply random crop and flip to the input
#Input: 3-dim tensor [batchsize, 3, 40, 40], 3-dim tensor [batchsize, 3, 32, 32], transform information {{x,y}, flip}
#Output: 3-dim tensor [batchsize, 3, 40, 40]
def inverse_atta_aug(source_tensor, adv_tensor, transform_info):
  x = transform_info['crop']['x']
  y = transform_info['crop']['y']
  flipped = transform_info['flipped']
  batch_size = source_tensor.shape[0]

  for i in range(batch_size):
    flip_t = int(flipped[i])
    x_t = int(x[i])
    y_t = int(y[i])
    if flip_t:
      adv_tensor[i] = torch.flip(adv_tensor[i], [2])
    source_tensor[i,:,x_t:x_t+32,y_t:y_t+32] = adv_tensor[i]

  return source_tensor

###----- code in cifar_dataloader.py ----######
def load_pading_training_data(device, dataset='cifar10'):
  transform_padding = transforms.Compose([
    transforms.Pad(padding=4),
    transforms.ToTensor(),
  ])
  if dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_padding)
  else:
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_padding)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True)

  for batch_idx, (data, target) in enumerate(train_loader):
        cifar_images, cifar_labels = data.to(device), target.to(device)
  cifar_images[:,:,:4,:] = 0.5
  cifar_images[:,:,-4:,:] = 0.5
  cifar_images[:,:,:,:4] = 0.5
  cifar_images[:,:,:,-4:] = 0.5

  logger.log('Load %s dataset' % dataset, verbose=True)
  return cifar_images, cifar_labels

def load_test_data(device, dataset='cifar10'):
  transform_padding = transforms.Compose([
    transforms.ToTensor(),
  ])
  if dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_padding)
  else:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_padding)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=False)

  for batch_idx, (data, target) in enumerate(train_loader):
        cifar_images, cifar_labels = data.to(device), target.to(device)

  logger.log('Load %s dataset' % dataset, verbose=True)
  return cifar_images, cifar_labels

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                    help='Dataset of the experiments, default cifar10')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs-reset', type=int, default=10, metavar='N',
                    help='number of epochs to reset perturbation')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')

parser.add_argument('--training-method', default='mat',
                    help='Adversarial training method: mat or trades')

parser.add_argument('--attack', action = DictParser, default = None,
        help = 'The adversarial attack, default is None.')
parser.add_argument('--test_attack', action = DictParser, default = None,
        help = 'The adversarial attack, default is None.')
parser.add_argument('--grad_align_lambda', type = float, default = 0,
        help = 'Coeffient for the cosine gradient alignment regularization')

parser.add_argument('--rs', action = 'store_true', default = False,
        help = 'Random start for attack in the first epoch')

parser.add_argument('--config-file',
                    help='The path of config file.')


args = parser.parse_args()

if (args.config_file is None):
    pass
else:
    with open(args.config_file) as config_file:
        config = json.load(config_file)
        args.model_dir = config['model-dir']
        args.num_steps = config['num-steps']
        args.step_size = config['step-size']
        args.epochs_reset = config['epochs-reset']
        args.epsilon = config['epsilon']
        args.beta = config['beta']
        args.training_method = config['training-method']

epochs_reset = args.epochs_reset
training_method = args.training_method
beta = args.beta

#Config file will overlap commend line args
GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print('create direction ', model_dir)
else:
    print('dir exist', model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

logger = Logger(log_path='{}/logger.txt'.format(args.model_dir))
logger.log(str(args))
# setup data loader

def train(args, model, device, cifar_nat_x, cifar_x, cifar_y, optimizer, epoch, attacker=None, rs=False):
    model.train()
    num_of_example = 50000
    batch_size = args.batch_size
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size

    loss_this_epoch = 0
    acc_this_epoch = 0
    for i in range(iter_num):
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        x_batch = cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].cuda()
        x_nat_batch = cifar_nat_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].cuda()
        y_batch = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].cuda()

        batch_size = y_batch.shape[0]

        #atta-aug
        rst = torch.zeros(batch_size,3,32,32).to(device)
        x_batch, transform_info = atta_aug(x_batch, rst)
        rst = torch.zeros(batch_size,3,32,32).to(device)
        x_nat_batch = atta_aug_trans(x_nat_batch, transform_info, rst)

        criterion = nn.CrossEntropyLoss().cuda()
        random_start = (rs and epoch == 1)
            
        x_adv_next, y_adv_next = attacker.attack(model, x_batch, y_batch, criterion=criterion, random_start=random_start, ori_batch=x_nat_batch, is_train=True)     
            
        model.zero_grad()
        logits = model(x_adv_next)

        loss = criterion(logits, y_batch)
        acc = logits.argmax(dim=1) == y_batch

        loss_this_epoch += float(loss)
        acc_this_epoch += float(acc.detach().cpu().numpy().mean())

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.3f}'.format(
                epoch, batch_idx, num_of_example,
                       100. * batch_idx / num_of_example, loss.item(), acc.detach().cpu().numpy().mean()))

            if attacker.order in [-1, np.inf]:
                print("%.4f, %.4f" % (float(torch.min(x_adv_next - x_nat_batch)), float(torch.max(x_adv_next - x_nat_batch))))
            else:
                adv_l1_norm = torch.norm(x_adv_next - x_nat_batch, p=1, dim=(1,2,3))
                print('%.4f, %.4f' % (adv_l1_norm.min(), adv_l1_norm.max()))

        cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = inverse_atta_aug(
            cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].detach().cpu(),
            x_adv_next.detach().cpu(), transform_info)

    if attacker.order in [-1, np.inf]:
        logger.log("min max perturbation %.4f, %.4f" % (float(torch.min(x_adv_next - x_nat_batch)), float(torch.max(x_adv_next - x_nat_batch))))
    else:
        adv_l1_norm = torch.norm(x_adv_next - x_nat_batch, p=1, dim=(1,2,3))
        logger.log('min max perturbation %.4f, %.4f' % (adv_l1_norm.min(), adv_l1_norm.max()))

        
    return loss_this_epoch / (i+1), acc_this_epoch / (i+1)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 36:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # model = WideResNet().to(device)
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = PreActResNet18(n_cls=num_classes, activation='softplus1').to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cifar_x, cifar_y = load_pading_training_data(device='cpu', dataset=args.dataset)
    cifar_nat_x = cifar_x.clone()
    cifar_x = cifar_x.detach() + 0.001 * torch.randn(cifar_x.shape).detach()

    test_x, test_y = load_test_data(device='cpu', dataset=args.dataset)
    criterion = nn.CrossEntropyLoss()

    attacker = None if args.attack == None else parse_attacker(**args.attack)
    test_attacker = parse_attacker(**args.attack) if args.test_attack == None else parse_attacker(**args.test_attack)

    record = {'train_loss': [],
              'train_acc': [],
              'test_loss': [],
              'test_acc': [],
              'perc': []}

    best_acc = 0
    best_ckpt = None
    best_ep = 0
    sparsity_record = "Sparsity\n \tmin, \t25%, \t50%, \t75%, \tmax \n"
    runtime = []
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        #reset perturbation
        if epoch % epochs_reset == 0:
            cifar_x = cifar_nat_x.clone()
            cifar_x = cifar_x.detach() + 0.001 * torch.randn(cifar_x.shape).detach()
        time0 = time.time()
        train_loss, train_acc = train(args, model, device, cifar_nat_x, cifar_x, cifar_y, optimizer, epoch, attacker=attacker, rs=args.rs)
        time1 = time.time()
        runtime.append(round((time1 - time0) / 60, 3))
        logger.log('Epoch {}, train loss {:.3f}, acc {:.3f}'.format(epoch, train_loss, train_acc), verbose=True)

        # evaluation on natural examples
        print('================================================================')
        loss = 0
        acc = 0
        indexs = torch.randperm(10000)
        sparsity = []
        for k in range(10):
            idx = indexs[k*100:(k+1)*100]
            x, y= test_x[idx].to(device), test_y[idx].to(device)
            x_adv, y_adv = test_attacker.attack(model, x, y, criterion)
            
            logits = model(x_adv)
            loss += float(criterion(logits, y))
            acc += (logits.argmax(dim=1) == y).detach().cpu().numpy().mean()
            sparsity.append(torch.norm(x_adv - x, p=0, dim=(1,2,3)).detach().cpu().numpy())

        loss = float(loss) / (k+1)
        acc = float(acc) / (k+1)
        logger.log('Epoch {},  test loss {:.3f}, acc {:.3f}'.format(epoch, loss, acc), verbose=True)
        perc = np.percentile(np.concatenate(sparsity, axis=0), [0, 25, 50, 75, 100])
        sparsity_record += '\t{:.1f} \t{:.1f} \t{:.1f} \t{:.1f} \t{:.1f} \n'.format(perc[0], perc[1], perc[2], perc[3], perc[4])

        if acc > best_acc:
            best_acc = acc
            best_ckpt = copy.deepcopy(model.state_dict())
            best_ep = epoch

        record['train_loss'].append(train_loss)
        record['train_acc'].append(train_acc)
        record['test_loss'].append(loss)
        record['test_acc'].append(acc)
        record['perc'].append(perc)
        # save checkpoint
        if epoch % args.save_freq == 0 or epoch in [40]:
            torch.save(model.state_dict(), os.path.join(model_dir, 'ep_{}.ckpt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-cifar-epoch{}.tar'.format(epoch)))

    logger.log('Epoch {} has best test accuracy {:.3f}'.format(best_ep, best_acc))
    logger.log(sparsity_record)
    logger.log('Run time each epoch: \n')
    logger.log(str(runtime))
    torch.save(best_ckpt, os.path.join(model_dir, 'ep_bestvalid.ckpt'))

    np.save(os.path.join(model_dir, 'training_record.npy'), record)
    epoch_num = args.epochs
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(np.arange(epoch_num), record['train_loss'])
    plt.plot(np.arange(epoch_num), record['test_loss'])
    plt.legend(['train_loss', 'test_loss'])
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(np.arange(epoch_num), [100*v for v in record['train_acc']])
    plt.plot(np.arange(epoch_num), [100*v for v in record['test_acc']])
    plt.legend(['train_acc', 'test_acc'])
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(model_dir, 'training_curve.png'))

if __name__ == '__main__':
    t1 = time.localtime()
    main()

    logger.log('Start time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec), verbose=True)
    t2 = time.localtime()
    logger.log('End time, {}-{}-{}, {:02d}:{:02d}:{:02d}'.format(t2.tm_year, t2.tm_mon, t2.tm_mday, t2.tm_hour, t2.tm_min, t2.tm_sec), verbose=True)
