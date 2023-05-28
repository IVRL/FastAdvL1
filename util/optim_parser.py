# parse the optimizer configuration

import torch
import torch.nn as nn
import numpy as np

instructions = '''
instructions for setting an optimizer
>>> SGD
name=sgd,lr=$LR$,momentum=$0.9$,dampening=$0$,weight_decay=$0$

>>> Adagrad
name=adagrad,lr=$LR$,lr_decay=$0$,weight_decay=$0$

>>> Adadelta
name=adadelta,lr=$LR$,rho=$0.9$,eps=$1e-6$,weight_decay=$0$

>>> Adam
name=adam,lr=$LR$,beta1=$0.9$,beta2=$0.999$,eps=$1e-8$,weight_decay=$0$,amsgrad=$0$

>>> RMSprop
name=rmsprop,lr=$LR$,alpha=$0.99$,eps=$1e-8$,weight_decay=$0$,momentum=$0$
'''

def parse_optim(policy, params):

    kwargs = {}
    if 'weight_decay' in policy.keys() and 'not_wd_bn' in policy.keys() and policy['not_wd_bn']:
        decay, no_decay = [], []
        for name, param in params:
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': policy['weight_decay']},
                  {'params': no_decay, 'weight_decay': 0}]
        print('not using wd for bn layers')
    else:
        params = [param for name, param in params]

    if policy['name'].lower() in ['sgd']:

        kwargs['lr'] = policy['lr']
        kwargs['momentum'] = policy['momentum'] if 'momentum' in policy else 0.9
        kwargs['dampening'] = policy['dampening'] if 'dampening' in policy else 0
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0
        optimizer = torch.optim.SGD(params, **kwargs)

    elif policy['name'].lower() in ['extrasgd']:
        kwargs['lr'] = policy['lr']
        kwargs['momentum'] = policy['momentum'] if 'momentum' in policy else 0.9
        kwargs['dampening'] = policy['dampening'] if 'dampening' in policy else 0
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0
        optimizer = ExtraSGD(params, **kwargs)

    elif policy['name'].lower() in ['adagrad']:
        kwargs['lr'] = policy['lr']
        kwargs['lr_decay'] = policy['lr_decay'] if 'lr_decay' in policy else 0.
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        optimizer = torch.optim.Adagrad(params, **kwargs)

    elif policy['name'].lower() in ['adadelta']:

        kwargs['lr'] = policy['lr']
        kwargs['rho'] = policy['rho'] if 'rho' in policy else 0.9
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-6
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        optimizer = torch.optim.Adadelta(params, **kwargs)

    elif policy['name'].lower() in ['adam']:

        kwargs['lr'] = policy['lr']
        kwargs['betas'] = (policy['beta1'] if 'beta1' in policy else 0.9, policy['beta2'] if 'beta2' in policy else 0.999)
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-8
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        kwargs['amsgrad'] = True if 'amsgrad' in policy and np.abs(policy['amsgrad']) > 1e-6 else False
        optimizer = torch.optim.Adam(params, **kwargs)

    elif policy['name'].lower() in ['rmsprop']:

        kwargs['lr'] = policy['lr']
        kwargs['alpha'] = policy['alpha'] if 'alpha' in policy else 0.99
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-8
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        kwargs['momentum'] = policy['momentum'] if 'momentum' in policy else 0.
        optimizer = torch.optim.RMSprop(params, **kwargs)

    elif policy['name'].lower() in ['h', 'help']:

        print(instructions)
        exit(0)

    else:
        raise ValueError('Unrecognized policy: %s'%policy)

    print('Optimizer : %s --'%policy['name'])
    for key in kwargs:
        print('%s: %s'%(key, kwargs[key]))
    print('-----------------')

    return optimizer

def set_optim_params(optim, params):

    if isinstance(params, torch.Tensor):
        raise TypeError("params argument given to the optimizer should be an iterable of Tensors or dicts, but got " + torch.typename(params))

    param_groups = list(params)
    if len(param_groups) == 0:
        raise ValueError('optimizer got an empty parameter list')
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    optim.param_groups = []
    for param_group in param_groups:
        optim.add_param_group(param_group)

    return optim