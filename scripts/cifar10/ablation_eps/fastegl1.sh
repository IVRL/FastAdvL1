#!/bin/bash
GPU=0

# FastEGL1 on different adversarial budget
EPS=(        3      6      9    12    15     18    21    24)
ST=(         1.73   2.45   3    3.46  3.87   4.24  4.58  4.90)
ST_3quarter=(1.2975 1.8375 2.25 2.59  2.9025 3.18  3.435 3.675)
ST_half=(    0.865  1.225  1.5  1.73  1.935  2.12  2.29  2.45)

# step_size = \sqrt{\alpha}
for i in 0 1 2 3 4 5 6 7
do

eps=${EPS[i]}
st=${ST[i]}

predir="trained/CIFAR10/EPS=${eps}"
savedir="FastEGL1_stepsize=${st}"

python run/train_normal.py \
    --epoch_num 40 \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --optim name=sgd,lr=0.05,momentum=0.9,weight_decay=5e-4,not_wd_bn=True \
    --lr_schedule name=jump,min_jump_pt=30,jump_freq=5,start_v=0.05,power=0.1 \
    --valid_ratio 0.04 \
    --gpu ${GPU} \
    --batch_size 128 \
    --out_folder ${predir}/${savedir} \
    --attack name=FastEGL1,threshold=${eps},iter_num=1,step_size=${st},eps_train=2 \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --n_eval 1000
    
python run/eval_autoattack.py \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --model2load ${predir}/${savedir}/None_bestvalid.ckpt \
    --out_file ${predir}/${savedir}/eval_log.txt \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} 

done


# step_size = 0.75\sqrt{\alpha}
for i in 0 1 2 3 4 5 6 7
do

eps=${EPS[i]}
st=${ST_3quarter[i]}

predir="trained/CIFAR10/EPS=${eps}"
savedir="FastEGL1_stepsize=${st}"

python run/train_normal.py \
    --epoch_num 40 \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --optim name=sgd,lr=0.05,momentum=0.9,weight_decay=5e-4,not_wd_bn=True \
    --lr_schedule name=jump,min_jump_pt=30,jump_freq=5,start_v=0.05,power=0.1 \
    --valid_ratio 0.04 \
    --gpu ${GPU} \
    --batch_size 128 \
    --out_folder ${predir}/${savedir} \
    --attack name=FastEGL1,threshold=${eps},iter_num=1,step_size=${st},eps_train=2 \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --n_eval 1000
    
python run/eval_autoattack.py \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --model2load ${predir}/${savedir}/None_bestvalid.ckpt \
    --out_file ${predir}/${savedir}/eval_log.txt \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} 

done


# step_size = 0.5\sqrt{\alpha}
for i in 0 1 2 3 4 5 6 7
do

eps=${EPS[i]}
st=${ST_half[i]}

predir="trained/CIFAR10/EPS=${eps}"
savedir="FastEGL1_stepsize=${st}"

python run/train_normal.py \
    --epoch_num 40 \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --optim name=sgd,lr=0.05,momentum=0.9,weight_decay=5e-4,not_wd_bn=True \
    --lr_schedule name=jump,min_jump_pt=30,jump_freq=5,start_v=0.05,power=0.1 \
    --valid_ratio 0.04 \
    --gpu ${GPU} \
    --batch_size 128 \
    --out_folder ${predir}/${savedir} \
    --attack name=FastEGL1,threshold=${eps},iter_num=1,step_size=${st},eps_train=2 \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --n_eval 1000
    
python run/eval_autoattack.py \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --model2load ${predir}/${savedir}/None_bestvalid.ckpt \
    --out_file ${predir}/${savedir}/eval_log.txt \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} 

done