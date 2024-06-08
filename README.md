## Introduction

This repository contrains code for paper:

**Towards Stable and Efficient Adversarial Training against $l_1$ Bounded Adversarial Attacks** 

*Yulun Jiang\* and Chen Liu\* and Zhichao Huang and Mathieu Salzmann and Sabine Süsstrunk.* 

International Conference on Machine Learning (ICML) 2023.

*i)* We show adversarial training against $l_1$-norm bounded attacks is more likely to suffer from catastrophic overfitting, since existing method with coordinate descent incurs a strong bias towards generating sparse perturbations. *ii)* We address this problem by Fast-EG-$l_1$, an efficient (single-step) adversarial training algorithm based on Euclidean geometry.

### Overview

* `arch`: different model architectures used in this paper
* `dataset`: loaders of different datasets (CIFAR10, CIFAR100, ImageNet100)
* `external/autoattack`: AutoAttack imported from [Croce et al, ICML 2020](http://github.com/fra31/auto-attack)
* `util`: parsers, attacks, supporting functions
    * `util/attack.py`: Code for PGD, Fast-EG-l1 and other attacks 
    * `util/train.py`: Code for training the model
* `run`: code to run experiments
    * `run/train_normal.py`: adversarial training
    * `run/eval_autoattack.py`: evaluate model on AutoAttack
* `scripts`: shell scripts for all experiments in this paper
    * `scripts/cifar10`: experiments on CIFAR10
    * `scripts/cifar100`: experiments on CIFAR100
    * `scripts/imagenet100`: experiments on ImageNet100

### Packages

To run the code in this repository, be sure to install the following packages:
```
numpy
pytorch
torchvision
matplotlib
seaborn
tqdm
```

[ImageNet100](https://github.com/Continvvm/continuum/blob/838ad2ba3571f1563627301c30152c0f07d3cffa/continuum/datasets/imagenet.py#L44) is a subset of ImageNet1k with only 100 classes. For experiments on ImageNet100, please prepare the dataset with the following instructions:

1. Download [ImageNet dataset](https://image-net.org/). 
2. Extract the dataset with [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).
3. Download ['train_100.txt'](https://github.com/Continvvm/continuum/releases/download/v0.1/train_100.txt) and ['val_100.txt'](https://github.com/Continvvm/continuum/releases/download/v0.1/val_100.txt).
4. Specify the path of above downloaded files in ```dataset/format_imagenet100.py```, run the script.
5. Specify the 'data_path' in ```dataset/imagenet100.py```.

### Quick start

Use ```run/train_normal.py``` to train a model, specifty dataset at ```--dataset```, model architecture at ```--model_type```, attack type at ```--attack``` and output direction at ```--savedir```. We recommand reading source code for more details about parameter settings.

Below is an example of training PreactResNet18 on CIFAR10 with FastEGL1 attack：
```
python run/train_normal.py \
    --dataset cifar10 \
    --epoch_num 40 \
    --model_type preact_resnet_softplus \
    --optim name=sgd,lr=0.05,momentum=0.9,weight_decay=5e-4,not_wd_bn=True \
    --lr_schedule name=jump,min_jump_pt=30,jump_freq=5,start_v=0.05,power=0.1 \
    --valid_ratio 0.04 \
    --gpu ${GPU} \
    --batch_size 128 \
    --out_folder ${savedir} \
    --attack name=FastEGL1,threshold=${eps},iter_num=1,step_size=${st},eps_train=2 \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --n_eval 1000
```

Use ```run/eval_autoattack.py``` to evaluate a trained model against AutoAttack. Here is an example of evaluating the trained model on CIFAR10:
```
python run/eval_autoattack.py \
    --dataset cifar10 \
    --model_type preact_resnet_softplus \
    --model2load ${savedir}/None_bestvalid.ckpt \
    --out_file ${savedir}/eval_log.txt \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} 
```

For replicating experiments in this paper, please consider running the scripts file for each dataset at ```scripts```, for example:
```
sh scripts/cifar10/fastegl1.sh
```

### Checkpoints
We provide checkpoints of our trained model below, each model is trained for *40* epochs on CIFAR10 and CIFAR100, *25* epochs on ImageNet100 (*eps* denotes the radius of adversarial budget):

| Dataset | Model | Method Name | Clean Accuracy (\%) | Robust Accuracy (\%) | Checkpoint |
|---|---|---|---|---|---|
| CIFAR10 (eps=12) | PreactResNet18  | FastEGL1       | 76.14 | 50.27 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRolHt8LTXSZZmcQ?e=8fDMXS) |
| CIFAR10 (eps=12) | PreactResNet18  | FastEGL1+NuAT  | 73.51 | 51.37 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRkqEQULIGVOU1sH?e=mJ7GJ7) |
| CIFAR100 (eps=6) | PreactResNet18  | FastEGL1       | 59.43 | 38.03 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRgQO7vTNWYjXKDu?e=K0xm2Y) |
| CIFAR100 (eps=6) | PreactResNet18  | FastEGL1+NuAT  | 58.50 | 39.75 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRfP_bQa6A9CEXL7?e=AvRgGY) |
| ImageNet100 (eps=72) | ResNet34    | FastEGL1       | 67.60 | 46.74 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRyyoPo_FXle1oXL?e=70K5iP) |
| ImageNet100 (eps=72) | ResNet34    | FastEGL1+NuAT  | 67.16 | 48.82 | [download](https://1drv.ms/u/s!AmUa7lHOOIcoiRshl36lNuIWyuh7?e=Egy06N) |


### Acknowledgement
Some codes of this repository are built upon [MultiRobustness](https://github.com/ftramer/MultiRobustness), [AutoAttack](http://github.com/fra31/auto-attack), [Robust fine-tune](https://github.com/fra31/robust-finetuning) and  [RobustBinarySubNet](https://github.com/IVRL/RobustBinarySubNet).

### Bibliography
If you find this repository helpful for your project, please consider citing:
```
@inproceedings{
jiang2023towards,
title={Towards Stable and Efficient Adversarial Training against $l_1$ Bounded Adversarial Attacks},
author={Jiang, Yulun and Liu, Chen and Huang, Zhichao and Salzmann, Mathieu and Susstrunk, Sabine},
booktitle={International Conference on Machine Learning},
year={2023},
organization={PMLR}
}
```
