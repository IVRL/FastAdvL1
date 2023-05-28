GPU=0

eps=6
st=2.45 # sqrt(eps)
eps_train=2
lmbd=0.1
predir="trained/CIFAR100/EPS=${eps}"
savedir="FastEGL1_NuAT"

python run/train_normal.py \
    --epoch_num 40 \
    --dataset cifar100 \
    --model_type preact_resnet_softplus \
    --optim name=sgd,lr=0.05,momentum=0.9,weight_decay=5e-4,not_wd_bn=True \
    --lr_schedule name=jump,min_jump_pt=30,jump_freq=5,start_v=0.05,power=0.1 \
    --valid_ratio 0.04 \
    --gpu ${GPU} \
    --batch_size 128 \
    --out_folder ${predir}/${savedir} \
    --attack name=NuAT,threshold=${eps},iter_num=1,step_size=${st},update_order=2,eps_train=${eps_train},lmbd=${lmbd} \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --n_eval 1000
    
python run/eval_autoattack.py \
    --dataset cifar100 \
    --model_type preact_resnet_softplus \
    --model2load ${predir}/${savedir}/None_bestvalid.ckpt \
    --out_file ${predir}/${savedir}/eval_log.txt \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} \

