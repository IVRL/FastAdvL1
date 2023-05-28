GPU=0

eps=6
k=100
st=6
predir="trained/CIFAR100/EPS=${eps}"
OUTDIR="ATTA"

python run/train_atta_cifar.py \
    --dataset cifar100 \
    --attack name=pgd,order=1,threshold=${eps},iter_num=1,step_size=${st},k=${k} \
    --test_attack name=apgd,order=1,threshold=${eps},iter_num=20 \
    --epochs 40 --epochs-reset 40 --rs --lr 0.05 \
    --model-dir  ${predir}/${OUTDIR} --gpuid ${GPU} --save-freq 40

python run/eval_autoattack.py \
    --dataset cifar100 \
    --model_type preact_resnet_softplus \
    --model2load ${predir}/${OUTDIR}/ep_bestvalid.ckpt \
    --out_file ${predir}/${OUTDIR}/eval_best.log \
    --attack name=apgd,order=1,threshold=${eps} \
    --gpu ${GPU} 

