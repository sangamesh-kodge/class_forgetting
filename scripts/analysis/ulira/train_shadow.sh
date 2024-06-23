# Trains 8 shadow models without class 0
# MAX_RUNS=7
MAX_RUNS=0
IGNORE_CLASS=0
for run in $(seq 0 1 $MAX_RUNS);
do 
    CUDA_VISIBLE_DEVICES=0 python  ./analysis/shadow_train.py  --data-path ../data \
        --dataset cifar100 \
        --ignore-class $IGNORE_CLASS \
        --arch resnet18 \
        --group-name train-Ignoreing$IGNORE_CLASS \
        --project-name  tmlr-submission-shadow \
        --batch-size 64 
done 

# Trains 8 shadow models with entire dataset
for run in $(seq 0 1 $MAX_RUNS);
do 
    CUDA_VISIBLE_DEVICES=0 python  ./analysis/shadow_train.py --data-path ../data \
        --dataset cifar100 \
        --arch resnet18 \
        --group-name train \
        --project-name  tmlr-submission-shadow \
        --batch-size 64 
done 