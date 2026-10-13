#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
DATASET="cifar10"
EPOCHS=101
SAVE_STEPS=10
NUM_EXPERIMENTS=8
LOGDIR="./.experiments-wrn-28-2-dpsgd"

for ((EXPID=0; EXPID<NUM_EXPERIMENTS; EXPID++))
do
    python3 -u train_dpsgd.py \
        --config='./config.yml' \
        --dataset_name=$DATASET \
        --num_epochs=$EPOCHS \
        --save_steps=$SAVE_STEPS \
        --num_experiments=$NUM_EXPERIMENTS \
        --expid=$EXPID \
        --log_dir=$LOGDIR
done