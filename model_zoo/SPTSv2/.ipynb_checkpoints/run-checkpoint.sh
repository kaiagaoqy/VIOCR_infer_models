#!/bin/bash

bs=$1
lr=$2
out=$3
data=$4

python3 main.py --data_root ${data} --batch_size ${bs} --lr ${lr} --output_dir ${out} \
        --train_dataset totaltext_train \
        --val_dataset totaltext_val \
        --dec_layers 6 \
        --max_length 25 \
        --pad_rec \
        --pre_norm \
        --rotate_prob 0.3 \
        --train \
        --depths 6 \
        --padding_bins 0 \
        --epochs 280 \
        --warmup_epochs 5  \
        --finetune \
        --resume models/pretrained_model.pth \
        --filter 1:2:3:4:5:6:7:32:33:34:35:36:38:39:40:41 \
        --num_workers 1 \
 