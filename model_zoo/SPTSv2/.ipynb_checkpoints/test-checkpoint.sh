#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3  main.py \
        --train_dataset totaltext_train \
        --val_dataset ../../data/selected_images/37 \
        --max_length 25 \
        --data_root .. \
        --batch_size 1 \
        --depths 6 \
        --lr 0.0005 \
        --pre_norm \
        --num_workers 8 \
        --eval \
        --resume ./models/pretrained_model.pth \
        --output_dir ./output/37 \
        --padding_bins 0 \
        --pad_rec\
        --filters 1:2:3:4:5:6:7:32:33:34:35:36:38:39:40:41 \
        
