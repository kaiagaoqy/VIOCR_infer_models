#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3  eval.py \
        --result_path ./output/results/totaltext_val.json \
        --lexicon_type 0\
        
