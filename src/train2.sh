#!/bin/bash

python3 main.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'ttnet_multi' \
  --no-val \
  --batch_size 8 \
  --num_workers 64 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 5 \
  --lr_factor 0.1 \
  --gpu_idx 0 \
  --smooth-labelling \
  --multitask_learning