#!/bin/bash

python3 main.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'ttnet_1st_phase-2' \
  --no-val \
  --batch_size 8 \
  --num_workers 64 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --gpu_idx 0 \
  --global_weight 5. \
  --seg_weight 1. \
  --no_local \
  --no_event \
  --smooth-labelling \
  --multitask_learning