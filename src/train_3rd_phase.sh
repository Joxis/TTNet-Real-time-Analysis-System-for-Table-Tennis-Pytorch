#!/bin/bash

python3 main.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'ttnet_phase3' \
  --no-val \
  --batch_size 8 \
  --num_workers 64 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 5 \
  --lr_factor 0.2 \
  --gpu_idx 0 \
  --global_weight 1. \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_phase2/ttnet_phase2_epoch_30.pth \
  --smooth-labelling