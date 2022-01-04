#!/bin/bash

python3 test.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'ttnet_3rd_phase' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_phase3/ttnet_phase3_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --smooth-labelling