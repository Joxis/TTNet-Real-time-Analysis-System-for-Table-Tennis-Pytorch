#!/bin/bash

python3 test.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'ttnet_1st_phase-2' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_1st_phase-9/ttnet_1st_phase-9_epoch_30.pth \
  --no_seg \
  --no_local \
  --no_event \
  --smooth-labelling