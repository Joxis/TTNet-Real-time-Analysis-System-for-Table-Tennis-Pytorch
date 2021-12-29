#!/bin/bash

python3 demo.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'demo' \
  --arch 'ttnet' \
  --gpu_idx 0 \
  --num_workers 128 \
  --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --video_path ../dataset/test/videos/test_1.mp4 \
  --save_demo_output