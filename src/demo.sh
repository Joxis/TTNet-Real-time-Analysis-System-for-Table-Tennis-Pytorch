#!/bin/bash

python3 demo.py \
  --working-dir '/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch' \
  --saved_fn 'demo' \
  --arch 'ttnet' \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/ttnet_phase2/ttnet_phase2_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --video_path ../dataset/test/videos/test_1-10s.mp4 \
  --save_demo_output \
  --output_format 'video'