#!/usr/bin/env bash
# scripts/train_mambavision_small_nam_input_sod.sh

cd /home/MLab

python train.py \
  --network models.networks.mambavision_small_nam_input_sod \
  --train-images datasets/DUTS/DUTS-TR/DUTS-TR-Image \
  --train-masks datasets/DUTS/DUTS-TR/DUTS-TR-Mask \
  --train-nam datasets/DUTS/DUTS-TR/nam \
  --run-dir runs/mambavision_small_nam_input_sod_duts \
  --image-size 352 \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 1e-4 \
  --min-lr 1e-6 \
  --weight-decay 1e-4 \
  --aux-weight 0.4 \
  --seed 42 \
  --save-every 5 \
  --log-interval 100