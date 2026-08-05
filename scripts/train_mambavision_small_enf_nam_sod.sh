#!/usr/bin/env bash
# scripts/train_mambavision_small_enf_nam_sod.sh

cd /home/MLab

python train.py \
  --network models.networks.mambavision_small_enf_nam_sod \
  --train-images datasets/EORSSD/train-images \
  --train-masks datasets/EORSSD/train-labels \
  --train-nam datasets/EORSSD/train-nam \
  --run-dir runs/mambavision_small_enf_nam_sod_eorssd \
  --image-size 352 \
  --epochs 45 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 1e-4 \
  --min-lr 1e-6 \
  --weight-decay 1e-4 \
  --aux-weight 0.4 \
  --augment-8way \
  --seed 42 \
  --save-every 5 \
  --log-interval 100