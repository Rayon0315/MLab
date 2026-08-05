#!/usr/bin/env bash
# scripts/train_mv_ness_eorssd.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python train.py \
    --network models.networks.mambavision_small_ness_nam_sod \
    --run-dir runs/mv_ness_eorssd \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-nam datasets/EORSSD/train-nam \
    --image-size 352 \
    --epochs 45 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --edge-weight 0.1 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100