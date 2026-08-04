#!/usr/bin/env bash
# scripts/train_mv_base_eorssd.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python train.py \
    --network models.networks.mambavision_small_sod \
    --run-dir runs/mv_base_eorssd \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
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