#!/usr/bin/env bash
# scripts/train_mv_progressive_region_direct_hier60_f_loss_eorssd_aug8.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python train_f_loss.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_sod \
    --run-dir runs/mv_progressive_region_direct_hier60_f_loss_eorssd_aug8_e45 \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 45 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --f-weight 0.2 \
    --f-beta2 0.3 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100