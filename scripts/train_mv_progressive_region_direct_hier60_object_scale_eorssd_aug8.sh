#!/usr/bin/env bash
# scripts/train_mv_progressive_region_direct_hier60_object_scale_eorssd_aug8.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python train_object_scale_loss.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_sod \
    --run-dir runs/mv_progressive_region_direct_hier60_object_scale_eorssd_aug8_e45 \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-instances datasets/EORSSD/train-instances \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 45 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --object-weight 0.1 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
