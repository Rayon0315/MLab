#!/usr/bin/env bash
# scripts/train_mv_progressive_region_direct_hier60_region_hybrid_ema_refinement_eorssd_aug8.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python train_ema_refinement.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod \
    --stage1-checkpoint runs/mv_progressive_region_direct_hier60_region_hybrid_eorssd_aug8_e45/checkpoints/final.pth \
    --run-dir runs/mv_progressive_region_direct_hier60_region_hybrid_ema_refinement_eorssd_aug8_e15 \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 15 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --ema-decay 0.999 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
