#!/usr/bin/env bash
# scripts/train_vmamba_progressive_region_direct_hier60_region_hybrid_eorssd.sh

set -e

unset OMP_NUM_THREADS

PROJECT_ROOT="${PROJECT_ROOT:-/home/MLab}"
DATA_ROOT="${DATA_ROOT:-$HOME/autodl-tmp/datasets}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    --network models.networks.vmamba_small_progressive_region_direct_hier60_region_hybrid_sod \
    --train-images "$DATA_ROOT/EORSSD/train-images" \
    --train-masks "$DATA_ROOT/EORSSD/train-labels" \
    --train-mean "$DATA_ROOT/EORSSD/train-mean" \
    --image-size 352 \
    --batch-size 8 \
    --epochs 45 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --edge-weight 0.0 \
    --region-weight 0.0 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100 \
    --run-dir runs/vmamba_progressive_region_direct_hier60_region_hybrid_eorssd_aug8_e45