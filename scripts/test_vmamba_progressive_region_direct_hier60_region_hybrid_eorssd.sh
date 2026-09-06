#!/usr/bin/env bash
# scripts/test_vmamba_progressive_region_direct_hier60_region_hybrid_eorssd.sh

set -e

unset OMP_NUM_THREADS

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MLab}"
DATA_ROOT="${DATA_ROOT:-$HOME/autodl-tmp/datasets}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python test.py \
    --network models.networks.vmamba_small_progressive_region_direct_hier60_region_hybrid_sod \
    --checkpoint runs/vmamba_progressive_region_direct_hier60_region_hybrid_eorssd_aug8_e45/checkpoints/final.pth \
    --test-images "$DATA_ROOT/EORSSD/test-images" \
    --test-masks "$DATA_ROOT/EORSSD/test-labels" \
    --test-mean "$DATA_ROOT/EORSSD/test-mean" \
    --output-dir runs/vmamba_progressive_region_direct_hier60_region_hybrid_eorssd_aug8_e45/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
