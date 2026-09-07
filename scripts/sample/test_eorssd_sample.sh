#!/usr/bin/env bash
# scripts/test_vmamba_progressive_region_direct_hier60_region_hybrid_dictionary_routing_eorssd.sh

set -e

unset OMP_NUM_THREADS

PROJECT_ROOT="/home/MLab"
DATA_ROOT="/root/autodl-tmp/datasets"

cd "$PROJECT_ROOT"

python test.py \
    --network models.networks.vmamba_small_progressive_region_direct_hier60_region_hybrid_dictionary_routing_sod \
    --checkpoint runs/vmamba_progressive_region_direct_hier60_region_hybrid_dictionary_routing_eorssd_aug8_e45/checkpoints/final.pth \
    --test-images "$DATA_ROOT/EORSSD/test-images" \
    --test-masks "$DATA_ROOT/EORSSD/test-labels" \
    --test-mean "$DATA_ROOT/EORSSD/test-mean" \
    --output-dir runs/vmamba_progressive_region_direct_hier60_region_hybrid_dictionary_routing_eorssd_aug8_e45/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
