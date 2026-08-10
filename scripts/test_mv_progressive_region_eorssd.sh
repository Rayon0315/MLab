#!/usr/bin/env bash
# scripts/test_mv_progressive_region_eorssd.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python test.py \
    --network models.networks.mambavision_small_progressive_region_sod \
    --checkpoint runs/mv_progressive_region_eorssd_aug8_e45/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-mean datasets/EORSSD/test-mean \
    --output-dir runs/mv_progressive_region_eorssd_aug8_e45/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100