#!/usr/bin/env bash
# scripts/test_mambavision_small_nam_oasis_sod.sh

set -euo pipefail

unset OMP_NUM_THREADS

cd /home/MLab

RUN_DIR="runs/train_mv_oasis"

python test.py \
    --network models.networks.mambavision_small_nam_oasis_sod \
    --checkpoint "${RUN_DIR}/checkpoints/final.pth" \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-nam datasets/EORSSD/test-nam \
    --output-dir "${RUN_DIR}/test/EORSSD" \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100 \
    "$@"