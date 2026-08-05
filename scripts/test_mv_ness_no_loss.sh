#!/usr/bin/env bash
# scripts/test_mv_ness_no_loss.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python test.py \
    --network models.networks.mambavision_small_ness_nam_sod \
    --checkpoint runs/train_mv_ness_no_loss/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-nam datasets/EORSSD/test-nam \
    --output-dir runs/train_mv_ness_no_loss/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100