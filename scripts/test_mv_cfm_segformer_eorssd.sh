#!/usr/bin/env bash
# scripts/test_mv_cfm_segformer_eorssd.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python test.py \
    --network models.networks.mambavision_small_cfm_segformer_sod \
    --checkpoint runs/mv_cfm_segformer_eorssd_aug8_e45/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --output-dir runs/mv_cfm_segformer_eorssd_aug8_e45/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
