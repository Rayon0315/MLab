#!/usr/bin/env bash
# scripts/test_cfmnet_upernet_eorssd.sh

set -e

unset OMP_NUM_THREADS

cd /home/MLab

python test.py \
    --network models.networks.cfmnet_upernet_sod \
    --checkpoint runs/cfmnet_upernet_eorssd_aug8_e45/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --output-dir runs/cfmnet_upernet_eorssd_aug8_e45/test/EORSSD \
    --warmup-steps 10 \
    --log-interval 100
