#!/usr/bin/env bash
# scripts/test_mambavision_small_enf_nam_sod.sh

cd /home/MLab

python test.py \
  --network models.networks.mambavision_small_enf_nam_sod \
  --checkpoint runs/mambavision_small_enf_nam_sod_eorssd/checkpoints/final.pth \
  --test-images datasets/EORSSD/test-images \
  --test-masks datasets/EORSSD/test-labels \
  --test-nam datasets/EORSSD/test-nam \
  --output-dir runs/mambavision_small_enf_nam_sod_eorssd/test/EORSSD \
  --dataset-name EORSSD \
  --image-size 352 \
  --batch-size 8 \
  --num-workers 8 \
  --warmup-steps 10 \
  --log-interval 100