#!/usr/bin/env bash
# scripts/test_mambavision_small_nam_sod.sh

cd /home/MLab

python test.py \
    --network models.networks.mambavision_small_nam_sod \
    --checkpoint runs/mambavision_small_nam_sod_duts/checkpoints/final.pth \
    --test-images datasets/DUTS/DUTS-TE/DUTS-TE-Image \
    --test-masks datasets/DUTS/DUTS-TE/DUTS-TE-Mask \
    --test-nam datasets/DUTS/DUTS-TE/nam \
    --output-dir runs/mambavision_small_nam_sod_duts/test/DUTS-TE \
    --dataset-name DUTS-TE \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100