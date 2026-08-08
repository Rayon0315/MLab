# scripts/test_convnext_small_sod.sh
#!/usr/bin/env bash

cd /home/MLab

RUN_DIR="runs/convnext_small_sod_eorssd"

python test.py \
  --network models.networks.convnext_small_sod \
  --checkpoint "$RUN_DIR/checkpoints/final.pth" \
  --test-images datasets/EORSSD/test-images \
  --test-masks datasets/EORSSD/test-labels \
  --dataset-name EORSSD \
  --output-dir "$RUN_DIR/test/EORSSD" \
  --image-size 352 \
  --batch-size 8 \
  --num-workers 8 \
  --log-interval 100 \
  --amp