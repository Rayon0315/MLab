# scripts/test_mambavision_small_progressive_sod_eorssd.sh
#!/usr/bin/env bash

cd /home/MLab

RUN_DIR="runs/mambavision_small_progressive_sod_eorssd"

python test.py \
  --network models.networks.mambavision_small_progressive_sod \
  --checkpoint "$RUN_DIR/checkpoints/final.pth" \
  --test-images datasets/EORSSD/test-images \
  --test-masks datasets/EORSSD/test-labels \
  --dataset-name EORSSD \
  --image-size 352 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --amp \
  --output-dir "$RUN_DIR/test/EORSSD" \
  --log-interval 100