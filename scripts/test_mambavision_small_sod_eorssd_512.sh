# scripts/test_mambavision_small_sod_eorssd_512.sh
#!/usr/bin/env bash

cd /home/MLab

RUN_DIR="runs/mambavision_small_sod_eorssd_512"

python test.py \
  --network models.networks.mambavision_small_sod \
  --checkpoint "$RUN_DIR/checkpoints/final.pth" \
  --test-images datasets/EORSSD/test-images \
  --test-masks datasets/EORSSD/test-labels \
  --dataset-name EORSSD \
  --image-size 512 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --amp \
  --output-dir "$RUN_DIR/test/EORSSD" \
  --log-interval 100