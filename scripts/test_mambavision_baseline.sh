# scripts/test_mambavision_baseline.sh
#!/usr/bin/env bash

cd /home/MLab

python test.py \
    --network models.networks.mambavision_baseline \
    --checkpoint runs/mambavision_tiny_baseline_duts/checkpoints/final.pth \
    --output-dir runs/mambavision_tiny_baseline_duts/test/DUTS-TE-standard \
    --dataset-name DUTS-TE \
    --batch-size 8 \
    --num-workers 4 \
    --log-interval 100