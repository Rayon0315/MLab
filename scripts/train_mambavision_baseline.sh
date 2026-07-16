# scripts/train_mambavision_baseline.sh
#!/usr/bin/env bash

cd /home/MLab

python train.py \
    --network models.networks.mambavision_baseline \
    --run-dir runs/mambavision_tiny_baseline_duts \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 4 \
    --val-count 500 \
    --seed 42 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --save-every 5 \
    --log-interval 100