# scripts/train_mambavision_small_sod.sh
#!/usr/bin/env bash

cd /home/MLab

python train.py \
    --network models.networks.mambavision_small_sod \
    --run-dir runs/mambavision_small_sod_duts \
    --image-size 352 \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --val-count 500 \
    --seed 42 \
    --save-every 5 \
    --log-interval 100