# scripts/train_mambavision_sod_cosine.sh
#!/usr/bin/env bash

cd /home/MLab

python train.py \
    --network models.networks.mambavision_sod \
    --run-dir runs/mambavision_sod_cosine_duts \
    --image-size 352 \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --seed 42 \
    --save-every 5 \
    --log-interval 100