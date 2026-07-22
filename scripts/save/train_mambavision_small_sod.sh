# scripts/train_mambavision_small_sod.sh
#!/usr/bin/env bash

cd /home/MLab

python train.py \
    --network models.networks.mambavision_small_sod \
    --run-dir runs/mambavision_small_sod_cosine_duts \
    --train-images datasets/DUTS/DUTS-TR/DUTS-TR-Image \
    --train-masks datasets/DUTS/DUTS-TR/DUTS-TR-Mask \
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