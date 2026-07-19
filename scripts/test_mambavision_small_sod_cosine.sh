# scripts/test_mambavision_small_sod_cosine.sh
#!/usr/bin/env bash

cd /home/MLab

python test.py \
    --network models.networks.mambavision_small_sod \
    --checkpoint runs/mambavision_small_sod_cosine_duts/checkpoints/final.pth \
    --output-dir runs/mambavision_small_sod_cosine_duts/test/DUTS-TE \
    --dataset-name DUTS-TE \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --log-interval 100