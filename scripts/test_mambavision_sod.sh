# scripts/test_mambavision_sod.sh
python test.py \
    --network models.networks.mambavision_sod \
    --checkpoint runs/mambavision_sod_duts/checkpoints/best.pth \
    --output-dir runs/mambavision_sod_duts/test/DUTS-TE \
    --dataset-name DUTS-TE \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --log-interval 100