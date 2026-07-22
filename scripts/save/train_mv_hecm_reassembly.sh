# scripts/train_mv_hecm_reassembly.sh
python train.py \
    --network models.networks.mambavision_small_hecm_reassembly_sod \
    --train-images datasets/DUTS/DUTS-TR/DUTS-TR-Image \
    --train-masks datasets/DUTS/DUTS-TR/DUTS-TR-Mask \
    --train-nam datasets/DUTS/DUTS-TR/nam \
    --run-dir runs/mv_hecm_reassembly \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --aux-weight 0.4 \
    --edge-weight 0 \
    --save-every 5 \
    --log-interval 100