# scripts/test_mv_hecm.sh

python test.py \
    --network models.networks.mambavision_small_hecm_sod \
    --checkpoint runs/mv_hecm_no_edge/checkpoints/final.pth \
    --test-images datasets/DUTS/DUTS-TE/DUTS-TE-Image \
    --test-masks datasets/DUTS/DUTS-TE/DUTS-TE-Mask \
    --test-nam datasets/DUTS/DUTS-TE/nam \
    --dataset-name DUTS-TE \
    --output-dir runs/mv_hecm_no_edge/test/DUTS-TE \
    --batch-size 8 \
    --num-workers 8