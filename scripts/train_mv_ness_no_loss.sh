#!/usr/bin/env bash
# scripts/train_mv_ness_no_loss.sh

set -euo pipefail

unset OMP_NUM_THREADS

cd /home/MLab

RUN_DIR="runs/train_mv_ness_no_loss"
LATEST_CHECKPOINT="${RUN_DIR}/checkpoints/latest.pth"

RESUME_ARGS=()

if [[ -f "${LATEST_CHECKPOINT}" ]]; then
    RESUME_ARGS=(
        --resume
        "${LATEST_CHECKPOINT}"
    )
fi

python train.py \
    --network models.networks.mambavision_small_ness_nam_sod \
    --run-dir "${RUN_DIR}" \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-nam datasets/EORSSD/train-nam \
    --image-size 352 \
    --epochs 45 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --edge-weight 0 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100 \
    "${RESUME_ARGS[@]}"