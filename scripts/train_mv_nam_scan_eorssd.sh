#!/usr/bin/env bash
# scripts/train_mv_nam_scan_eorssd.sh

set -euo pipefail

unset OMP_NUM_THREADS || true

cd /home/MLab

RUN_DIR="${RUN_DIR:-runs/mv_nam_scan_eorssd}"
SCAN_MODE="${MLAB_NAM_SCAN_MODE:-nam_hierarchical}"
MODE_FILE="${RUN_DIR}/scan_mode.txt"
LATEST_CHECKPOINT="${RUN_DIR}/checkpoints/latest.pth"

export MLAB_NAM_SCAN_MODE="${SCAN_MODE}"

mkdir -p "${RUN_DIR}"

if [[ -f "${MODE_FILE}" ]]; then
    SAVED_MODE="$(tr -d '[:space:]' < "${MODE_FILE}")"

    if [[ "${SAVED_MODE}" != "${SCAN_MODE}" ]]; then
        echo "Scan mode mismatch: saved=${SAVED_MODE}, requested=${SCAN_MODE}"
        exit 1
    fi
else
    printf '%s\n' "${SCAN_MODE}" > "${MODE_FILE}"
fi

RESUME_ARGS=()

if [[ -f "${LATEST_CHECKPOINT}" ]]; then
    RESUME_ARGS=(
        --resume "${LATEST_CHECKPOINT}"
    )

    echo "Resuming from ${LATEST_CHECKPOINT}"
else
    echo "Starting a new run"
fi

echo "Run directory: ${RUN_DIR}"
echo "Scan mode: ${SCAN_MODE}"

python train.py \
    --network models.networks.mambavision_small_nam_scan_sod \
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
    --amp \
    "${RESUME_ARGS[@]}"