#!/usr/bin/env bash
# scripts/test_mv_nam_scan_eorssd.sh

set -euo pipefail

unset OMP_NUM_THREADS || true

cd /home/MLab

RUN_DIR="${RUN_DIR:-runs/mv_nam_scan_eorssd}"
MODE_FILE="${RUN_DIR}/scan_mode.txt"
FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/final.pth"
LATEST_CHECKPOINT="${RUN_DIR}/checkpoints/latest.pth"

if [[ -f "${MODE_FILE}" ]]; then
    SCAN_MODE="$(tr -d '[:space:]' < "${MODE_FILE}")"
else
    SCAN_MODE="${MLAB_NAM_SCAN_MODE:-nam_hierarchical}"
fi

export MLAB_NAM_SCAN_MODE="${SCAN_MODE}"

if [[ -n "${CHECKPOINT:-}" ]]; then
    SELECTED_CHECKPOINT="${CHECKPOINT}"
elif [[ -f "${FINAL_CHECKPOINT}" ]]; then
    SELECTED_CHECKPOINT="${FINAL_CHECKPOINT}"
elif [[ -f "${LATEST_CHECKPOINT}" ]]; then
    SELECTED_CHECKPOINT="${LATEST_CHECKPOINT}"
else
    echo "No checkpoint found under ${RUN_DIR}/checkpoints"
    exit 1
fi

echo "Run directory: ${RUN_DIR}"
echo "Scan mode: ${SCAN_MODE}"
echo "Checkpoint: ${SELECTED_CHECKPOINT}"

python test.py \
    --network models.networks.mambavision_small_nam_scan_sod \
    --checkpoint "${SELECTED_CHECKPOINT}" \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-nam datasets/EORSSD/test-nam \
    --output-dir "${RUN_DIR}/test/EORSSD" \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100 \
    --amp