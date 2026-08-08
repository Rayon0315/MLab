# scripts/train_convnext_small_sod.sh
#!/usr/bin/env bash

cd /home/MLab

RUN_DIR="runs/convnext_small_sod_eorssd"

ARGS=(
  --network models.networks.convnext_small_sod
  --run-dir "$RUN_DIR"
  --train-images datasets/EORSSD/train-images
  --train-masks datasets/EORSSD/train-labels
  --image-size 352
  --epochs 45
  --batch-size 8
  --num-workers 8
  --lr 1e-4
  --min-lr 1e-6
  --weight-decay 1e-4
  --aux-weight 0.4
  --edge-weight 0
  --augment-8way
  --seed 42
  --save-every 5
  --log-interval 100
  --amp
)

if [ -f "$RUN_DIR/checkpoints/latest.pth" ]; then
  ARGS+=(
    --resume "$RUN_DIR/checkpoints/latest.pth"
  )
fi

python train.py "${ARGS[@]}"