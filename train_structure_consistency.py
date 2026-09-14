from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset
from engine.trainer import train_one_epoch
from losses.cfm_structure_consistency_loss import (
    CFMStructureConsistencyLoss,
)
from train import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CFMNet structure consistency experiment."
    )

    parser.add_argument("--network", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)

    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--structure-weight", type=float, default=0.10)
    parser.add_argument("--consistency-weight", type=float, default=0.10)

    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    source_dir = run_dir / "network_source"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=log_dir / "train.log",
        resume=(args.resume is not None),
    )

    logger = logging.getLogger(__name__)

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    logger.info("AMP: %s", use_amp)
    logger.info("Network: %s", args.network)
    logger.info(
        "Structure weights | structure %.3f | consistency %.3f",
        args.structure_weight,
        args.consistency_weight,
    )

    model, network_module = build_model(args.network)

    network_source_path = Path(network_module.__file__)
    shutil.copy2(
        network_source_path,
        source_dir / network_source_path.name,
    )

    model = model.to(device)

    train_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=(args.image_size, args.image_size),
        augment_8way=args.augment_8way,
    )

    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(
                min(
                    args.max_train_samples,
                    len(train_dataset),
                )
            ),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    criterion = CFMStructureConsistencyLoss(
        aux_weight=args.aux_weight,
        structure_weight=args.structure_weight,
        consistency_weight=args.consistency_weight,
        edge_kernel=5,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            network_path=args.network,
        )

        logger.info(
            "Resumed from %s | Next epoch %d | Step %d",
            args.resume,
            start_epoch,
            global_step,
        )

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Batches per epoch: %d", len(train_loader))

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        statistics, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            log_interval=args.log_interval,
            region_target_key=None,
        )

        logger.info(
            "Epoch %03d completed | "
            "Loss %.6f | Main %.6f | Aux %.6f | "
            "Structure %.6f | Consistency %.6f | "
            "LR %.8f | Train %.1fs",
            epoch,
            statistics["loss"],
            statistics.get("loss_main", 0.0),
            statistics.get("loss_aux", 0.0),
            statistics.get("loss_structure", 0.0),
            statistics.get("loss_structure_consistency", 0.0),
            statistics["lr"],
            statistics["time_seconds"],
        )

        scheduler.step()

        save_checkpoint(
            path=checkpoint_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

        if epoch == args.epochs:
            save_checkpoint(
                path=checkpoint_dir / "final.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

    logger.info(
        "Training completed | Final checkpoint: %s",
        checkpoint_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
