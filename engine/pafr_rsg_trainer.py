from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def build_scaled_rsg_model_inputs(
    model_inputs: dict[str, Tensor],
    scaled_image: Tensor,
) -> dict[str, Tensor]:
    """Build the PAFR/SRC scaled forward inputs for an RSG model."""
    scaled_inputs: dict[str, Tensor] = {"image": scaled_image}

    for key, value in model_inputs.items():
        if key == "image":
            continue
        if key == "region_partition":
            partition = value
            if partition.ndim == 3:
                partition = partition.unsqueeze(1)
            elif partition.ndim != 4 or partition.shape[1] != 1:
                raise ValueError(
                    "region_partition must be [B,H,W] or [B,1,H,W]"
                )
            partition = F.interpolate(
                partition.float(),
                size=scaled_image.shape[-2:],
                mode="nearest",
            )
            scaled_inputs[key] = partition[:, 0].long()
            continue
        if key.startswith("mean_"):
            scaled_inputs[key] = F.interpolate(
                value.float(),
                size=scaled_image.shape[-2:],
                mode="nearest",
            ).to(dtype=value.dtype)
            continue
        raise KeyError(
            "Unsupported non-image input for scaled RSG forward: "
            f"{key}"
        )

    return scaled_inputs


def train_pafr_rsg_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
    log_interval: int,
    src_scale: float = 0.3,
) -> tuple[dict[str, float], int]:
    from engine.model_inputs import prepare_model_inputs
    from engine.pafr_trainer import make_pafr_color_jitter
    from engine.trainer import (
        MAX_GRAD_NORM,
        build_non_finite_report,
        get_amp_dtype,
        get_batch_names,
    )

    model.train()
    criterion = criterion.to(device)
    criterion.train()

    amp_dtype = get_amp_dtype(device=device, use_amp=use_amp)
    use_grad_scaler = (
        use_amp
        and amp_dtype == torch.float16
        and scaler.is_enabled()
    )

    logger.info(
        "Training precision: %s | GradScaler: %s | Max grad norm: %.1f | SRC scale: %.3f",
        (
            str(amp_dtype).removeprefix("torch.")
            if amp_dtype is not None
            else "float32"
        ),
        use_grad_scaler,
        MAX_GRAD_NORM,
        src_scale,
    )

    total_samples = 0
    loss_sums: dict[str, float] = {}
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(data_loader, start=1):
        model_inputs = prepare_model_inputs(
            model=model,
            batch=batch,
            device=device,
        )
        image = model_inputs["image"]
        region_partition = model_inputs["region_partition"]
        scribble_gt = batch["scribble_gt"].to(device, non_blocking=True)
        scribble_mask = batch["scribble_mask"].to(device, non_blocking=True)

        jittered = make_pafr_color_jitter(image)
        scaled_image = F.interpolate(
            jittered,
            scale_factor=src_scale,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=False,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=(amp_dtype if amp_dtype is not None else torch.float32),
            enabled=(amp_dtype is not None),
        ):
            outputs = model(**model_inputs)
            scaled_model_inputs = build_scaled_rsg_model_inputs(
                model_inputs=model_inputs,
                scaled_image=scaled_image,
            )
            scaled_outputs = model(**scaled_model_inputs)

            loss_dict = criterion(
                outputs=outputs,
                scaled_outputs=scaled_outputs,
                image=image,
                target=scribble_gt,
                valid_mask=scribble_mask,
                region_partition=region_partition,
            )
            loss = loss_dict["loss"]

        if not torch.isfinite(loss).all():
            report = build_non_finite_report(
                outputs=outputs,
                loss_dict=loss_dict,
            )
            raise FloatingPointError(
                "Non-finite PAFR RSG loss | "
                f"epoch={epoch} | batch={batch_index} | "
                f"step={global_step + 1} | samples={get_batch_names(batch)} | {report}"
            )

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=False,
        )
        if not torch.isfinite(gradient_norm):
            optimizer.zero_grad(set_to_none=True)
            if use_grad_scaler:
                scaler.update()
            raise FloatingPointError(
                "Non-finite gradient norm | "
                f"epoch={epoch} | batch={batch_index} | "
                f"step={global_step + 1} | samples={get_batch_names(batch)}"
            )

        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_size = image.shape[0]
        total_samples += batch_size
        global_step += 1

        for name, value in loss_dict.items():
            loss_sums[name] = (
                loss_sums.get(name, 0.0)
                + value.detach().float().item() * batch_size
            )

        if batch_index % log_interval == 0 or batch_index == len(data_loader):
            logger.info(
                "Epoch %03d | Batch %05d/%05d | Step %07d | "
                "Loss %.6f | Main %.6f | PCE %.6f | CRF %.6f | SRC %.6f | "
                "Region %.6f | Valid %.4f | FG(valid) %.4f | Grad %.4f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                loss.detach().float().item(),
                loss_dict["loss_main"].detach().float().item(),
                loss_dict["loss_pce"].detach().float().item(),
                loss_dict["loss_crf"].detach().float().item(),
                loss_dict["loss_src"].detach().float().item(),
                loss_dict["loss_region_consistency"].detach().float().item(),
                loss_dict["stat_valid_ratio"].detach().float().item(),
                loss_dict["stat_fg_ratio_on_valid"].detach().float().item(),
                gradient_norm.detach().float().item(),
            )

    elapsed = time.perf_counter() - start_time
    statistics = {
        name: value / total_samples
        for name, value in loss_sums.items()
    }
    statistics["lr"] = optimizer.param_groups[0]["lr"]
    statistics["time_seconds"] = elapsed
    return statistics, global_step
