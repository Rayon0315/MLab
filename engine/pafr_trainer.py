from __future__ import annotations

import logging
import time

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from engine.model_inputs import prepare_model_inputs
from engine.pafr_mean_inputs import build_scaled_pafr_model_inputs
from engine.trainer import (
    MAX_GRAD_NORM,
    build_non_finite_report,
    get_amp_dtype,
    get_batch_names,
)


logger = logging.getLogger(__name__)

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


def _denormalize(image: Tensor) -> Tensor:
    mean = image.new_tensor(_IMAGE_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(_IMAGE_STD).view(1, 3, 1, 1)
    return image * std + mean


def _normalize(image: Tensor) -> Tensor:
    mean = image.new_tensor(_IMAGE_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(_IMAGE_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def make_pafr_color_jitter(image: Tensor) -> Tensor:
    """PAFR ColorJitter(0.2, 0.2, 0.2, 0.1), applied safely in RGB [0,1]."""
    rgb = _denormalize(image.float()).clamp(0.0, 1.0)

    brightness = float(torch.empty(1).uniform_(0.8, 1.2).item())
    contrast = float(torch.empty(1).uniform_(0.8, 1.2).item())
    saturation = float(torch.empty(1).uniform_(0.8, 1.2).item())
    hue = float(torch.empty(1).uniform_(-0.1, 0.1).item())

    transforms = [
        lambda x: TF.adjust_brightness(x, brightness),
        lambda x: TF.adjust_contrast(x, contrast),
        lambda x: TF.adjust_saturation(x, saturation),
        lambda x: TF.adjust_hue(x, hue),
    ]
    order = torch.randperm(len(transforms)).tolist()
    for index in order:
        rgb = transforms[index](rgb)

    return _normalize(rgb.clamp(0.0, 1.0))


def train_pafr_one_epoch(
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
        scribble_gt = batch["scribble_gt"].to(device, non_blocking=True)
        scribble_mask = batch["scribble_mask"].to(device, non_blocking=True)

        jittered = make_pafr_color_jitter(image)
        scaled_image = torch.nn.functional.interpolate(
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
            scaled_model_inputs = build_scaled_pafr_model_inputs(
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
            )
            loss = loss_dict["loss"]

        if not torch.isfinite(loss).all():
            report = build_non_finite_report(
                outputs=outputs,
                loss_dict=loss_dict,
            )
            raise FloatingPointError(
                "Non-finite PAFR training loss | "
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
                "Valid %.4f | FG(valid) %.4f | Grad %.4f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                loss.detach().float().item(),
                loss_dict["loss_main"].detach().float().item(),
                loss_dict["loss_pce"].detach().float().item(),
                loss_dict["loss_crf"].detach().float().item(),
                loss_dict["loss_src"].detach().float().item(),
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
