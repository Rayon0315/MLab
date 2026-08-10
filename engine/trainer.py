# engine/trainer.py

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from engine.model_inputs import (
    get_model_input_keys,
    prepare_model_inputs,
)


logger = logging.getLogger(__name__)


MAX_GRAD_NORM = 5.0


def get_edge_target_key(
    model: nn.Module,
) -> str | None:
    nam_keys = [
        key
        for key in get_model_input_keys(model)
        if key.startswith("nam_")
    ]

    if not nam_keys:
        return None

    return max(
        nam_keys,
        key=lambda key: int(
            key.removeprefix(
                "nam_"
            )
        ),
    )


def get_amp_dtype(
    device: torch.device,
    use_amp: bool,
) -> torch.dtype | None:
    if not use_amp:
        return None

    if (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def get_batch_names(
    batch: dict,
) -> str:
    names = batch.get(
        "name",
        [],
    )

    if isinstance(
        names,
        str,
    ):
        return names

    return ", ".join(
        str(name)
        for name in names
    )


def describe_tensor(
    name: str,
    tensor: Tensor,
) -> str:
    detached = tensor.detach()

    finite_mask = torch.isfinite(
        detached
    )

    finite_count = int(
        finite_mask.sum().item()
    )

    total_count = (
        detached.numel()
    )

    if finite_count == 0:
        value_range = (
            "no finite values"
        )
    else:
        finite_values = (
            detached[
                finite_mask
            ].float()
        )

        minimum = (
            finite_values
            .min()
            .item()
        )

        maximum = (
            finite_values
            .max()
            .item()
        )

        value_range = (
            f"min={minimum:.6g}, "
            f"max={maximum:.6g}"
        )

    return (
        f"{name}: "
        f"shape={tuple(detached.shape)}, "
        f"dtype={detached.dtype}, "
        f"finite={finite_count}/"
        f"{total_count}, "
        f"{value_range}"
    )


def iter_output_tensors(
    outputs: dict,
) -> Iterable[
    tuple[str, Tensor]
]:
    for name, value in (
        outputs.items()
    ):
        if isinstance(
            value,
            Tensor,
        ):
            yield name, value
            continue

        if isinstance(
            value,
            (list, tuple),
        ):
            for index, item in enumerate(
                value
            ):
                if isinstance(
                    item,
                    Tensor,
                ):
                    yield (
                        f"{name}[{index}]",
                        item,
                    )


def build_non_finite_report(
    outputs: dict,
    loss_dict: dict[
        str,
        Tensor,
    ],
) -> str:
    descriptions: list[
        str
    ] = []

    for name, tensor in (
        iter_output_tensors(
            outputs
        )
    ):
        if not torch.isfinite(
            tensor
        ).all():
            descriptions.append(
                describe_tensor(
                    name,
                    tensor,
                )
            )

    for name, tensor in (
        loss_dict.items()
    ):
        if not torch.isfinite(
            tensor
        ).all():
            descriptions.append(
                describe_tensor(
                    name,
                    tensor,
                )
            )

    if not descriptions:
        return (
            "No non-finite output "
            "tensor was identified."
        )

    return " | ".join(
        descriptions
    )


def train_one_epoch(
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
    region_target_key: str | None = None,
) -> tuple[
    dict[str, float],
    int,
]:
    model.train()

    criterion = criterion.to(
        device
    )

    criterion.train()

    edge_target_key = (
        get_edge_target_key(
            model
        )
    )

    amp_dtype = get_amp_dtype(
        device=device,
        use_amp=use_amp,
    )

    use_grad_scaler = (
        use_amp
        and amp_dtype
        == torch.float16
        and scaler.is_enabled()
    )

    precision_name = (
        str(
            amp_dtype
        ).removeprefix(
            "torch."
        )
        if amp_dtype is not None
        else "float32"
    )

    logger.info(
        "Training precision: %s | "
        "GradScaler: %s | "
        "Max grad norm: %.1f",
        precision_name,
        use_grad_scaler,
        MAX_GRAD_NORM,
    )

    if region_target_key is not None:
        logger.info(
            "Region loss target: %s",
            region_target_key,
        )

    total_samples = 0

    loss_sums: dict[
        str,
        float,
    ] = {}

    start_time = (
        time.perf_counter()
    )

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        model_inputs = (
            prepare_model_inputs(
                model=model,
                batch=batch,
                device=device,
            )
        )

        mask = batch[
            "mask"
        ].to(
            device,
            non_blocking=True,
        )

        nam_target = None

        if edge_target_key is not None:
            nam_target = model_inputs[
                edge_target_key
            ]

        region_target = None

        if region_target_key is not None:
            if region_target_key not in batch:
                raise KeyError(
                    "Batch does not contain "
                    "region loss target: "
                    f"{region_target_key}"
                )

            region_target = batch[
                region_target_key
            ].to(
                device,
                non_blocking=True,
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=(
                amp_dtype
                if amp_dtype is not None
                else torch.float32
            ),
            enabled=(
                amp_dtype is not None
            ),
        ):
            outputs = model(
                **model_inputs
            )

            loss_dict = criterion(
                outputs,
                mask,
                nam_target=nam_target,
                region_target=region_target,
            )

            loss = loss_dict[
                "loss"
            ]

        if not torch.isfinite(
            loss
        ).all():
            report = (
                build_non_finite_report(
                    outputs=outputs,
                    loss_dict=loss_dict,
                )
            )

            raise FloatingPointError(
                "Non-finite training loss | "
                f"epoch={epoch} | "
                f"batch={batch_index} | "
                f"step={global_step + 1} | "
                f"samples="
                f"{get_batch_names(batch)} | "
                f"{report}"
            )

        if use_grad_scaler:
            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )
        else:
            loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
                error_if_nonfinite=False,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            if use_grad_scaler:
                scaler.update()

            raise FloatingPointError(
                "Non-finite gradient norm | "
                f"epoch={epoch} | "
                f"batch={batch_index} | "
                f"step={global_step + 1} | "
                f"samples="
                f"{get_batch_names(batch)} | "
                f"grad_norm="
                f"{gradient_norm.detach().float().item()}"
            )

        if use_grad_scaler:
            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            optimizer.step()

        batch_size = model_inputs[
            "image"
        ].shape[0]

        total_samples += (
            batch_size
        )

        global_step += 1

        for name, value in (
            loss_dict.items()
        ):
            loss_sums[name] = (
                loss_sums.get(
                    name,
                    0.0,
                )
                + value
                .detach()
                .float()
                .item()
                * batch_size
            )

        if (
            batch_index
            % log_interval
            == 0
            or batch_index
            == len(data_loader)
        ):
            zero = (
                loss.detach()
                .new_zeros(())
            )

            logger.info(
                "Epoch %03d | "
                "Batch %05d/%05d | "
                "Step %07d | "
                "Loss %.6f | "
                "Main %.6f | "
                "Aux %.6f | "
                "Region %.6f | "
                "Edge %.6f | "
                "Grad %.4f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                (
                    loss
                    .detach()
                    .float()
                    .item()
                ),
                (
                    loss_dict.get(
                        "loss_main",
                        loss,
                    )
                    .detach()
                    .float()
                    .item()
                ),
                (
                    loss_dict.get(
                        "loss_aux",
                        zero,
                    )
                    .detach()
                    .float()
                    .item()
                ),
                (
                    loss_dict.get(
                        "loss_region",
                        zero,
                    )
                    .detach()
                    .float()
                    .item()
                ),
                (
                    loss_dict.get(
                        "loss_edge",
                        zero,
                    )
                    .detach()
                    .float()
                    .item()
                ),
                (
                    gradient_norm
                    .detach()
                    .float()
                    .item()
                ),
            )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    statistics = {
        name: (
            value
            / total_samples
        )
        for name, value
        in loss_sums.items()
    }

    statistics["lr"] = (
        optimizer.param_groups[
            0
        ]["lr"]
    )

    statistics[
        "time_seconds"
    ] = elapsed_time

    return (
        statistics,
        global_step,
    )