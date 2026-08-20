# tools/repro_probe.py

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    prepare_model_inputs,
)
from losses.sod_loss import SODLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short-run reproducibility probe for MLab training."
        )
    )

    parser.add_argument(
        "--network",
        default=(
            "models.networks."
            "mambavision_small_progressive_region_direct_sod"
        ),
    )

    parser.add_argument(
        "--train-images",
        default="datasets/EORSSD/train-images",
    )

    parser.add_argument(
        "--train-masks",
        default="datasets/EORSSD/train-labels",
    )

    parser.add_argument(
        "--train-mean",
        default="datasets/EORSSD/train-mean",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/repro_probe",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable strict PyTorch/CUDA deterministic mode. "
            "Run the normal probe first."
        ),
    )

    # Internal subprocess options.
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--run-label",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--trace-path",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def set_seed(
    seed: int,
    deterministic: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(
            True,
            warn_only=False,
        )


def get_package_version(
    package_name: str,
) -> str:
    try:
        return importlib.metadata.version(
            package_name
        )
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


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


def sample_tensor(
    tensor: Tensor,
    sample_count: int,
) -> Tensor:
    flat = (
        tensor
        .detach()
        .reshape(-1)
    )

    element_count = flat.numel()

    if element_count == 0:
        return torch.empty(
            0,
            dtype=torch.float32,
            device=flat.device,
        )

    if element_count <= sample_count:
        return flat.float()

    indices = torch.linspace(
        0,
        element_count - 1,
        steps=sample_count,
        device=flat.device,
    ).long()

    return flat[
        indices
    ].float()


def tensor_fingerprint(
    tensor: Tensor,
    sample_count: int = 4096,
) -> str:
    sample = sample_tensor(
        tensor,
        sample_count=sample_count,
    )

    sample_cpu = (
        sample
        .cpu()
        .contiguous()
        .numpy()
    )

    digest = hashlib.sha256()
    digest.update(
        sample_cpu.tobytes()
    )

    return digest.hexdigest()


def tensors_fingerprint(
    tensors: dict[str, Tensor],
    sample_count: int = 1024,
) -> str:
    digest = hashlib.sha256()

    for name in sorted(tensors):
        tensor = tensors[name]

        digest.update(
            name.encode("utf-8")
        )

        sample = sample_tensor(
            tensor,
            sample_count=sample_count,
        )

        sample_cpu = (
            sample
            .cpu()
            .contiguous()
            .numpy()
        )

        digest.update(
            sample_cpu.tobytes()
        )

    return digest.hexdigest()


def model_parameter_fingerprint(
    model: nn.Module,
    samples_per_parameter: int = 16,
) -> str:
    digest = hashlib.sha256()

    sampled_values: list[Tensor] = []

    for name, parameter in (
        model.named_parameters()
    ):
        digest.update(
            name.encode("utf-8")
        )

        sampled_values.append(
            sample_tensor(
                parameter,
                sample_count=(
                    samples_per_parameter
                ),
            )
        )

    if sampled_values:
        combined = torch.cat(
            sampled_values
        )

        combined_cpu = (
            combined
            .cpu()
            .contiguous()
            .numpy()
        )

        digest.update(
            combined_cpu.tobytes()
        )

    return digest.hexdigest()


def model_gradient_fingerprint(
    model: nn.Module,
    samples_per_parameter: int = 16,
) -> str:
    digest = hashlib.sha256()

    sampled_values: list[Tensor] = []

    for name, parameter in (
        model.named_parameters()
    ):
        digest.update(
            name.encode("utf-8")
        )

        if parameter.grad is None:
            digest.update(
                b"NO_GRAD"
            )
            continue

        sampled_values.append(
            sample_tensor(
                parameter.grad,
                sample_count=(
                    samples_per_parameter
                ),
            )
        )

    if sampled_values:
        combined = torch.cat(
            sampled_values
        )

        combined_cpu = (
            combined
            .cpu()
            .contiguous()
            .numpy()
        )

        digest.update(
            combined_cpu.tobytes()
        )

    return digest.hexdigest()


def normalize_batch_names(
    batch: dict,
) -> list[str]:
    names = batch.get(
        "name",
        [],
    )

    if isinstance(
        names,
        str,
    ):
        return [names]

    return [
        str(name)
        for name in names
    ]


def build_model(
    network_path: str,
) -> nn.Module:
    network_module = (
        importlib.import_module(
            network_path
        )
    )

    model = (
        network_module.build_model()
    )

    return model


def environment_info(
    device: torch.device,
    amp_dtype: torch.dtype | None,
    deterministic: bool,
) -> dict:
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mamba_ssm": get_package_version(
            "mamba-ssm"
        ),
        "timm": get_package_version(
            "timm"
        ),
        "device": str(device),
        "amp_dtype": (
            str(amp_dtype)
            if amp_dtype is not None
            else "float32"
        ),
        "deterministic_requested": (
            deterministic
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),
        "cuda_matmul_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_tf32": (
            torch.backends.cudnn.allow_tf32
        ),
        "cublas_workspace_config": (
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG",
                "<unset>",
            )
        ),
    }

    if device.type == "cuda":
        info["device_name"] = (
            torch.cuda.get_device_name(
                device
            )
        )

    return info


def run_single(
    args: argparse.Namespace,
) -> None:
    if (
        args.run_label is None
        or args.trace_path is None
    ):
        raise ValueError(
            "Internal single-run mode requires "
            "--run-label and --trace-path."
        )

    set_seed(
        seed=args.seed,
        deterministic=(
            args.deterministic
        ),
    )

    device = torch.device(
        args.device
    )

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    amp_dtype = get_amp_dtype(
        device=device,
        use_amp=use_amp,
    )

    print(
        f"\n=== Run {args.run_label} ==="
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Network: {args.network}"
    )

    print(
        f"AMP dtype: "
        f"{amp_dtype if amp_dtype else torch.float32}"
    )

    print(
        f"Deterministic: "
        f"{args.deterministic}"
    )

    model = build_model(
        args.network
    )

    model = model.to(
        device
    )

    model.train()

    input_keys = (
        get_model_input_keys(
            model
        )
    )

    mean_hierarchies = (
        get_model_mean_hierarchies(
            model
        )
    )

    print(
        "Model inputs: "
        + ", ".join(input_keys)
    )

    initial_parameter_fp = (
        model_parameter_fingerprint(
            model
        )
    )

    print(
        "Initial parameter fingerprint: "
        f"{initial_parameter_fp[:16]}"
    )

    train_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        mean_dir=(
            args.train_mean
            if mean_hierarchies
            else None
        ),
        mean_hierarchies=(
            mean_hierarchies
        ),
        image_size=(
            args.image_size,
            args.image_size,
        ),
        augment_8way=(
            args.augment_8way
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    criterion = SODLoss(
        aux_weight=args.aux_weight,
        edge_weight=0.0,
        region_weight=0.0,
    ).to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_grad_scaler = (
        use_amp
        and amp_dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    trace: dict = {
        "label": args.run_label,
        "seed": args.seed,
        "network": args.network,
        "environment": environment_info(
            device=device,
            amp_dtype=amp_dtype,
            deterministic=(
                args.deterministic
            ),
        ),
        "initial_parameter_fingerprint": (
            initial_parameter_fp
        ),
        "steps": [],
    }

    for step_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        if step_index > args.steps:
            break

        names = normalize_batch_names(
            batch
        )

        model_inputs = (
            prepare_model_inputs(
                model=model,
                batch=batch,
                device=device,
            )
        )

        input_fp = tensors_fingerprint(
            model_inputs
        )

        mask = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
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
            )

            loss = loss_dict[
                "loss"
            ]

        prediction_fp = (
            tensor_fingerprint(
                outputs["pred"]
            )
        )

        loss_value = float(
            loss
            .detach()
            .float()
            .item()
        )

        main_value = float(
            loss_dict["loss_main"]
            .detach()
            .float()
            .item()
        )

        aux_value = float(
            loss_dict.get(
                "loss_aux",
                loss.detach().new_zeros(
                    ()
                ),
            )
            .detach()
            .float()
            .item()
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

        gradient_fp = (
            model_gradient_fingerprint(
                model
            )
        )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    args.max_grad_norm
                ),
                error_if_nonfinite=True,
            )
        )

        grad_norm_value = float(
            grad_norm
            .detach()
            .float()
            .item()
        )

        if use_grad_scaler:
            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            optimizer.step()

        parameter_fp = (
            model_parameter_fingerprint(
                model
            )
        )

        step_record = {
            "step": step_index,
            "names": names,
            "input_fingerprint": (
                input_fp
            ),
            "prediction_fingerprint": (
                prediction_fp
            ),
            "loss": loss_value,
            "loss_main": main_value,
            "loss_aux": aux_value,
            "grad_norm": (
                grad_norm_value
            ),
            "gradient_fingerprint": (
                gradient_fp
            ),
            "parameter_fingerprint": (
                parameter_fp
            ),
        }

        trace["steps"].append(
            step_record
        )

        print(
            f"Step {step_index:03d} | "
            f"Loss {loss_value:.9f} | "
            f"Main {main_value:.9f} | "
            f"Aux {aux_value:.9f} | "
            f"Grad {grad_norm_value:.9f} | "
            f"Input {input_fp[:8]} | "
            f"Pred {prediction_fp[:8]} | "
            f"GradFP {gradient_fp[:8]} | "
            f"Param {parameter_fp[:8]}"
        )

    trace_path = Path(
        args.trace_path
    )

    trace_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with trace_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            trace,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Trace saved: {trace_path}"
    )


def classify_divergence(
    trace_a: dict,
    trace_b: dict,
) -> tuple[str, int | None]:
    if (
        trace_a[
            "initial_parameter_fingerprint"
        ]
        != trace_b[
            "initial_parameter_fingerprint"
        ]
    ):
        return (
            "INITIALIZATION",
            None,
        )

    steps_a = trace_a["steps"]
    steps_b = trace_b["steps"]

    step_count = min(
        len(steps_a),
        len(steps_b),
    )

    for index in range(
        step_count
    ):
        a = steps_a[index]
        b = steps_b[index]

        step = a["step"]

        if (
            a["names"]
            != b["names"]
        ):
            return (
                "DATALOADER_ORDER",
                step,
            )

        if (
            a["input_fingerprint"]
            != b["input_fingerprint"]
        ):
            return (
                "DATA_INPUT",
                step,
            )

        if (
            a["prediction_fingerprint"]
            != b["prediction_fingerprint"]
        ):
            return (
                "FORWARD",
                step,
            )

        if (
            a["gradient_fingerprint"]
            != b["gradient_fingerprint"]
        ):
            return (
                "BACKWARD",
                step,
            )

        if (
            a["parameter_fingerprint"]
            != b["parameter_fingerprint"]
        ):
            return (
                "OPTIMIZER_UPDATE",
                step,
            )

    if len(steps_a) != len(steps_b):
        return (
            "RUN_LENGTH",
            step_count + 1,
        )

    return (
        "NONE",
        None,
    )


def print_comparison(
    trace_a: dict,
    trace_b: dict,
) -> None:
    print(
        "\n"
        "========================================"
    )

    print(
        "REPRODUCIBILITY COMPARISON"
    )

    print(
        "========================================"
    )

    init_same = (
        trace_a[
            "initial_parameter_fingerprint"
        ]
        == trace_b[
            "initial_parameter_fingerprint"
        ]
    )

    print(
        "Initial parameters: "
        f"{'MATCH' if init_same else 'DIFFER'}"
    )

    divergence_type, divergence_step = (
        classify_divergence(
            trace_a,
            trace_b,
        )
    )

    steps_a = trace_a["steps"]
    steps_b = trace_b["steps"]

    step_count = min(
        len(steps_a),
        len(steps_b),
    )

    print()
    print(
        "Step | Batch | Input | Forward | "
        "Backward | Update | |Loss diff|"
    )
    print(
        "-" * 76
    )

    for index in range(
        step_count
    ):
        a = steps_a[index]
        b = steps_b[index]

        batch_same = (
            a["names"]
            == b["names"]
        )

        input_same = (
            a["input_fingerprint"]
            == b["input_fingerprint"]
        )

        forward_same = (
            a["prediction_fingerprint"]
            == b["prediction_fingerprint"]
        )

        backward_same = (
            a["gradient_fingerprint"]
            == b["gradient_fingerprint"]
        )

        update_same = (
            a["parameter_fingerprint"]
            == b["parameter_fingerprint"]
        )

        loss_diff = abs(
            a["loss"]
            - b["loss"]
        )

        print(
            f"{a['step']:4d} | "
            f"{'OK' if batch_same else 'DIFF':5s} | "
            f"{'OK' if input_same else 'DIFF':5s} | "
            f"{'OK' if forward_same else 'DIFF':7s} | "
            f"{'OK' if backward_same else 'DIFF':8s} | "
            f"{'OK' if update_same else 'DIFF':6s} | "
            f"{loss_diff:.12e}"
        )

    print()
    print(
        "========================================"
    )

    if divergence_type == "NONE":
        print(
            "RESULT: No exact divergence detected "
            f"within {step_count} steps."
        )

        print(
            "The short training trajectory is "
            "bitwise reproducible."
        )

        print(
            "If full 45-epoch results still differ, "
            "increase --steps to 200 or 500."
        )

    elif divergence_type == "INITIALIZATION":
        print(
            "RESULT: Initial model parameters differ."
        )

        print(
            "Likely area: model initialization, "
            "pretrained loading, RNG state, or "
            "environment/version differences."
        )

    elif divergence_type == "DATALOADER_ORDER":
        print(
            f"RESULT: First divergence at step "
            f"{divergence_step}: batch order."
        )

        print(
            "Likely area: DataLoader shuffle / "
            "RNG stream."
        )

    elif divergence_type == "DATA_INPUT":
        print(
            f"RESULT: First divergence at step "
            f"{divergence_step}: input tensors."
        )

        print(
            "Batch names match, but tensor values "
            "differ. Check dataset preprocessing "
            "or worker-side randomness."
        )

    elif divergence_type == "FORWARD":
        print(
            f"RESULT: First divergence at step "
            f"{divergence_step}: forward pass."
        )

        print(
            "Inputs are identical, but model outputs "
            "differ. CUDA / AMP / Mamba forward "
            "non-determinism is the primary suspect."
        )

    elif divergence_type == "BACKWARD":
        print(
            f"RESULT: First divergence at step "
            f"{divergence_step}: backward pass."
        )

        print(
            "Forward outputs match, but gradients "
            "differ. CUDA backward / selective scan "
            "backward is the primary suspect."
        )

    elif divergence_type == "OPTIMIZER_UPDATE":
        print(
            f"RESULT: First divergence at step "
            f"{divergence_step}: optimizer update."
        )

        print(
            "Gradients match, but parameters differ "
            "after AdamW step."
        )

    else:
        print(
            f"RESULT: Divergence type: "
            f"{divergence_type}"
        )

    print(
        "========================================"
    )


def run_parent(
    args: argparse.Namespace,
) -> None:
    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_a_path = (
        output_dir
        / "trace_A.json"
    )

    trace_b_path = (
        output_dir
        / "trace_B.json"
    )

    script_path = Path(
        __file__
    ).resolve()

    base_arguments = list(
        sys.argv[1:]
    )

    env = os.environ.copy()

    if args.deterministic:
        env[
            "CUBLAS_WORKSPACE_CONFIG"
        ] = ":4096:8"

    def launch(
        label: str,
        trace_path: Path,
    ) -> None:
        command = [
            sys.executable,
            str(script_path),
            *base_arguments,
            "--single-run",
            "--run-label",
            label,
            "--trace-path",
            str(trace_path),
        ]

        print(
            "\n"
            "========================================"
        )

        print(
            f"Launching run {label}"
        )

        print(
            "========================================"
        )

        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
        )

    launch(
        label="A",
        trace_path=trace_a_path,
    )

    launch(
        label="B",
        trace_path=trace_b_path,
    )

    with trace_a_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        trace_a = json.load(
            file
        )

    with trace_b_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        trace_b = json.load(
            file
        )

    print_comparison(
        trace_a=trace_a,
        trace_b=trace_b,
    )

    comparison_path = (
        output_dir
        / "comparison.json"
    )

    divergence_type, divergence_step = (
        classify_divergence(
            trace_a,
            trace_b,
        )
    )

    with comparison_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "divergence_type": (
                    divergence_type
                ),
                "divergence_step": (
                    divergence_step
                ),
                "trace_a": str(
                    trace_a_path
                ),
                "trace_b": str(
                    trace_b_path
                ),
            },
            file,
            indent=2,
        )

    print(
        f"\nComparison saved: "
        f"{comparison_path}"
    )


def main() -> None:
    args = parse_args()

    if args.single_run:
        run_single(
            args
        )
    else:
        run_parent(
            args
        )


if __name__ == "__main__":
    main()