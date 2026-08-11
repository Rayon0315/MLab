# tools/dictionary_test.py
from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from data.dataset import SODDataset


class RunningStats:
    def __init__(self) -> None:
        self.count = 0

        self.total = 0.0
        self.total_sq = 0.0

        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(
        self,
        values: torch.Tensor,
    ) -> None:
        values = (
            values
            .detach()
            .float()
        )

        if values.numel() == 0:
            return

        self.count += values.numel()

        self.total += (
            values.sum().item()
        )

        self.total_sq += (
            values
            .square()
            .sum()
            .item()
        )

        self.minimum = min(
            self.minimum,
            values.min().item(),
        )

        self.maximum = max(
            self.maximum,
            values.max().item(),
        )

    def result(
        self,
    ) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        mean = (
            self.total
            / self.count
        )

        variance = max(
            (
                self.total_sq
                / self.count
            )
            - mean * mean,
            0.0,
        )

        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(
                variance
            ),
            "min": self.minimum,
            "max": self.maximum,
        }


class AssignmentStats:
    def __init__(
        self,
        num_prototypes: int,
    ) -> None:
        self.num_prototypes = (
            num_prototypes
        )

        self.soft_sum = torch.zeros(
            num_prototypes,
            dtype=torch.float64,
        )

        self.hard_count = torch.zeros(
            num_prototypes,
            dtype=torch.int64,
        )

        self.positions = 0

        self.entropy = RunningStats()

        self.normalized_entropy = (
            RunningStats()
        )

        self.confidence = (
            RunningStats()
        )

        self.margin = RunningStats()

    def update(
        self,
        assignment: torch.Tensor,
    ) -> None:
        assignment = (
            assignment
            .detach()
            .float()
        )

        (
            _,
            num_prototypes,
            _,
            _,
        ) = assignment.shape

        if (
            num_prototypes
            != self.num_prototypes
        ):
            raise ValueError(
                "Prototype count mismatch: "
                f"expected {self.num_prototypes}, "
                f"got {num_prototypes}."
            )

        self.soft_sum += (
            assignment
            .sum(
                dim=(0, 2, 3)
            )
            .double()
            .cpu()
        )

        hard_index = (
            assignment.argmax(
                dim=1
            )
        )

        self.hard_count += (
            torch.bincount(
                hard_index
                .reshape(-1)
                .cpu(),
                minlength=(
                    self.num_prototypes
                ),
            )
        )

        position_count = (
            hard_index.numel()
        )

        self.positions += (
            position_count
        )

        probability = (
            assignment.clamp_min(
                1e-8
            )
        )

        entropy = -(
            probability
            * probability.log()
        ).sum(
            dim=1
        )

        self.entropy.update(
            entropy
        )

        self.normalized_entropy.update(
            entropy
            / math.log(
                self.num_prototypes
            )
        )

        top2 = torch.topk(
            assignment,
            k=2,
            dim=1,
        ).values

        self.confidence.update(
            top2[:, 0]
        )

        self.margin.update(
            top2[:, 0]
            - top2[:, 1]
        )

    def result(
        self,
    ) -> dict:
        if self.positions == 0:
            soft_usage = (
                [0.0]
                * self.num_prototypes
            )

            hard_usage = (
                [0.0]
                * self.num_prototypes
            )

        else:
            soft_usage = (
                self.soft_sum
                / self.positions
            ).tolist()

            hard_usage = (
                self.hard_count.double()
                / self.positions
            ).tolist()

        return {
            "positions": self.positions,
            "soft_usage": soft_usage,
            "hard_usage": hard_usage,
            "entropy": (
                self.entropy.result()
            ),
            "normalized_entropy": (
                self.normalized_entropy
                .result()
            ),
            "confidence": (
                self.confidence.result()
            ),
            "top1_top2_margin": (
                self.margin.result()
            ),
        }


class AgreementStats:
    def __init__(self) -> None:
        self.argmax_agreement = (
            RunningStats()
        )

        self.total_variation = (
            RunningStats()
        )

    def update(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> None:
        first = (
            first
            .detach()
            .float()
        )

        second = (
            second
            .detach()
            .float()
        )

        agreement = (
            first.argmax(
                dim=1
            )
            == second.argmax(
                dim=1
            )
        ).float()

        self.argmax_agreement.update(
            agreement
        )

        total_variation = (
            0.5
            * (
                first
                - second
            )
            .abs()
            .sum(
                dim=1
            )
        )

        self.total_variation.update(
            total_variation
        )

    def result(
        self,
    ) -> dict:
        return {
            "argmax_agreement": (
                self.argmax_agreement
                .result()
            ),
            "total_variation": (
                self.total_variation
                .result()
            ),
        }


class RouterScaleStats:
    def __init__(self) -> None:
        self.values = RunningStats()

        self.absolute_delta = (
            RunningStats()
        )

        self.spatial_std = (
            RunningStats()
        )

        self.channel_std = (
            RunningStats()
        )

        self.sample_mean = (
            RunningStats()
        )

    def update(
        self,
        scale: torch.Tensor,
    ) -> None:
        scale = (
            scale
            .detach()
            .float()
        )

        self.values.update(
            scale
        )

        self.absolute_delta.update(
            (
                scale
                - 1.0
            ).abs()
        )

        self.spatial_std.update(
            scale.std(
                dim=(-2, -1),
                unbiased=False,
            )
        )

        self.channel_std.update(
            scale.std(
                dim=1,
                unbiased=False,
            )
        )

        self.sample_mean.update(
            scale.mean(
                dim=(1, 2, 3)
            )
        )

    def result(
        self,
    ) -> dict:
        return {
            "scale": (
                self.values.result()
            ),
            "absolute_delta_from_1": (
                self.absolute_delta
                .result()
            ),
            "spatial_std": (
                self.spatial_std.result()
            ),
            "channel_std": (
                self.channel_std.result()
            ),
            "sample_mean": (
                self.sample_mean.result()
            ),
        }


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose latent dictionary "
            "usage and routing behavior."
        )
    )

    parser.add_argument(
        "--network",
        default=(
            "models.networks."
            "mambavision_small_progressive_"
            "region_direct_dictionary_"
            "routing_sod"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/"
            "mv_progressive_region_direct_"
            "dictionary_routing_"
            "eorssd_aug8_e45/"
            "checkpoints/final.pth"
        ),
    )

    parser.add_argument(
        "--test-images",
        default=(
            "datasets/EORSSD/"
            "test-images"
        ),
    )

    parser.add_argument(
        "--test-masks",
        default=(
            "datasets/EORSSD/"
            "test-labels"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "runs/"
            "mv_progressive_region_direct_"
            "dictionary_routing_"
            "eorssd_aug8_e45/"
            "dictionary_test"
        ),
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
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--amp",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    return parser.parse_args()


def setup_logging(
    output_dir: Path,
) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                (
                    output_dir
                    / "dictionary_test.log"
                ),
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    return logging.getLogger(
        __name__
    )


def build_model(
    network_path: str,
) -> nn.Module:
    network_module = (
        importlib.import_module(
            network_path
        )
    )

    return (
        network_module
        .build_model()
    )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    network_path: str,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_network = (
        checkpoint.get(
            "network"
        )
    )

    if (
        checkpoint_network
        != network_path
    ):
        raise RuntimeError(
            "Checkpoint network does "
            "not match:\n"
            f"checkpoint: "
            f"{checkpoint_network}\n"
            f"command: "
            f"{network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return checkpoint


def get_assignments(
    model: nn.Module,
    image: torch.Tensor,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]:
    (
        stage1,
        stage2,
        stage3,
        stage4,
    ) = model.backbone(
        image
    )

    dictionary = (
        model.latent_type_dictionary
    )

    latent2 = (
        dictionary
        .stage2_projection(
            stage2
        )
    )

    latent3 = (
        dictionary
        .stage3_projection(
            stage3
        )
    )

    latent4 = (
        dictionary
        .stage4_projection(
            stage4
        )
    )

    assignment2 = (
        dictionary._assign(
            latent2
        )
    )

    assignment3 = (
        dictionary._assign(
            latent3
        )
    )

    assignment4 = (
        dictionary._assign(
            latent4
        )
    )

    target_size = (
        assignment2.shape[-2:]
    )

    assignment3_up = (
        F.interpolate(
            assignment3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    )

    assignment4_up = (
        F.interpolate(
            assignment4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    )

    shared = (
        assignment2
        + assignment3_up
        + assignment4_up
    ) / 3.0

    return (
        (
            stage1,
            stage2,
            stage3,
            stage4,
        ),
        (
            assignment2,
            assignment3,
            assignment4,
            shared,
        ),
    )


def compute_router_scale(
    type_field: torch.Tensor,
    router_module: nn.Module,
    stream_name: str,
    target_size: tuple[int, int],
) -> torch.Tensor:
    resized = F.interpolate(
        type_field,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    router = getattr(
        router_module,
        f"{stream_name}_router",
    )

    logits = router(
        resized
    )

    return (
        1.0
        + router_module.routing_strength
        * torch.tanh(
            logits
        )
    )


def build_router_scales(
    model: nn.Module,
    type_field: torch.Tensor,
    stages: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> dict[
    str,
    dict[
        str,
        torch.Tensor,
    ],
]:
    (
        stage1,
        stage2,
        stage3,
        stage4,
    ) = stages

    specifications = {
        "stage3": {
            "module": (
                model
                .dictionary_router3
            ),
            "low": (
                stage3.shape[-2:]
            ),
            "high": (
                stage4.shape[-2:]
            ),
            "global": (
                stage3.shape[-2:]
            ),
        },
        "stage2": {
            "module": (
                model
                .dictionary_router2
            ),
            "low": (
                stage2.shape[-2:]
            ),
            "high": (
                stage3.shape[-2:]
            ),
            "global": (
                stage2.shape[-2:]
            ),
        },
        "stage1": {
            "module": (
                model
                .dictionary_router1
            ),
            "low": (
                stage1.shape[-2:]
            ),
            "high": (
                stage2.shape[-2:]
            ),
            "global": (
                stage1.shape[-2:]
            ),
        },
    }

    scales: dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ] = {}

    for (
        stage_name,
        specification,
    ) in specifications.items():
        router_module = (
            specification[
                "module"
            ]
        )

        scales[
            stage_name
        ] = {}

        for stream_name in (
            "low",
            "high",
            "global",
        ):
            scales[
                stage_name
            ][
                stream_name
            ] = (
                compute_router_scale(
                    type_field=(
                        type_field
                    ),
                    router_module=(
                        router_module
                    ),
                    stream_name=(
                        stream_name
                    ),
                    target_size=(
                        specification[
                            stream_name
                        ]
                    ),
                )
            )

    return scales


def prototype_similarity(
    model: nn.Module,
) -> torch.Tensor:
    prototypes = (
        model
        .latent_type_dictionary
        .prototypes
        .detach()
        .float()
    )

    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    )

    return (
        prototypes
        @ prototypes.transpose(
            0,
            1,
        )
    )


def summarize_prototype_similarity(
    similarity: torch.Tensor,
) -> dict:
    num_prototypes = (
        similarity.shape[0]
    )

    off_diagonal_mask = (
        ~torch.eye(
            num_prototypes,
            dtype=torch.bool,
            device=(
                similarity.device
            ),
        )
    )

    off_diagonal = (
        similarity[
            off_diagonal_mask
        ]
    )

    without_diagonal = (
        similarity.clone()
    )

    without_diagonal.fill_diagonal_(
        -float("inf")
    )

    (
        nearest_similarity,
        nearest_index,
    ) = (
        without_diagonal.max(
            dim=1
        )
    )

    return {
        "off_diagonal_mean": (
            off_diagonal
            .mean()
            .item()
        ),
        "off_diagonal_std": (
            off_diagonal
            .std(
                unbiased=False
            )
            .item()
        ),
        "off_diagonal_min": (
            off_diagonal
            .min()
            .item()
        ),
        "off_diagonal_max": (
            off_diagonal
            .max()
            .item()
        ),
        "nearest_prototype": [
            {
                "prototype": index,
                "nearest": (
                    nearest_index[
                        index
                    ].item()
                ),
                "cosine": (
                    nearest_similarity[
                        index
                    ].item()
                ),
            }
            for index
            in range(
                num_prototypes
            )
        ],
    }


def save_prototype_similarity_csv(
    path: Path,
    similarity: torch.Tensor,
) -> None:
    matrix = (
        similarity
        .cpu()
        .tolist()
    )

    num_prototypes = len(
        matrix
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            ["prototype"]
            + [
                f"P{index}"
                for index
                in range(
                    num_prototypes
                )
            ]
        )

        for (
            index,
            row,
        ) in enumerate(
            matrix
        ):
            writer.writerow(
                [
                    f"P{index}"
                ]
                + row
            )


def save_prototype_usage_csv(
    path: Path,
    assignment_results: dict[
        str,
        dict,
    ],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "source",
                "prototype",
                "soft_usage",
                "hard_usage",
            ]
        )

        for (
            source_name,
            result,
        ) in assignment_results.items():

            for (
                prototype_index,
                (
                    soft_usage,
                    hard_usage,
                ),
            ) in enumerate(
                zip(
                    result[
                        "soft_usage"
                    ],
                    result[
                        "hard_usage"
                    ],
                )
            ):
                writer.writerow(
                    [
                        source_name,
                        prototype_index,
                        soft_usage,
                        hard_usage,
                    ]
                )


def save_router_csv(
    path: Path,
    router_results: dict[
        str,
        dict[
            str,
            dict,
        ],
    ],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "stage",
                "stream",
                "mean",
                "std",
                "min",
                "max",
                "mean_abs_delta_from_1",
                "spatial_std_mean",
                "channel_std_mean",
                "sample_mean_std",
            ]
        )

        for (
            stage_name,
            streams,
        ) in router_results.items():

            for (
                stream_name,
                result,
            ) in streams.items():

                writer.writerow(
                    [
                        stage_name,
                        stream_name,
                        result[
                            "scale"
                        ][
                            "mean"
                        ],
                        result[
                            "scale"
                        ][
                            "std"
                        ],
                        result[
                            "scale"
                        ][
                            "min"
                        ],
                        result[
                            "scale"
                        ][
                            "max"
                        ],
                        result[
                            "absolute_delta_from_1"
                        ][
                            "mean"
                        ],
                        result[
                            "spatial_std"
                        ][
                            "mean"
                        ],
                        result[
                            "channel_std"
                        ][
                            "mean"
                        ],
                        result[
                            "sample_mean"
                        ][
                            "std"
                        ],
                    ]
                )


def save_summary(
    path: Path,
    results: dict,
) -> None:
    lines: list[str] = []

    lines.append(
        "Dictionary diagnostic summary"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Network: "
        f"{results['network']}"
    )

    lines.append(
        f"Checkpoint: "
        f"{results['checkpoint']}"
    )

    lines.append(
        f"Samples: "
        f"{results['samples']}"
    )

    lines.append(
        f"Prototypes: "
        f"{results['num_prototypes']}"
    )

    lines.append(
        f"Dictionary dim: "
        f"{results['dictionary_dim']}"
    )

    lines.append(
        f"Temperature: "
        f"{results['temperature']}"
    )

    lines.append("")

    lines.append(
        "Prototype similarity"
    )

    lines.append(
        "-" * 80
    )

    similarity = (
        results[
            "prototype_similarity"
        ]
    )

    lines.append(
        "Off-diagonal cosine | "
        f"mean "
        f"{similarity['off_diagonal_mean']:.6f} | "
        f"std "
        f"{similarity['off_diagonal_std']:.6f} | "
        f"min "
        f"{similarity['off_diagonal_min']:.6f} | "
        f"max "
        f"{similarity['off_diagonal_max']:.6f}"
    )

    lines.append("")

    lines.append(
        "Assignment statistics"
    )

    lines.append(
        "-" * 80
    )

    for (
        source_name,
        source_result,
    ) in (
        results[
            "assignments"
        ].items()
    ):
        lines.append(
            f"{source_name:>7} | "
            f"entropy "
            f"{source_result['normalized_entropy']['mean']:.6f} | "
            f"confidence "
            f"{source_result['confidence']['mean']:.6f} | "
            f"margin "
            f"{source_result['top1_top2_margin']['mean']:.6f}"
        )

        soft_usage_text = (
            ", ".join(
                (
                    f"P{index}="
                    f"{value:.4f}"
                )
                for (
                    index,
                    value,
                ) in enumerate(
                    source_result[
                        "soft_usage"
                    ]
                )
            )
        )

        hard_usage_text = (
            ", ".join(
                (
                    f"P{index}="
                    f"{value:.4f}"
                )
                for (
                    index,
                    value,
                ) in enumerate(
                    source_result[
                        "hard_usage"
                    ]
                )
            )
        )

        lines.append(
            "         soft: "
            + soft_usage_text
        )

        lines.append(
            "         hard: "
            + hard_usage_text
        )

    lines.append("")

    lines.append(
        "Cross-scale agreement"
    )

    lines.append(
        "-" * 80
    )

    for (
        pair_name,
        pair_result,
    ) in (
        results[
            "agreement"
        ].items()
    ):
        lines.append(
            f"{pair_name:>7} | "
            "argmax agreement "
            f"{pair_result['argmax_agreement']['mean']:.6f} | "
            "total variation "
            f"{pair_result['total_variation']['mean']:.6f}"
        )

    lines.append("")

    lines.append(
        "Router scale statistics"
    )

    lines.append(
        "-" * 80
    )

    for (
        stage_name,
        streams,
    ) in (
        results[
            "routers"
        ].items()
    ):
        for (
            stream_name,
            stream_result,
        ) in streams.items():

            scale = (
                stream_result[
                    "scale"
                ]
            )

            lines.append(
                f"{stage_name}/"
                f"{stream_name:<6} | "
                f"mean "
                f"{scale['mean']:.6f} | "
                f"std "
                f"{scale['std']:.6f} | "
                f"min "
                f"{scale['min']:.6f} | "
                f"max "
                f"{scale['max']:.6f} | "
                f"abs_delta "
                f"{stream_result['absolute_delta_from_1']['mean']:.6f} | "
                f"spatial_std "
                f"{stream_result['spatial_std']['mean']:.6f} | "
                f"channel_std "
                f"{stream_result['channel_std']['mean']:.6f}"
            )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        output_dir
    )

    with (
        output_dir
        / "dictionary_test_args.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
        )

    device = torch.device(
        args.device
    )

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    model = build_model(
        args.network
    )

    checkpoint = (
        load_checkpoint(
            checkpoint_path=(
                args.checkpoint
            ),
            model=model,
            network_path=(
                args.network
            ),
        )
    )

    model = model.to(
        device
    )

    model.eval()

    if not hasattr(
        model,
        "latent_type_dictionary",
    ):
        raise AttributeError(
            "Model does not contain "
            "latent_type_dictionary."
        )

    for router_name in (
        "dictionary_router1",
        "dictionary_router2",
        "dictionary_router3",
    ):
        if not hasattr(
            model,
            router_name,
        ):
            raise AttributeError(
                "Model does not contain "
                f"{router_name}."
            )

    dictionary = (
        model
        .latent_type_dictionary
    )

    num_prototypes = (
        dictionary
        .num_prototypes
    )

    dataset = SODDataset(
        image_dir=(
            args.test_images
        ),
        mask_dir=(
            args.test_masks
        ),
        image_size=(
            args.image_size,
            args.image_size,
        ),
    )

    if (
        args.max_samples
        is not None
    ):
        dataset = Subset(
            dataset,
            range(
                min(
                    args.max_samples,
                    len(dataset),
                )
            ),
        )

    data_loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    assignment_stats = {
        "stage2": (
            AssignmentStats(
                num_prototypes
            )
        ),
        "stage3": (
            AssignmentStats(
                num_prototypes
            )
        ),
        "stage4": (
            AssignmentStats(
                num_prototypes
            )
        ),
        "shared": (
            AssignmentStats(
                num_prototypes
            )
        ),
    }

    agreement_stats = {
        "S2-S3": (
            AgreementStats()
        ),
        "S2-S4": (
            AgreementStats()
        ),
        "S3-S4": (
            AgreementStats()
        ),
    }

    router_stats = {
        stage_name: {
            stream_name: (
                RouterScaleStats()
            )
            for stream_name
            in (
                "low",
                "high",
                "global",
            )
        }
        for stage_name
        in (
            "stage3",
            "stage2",
            "stage1",
        )
    }

    sample_count = 0

    progress = tqdm(
        data_loader,
        desc="Dictionary test",
        unit="batch",
        dynamic_ncols=True,
        miniters=(
            args.log_interval
        ),
    )

    for batch in progress:
        image = (
            batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type=(
                device.type
            ),
            dtype=torch.float16,
            enabled=use_amp,
        ):
            (
                stages,
                assignments,
            ) = get_assignments(
                model=model,
                image=image,
            )

            (
                assignment2,
                assignment3,
                assignment4,
                shared,
            ) = assignments

            assignment3_up = (
                F.interpolate(
                    assignment3,
                    size=(
                        assignment2
                        .shape[-2:]
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            assignment4_up = (
                F.interpolate(
                    assignment4,
                    size=(
                        assignment2
                        .shape[-2:]
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            scales = (
                build_router_scales(
                    model=model,
                    type_field=shared,
                    stages=stages,
                )
            )

        assignment_stats[
            "stage2"
        ].update(
            assignment2
        )

        assignment_stats[
            "stage3"
        ].update(
            assignment3
        )

        assignment_stats[
            "stage4"
        ].update(
            assignment4
        )

        assignment_stats[
            "shared"
        ].update(
            shared
        )

        agreement_stats[
            "S2-S3"
        ].update(
            assignment2,
            assignment3_up,
        )

        agreement_stats[
            "S2-S4"
        ].update(
            assignment2,
            assignment4_up,
        )

        agreement_stats[
            "S3-S4"
        ].update(
            assignment3_up,
            assignment4_up,
        )

        for (
            stage_name,
            streams,
        ) in scales.items():

            for (
                stream_name,
                scale,
            ) in streams.items():

                router_stats[
                    stage_name
                ][
                    stream_name
                ].update(
                    scale
                )

        sample_count += (
            image.shape[0]
        )

        progress.set_postfix(
            samples=sample_count
        )

    similarity = (
        prototype_similarity(
            model
        )
    )

    assignment_results = {
        name: stats.result()
        for (
            name,
            stats,
        ) in (
            assignment_stats
            .items()
        )
    }

    agreement_results = {
        name: stats.result()
        for (
            name,
            stats,
        ) in (
            agreement_stats
            .items()
        )
    }

    router_results = {
        stage_name: {
            stream_name: (
                stats.result()
            )
            for (
                stream_name,
                stats,
            ) in streams.items()
        }
        for (
            stage_name,
            streams,
        ) in router_stats.items()
    }

    results = {
        "network": (
            args.network
        ),
        "checkpoint": (
            args.checkpoint
        ),
        "checkpoint_epoch": (
            checkpoint.get(
                "epoch"
            )
        ),
        "samples": (
            sample_count
        ),
        "num_prototypes": (
            num_prototypes
        ),
        "dictionary_dim": (
            dictionary
            .dictionary_dim
        ),
        "temperature": (
            dictionary
            .temperature
        ),
        "prototype_similarity": (
            summarize_prototype_similarity(
                similarity
            )
        ),
        "assignments": (
            assignment_results
        ),
        "agreement": (
            agreement_results
        ),
        "routers": (
            router_results
        ),
    }

    with (
        output_dir
        / "dictionary_stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_prototype_similarity_csv(
        (
            output_dir
            / "prototype_cosine.csv"
        ),
        similarity,
    )

    save_prototype_usage_csv(
        (
            output_dir
            / "prototype_usage.csv"
        ),
        assignment_results,
    )

    save_router_csv(
        (
            output_dir
            / "router_scales.csv"
        ),
        router_results,
    )

    save_summary(
        (
            output_dir
            / "summary.txt"
        ),
        results,
    )

    logger.info(
        "Dictionary test completed | "
        "Samples %d",
        sample_count,
    )

    logger.info(
        "Output: %s",
        output_dir,
    )

    logger.info(
        "Summary: %s",
        (
            output_dir
            / "summary.txt"
        ),
    )

    logger.info(
        "Statistics: %s",
        (
            output_dir
            / "dictionary_stats.json"
        ),
    )

    logger.info(
        "Prototype usage: %s",
        (
            output_dir
            / "prototype_usage.csv"
        ),
    )

    logger.info(
        "Prototype cosine: %s",
        (
            output_dir
            / "prototype_cosine.csv"
        ),
    )

    logger.info(
        "Router scales: %s",
        (
            output_dir
            / "router_scales.csv"
        ),
    )


if __name__ == "__main__":
    main()