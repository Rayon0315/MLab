# models/networks/cfmnet_stage2_cross_matching_unetformer_sod.py

from __future__ import annotations

from pathlib import Path
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
    ConvBNReLU,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class Stage2BranchTap:
    """
    Capture TCRM / SSRM / LDFM / EGPCM from the last block
    of paper Stage 2.
    """

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        block = (
            backbone
            .stages[2]
            .blocks[-1]
        )

        self._outputs: dict[
            str,
            torch.Tensor,
        ] = {}

        self._handles = []

        for (
            name,
            module,
        ) in (
            ("tcrm", block.TCRM),
            ("ssrm", block.SSRM),
            ("ldfm", block.LDFM),
            ("egpcm", block.EGPCM),
        ):
            self._register(
                name,
                module,
            )

    def _register(
        self,
        name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[
                name
            ] = output

        self._handles.append(
            module.register_forward_hook(
                hook
            )
        )

    def clear(
        self,
    ) -> None:
        self._outputs.clear()

    def pop(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        expected = {
            "tcrm",
            "ssrm",
            "ldfm",
            "egpcm",
        }

        missing = (
            expected
            - self._outputs.keys()
        )

        if missing:
            raise RuntimeError(
                "Failed to capture Stage2 branches: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        outputs = self._outputs
        self._outputs = {}

        return outputs


class Stage2LocalCrossMatching(
    nn.Module
):
    """
    Local Q/K/V matching at Stage 2.

    Q:
        upsampled Stage-3 decoder semantic feature

    K:
        Stage2 TCRM + EGPCM semantic representation

    V:
        Stage2 SSRM + LDFM structure/detail representation

    For every Stage2 location, Q searches a local KxK neighborhood
    of semantic keys and gathers the corresponding structure/detail
    values. This avoids a single scalar affinity map and avoids
    full HW x HW global attention.

    The result is a residual on top of the original UNetFormer
    Stage2 weighted-fusion output.
    """

    def __init__(
        self,
        branch_pair_channels: int,
        decode_channels: int = 64,
        window_size: int = 7,
    ) -> None:
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError(
                "window_size must be odd."
            )

        self.window_size = (
            window_size
        )

        self.padding = (
            window_size
            // 2
        )

        self.decode_channels = (
            decode_channels
        )

        self.query_proj = (
            ConvBNReLU(
                decode_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.key_proj = (
            ConvBNReLU(
                branch_pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.value_proj = (
            ConvBNReLU(
                branch_pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.output_proj = (
            nn.Sequential(
                ConvBNReLU(
                    decode_channels,
                    decode_channels,
                    kernel_size=3,
                ),
                nn.Conv2d(
                    decode_channels,
                    decode_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    decode_channels
                ),
            )
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        stage2_base: torch.Tensor,
        decoder_stage3: torch.Tensor,
        branches: dict[
            str,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic = torch.cat(
            [
                branches["tcrm"],
                branches["egpcm"],
            ],
            dim=1,
        )

        structure = torch.cat(
            [
                branches["ssrm"],
                branches["ldfm"],
            ],
            dim=1,
        )

        query = F.interpolate(
            decoder_stage3,
            size=stage2_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        query = self.query_proj(
            query
        )

        key = self.key_proj(
            semantic
        )

        value = self.value_proj(
            structure
        )

        (
            batch_size,
            channels,
            height,
            width,
        ) = query.shape

        patch_count = (
            self.window_size
            * self.window_size
        )

        key_patches = F.unfold(
            key,
            kernel_size=self.window_size,
            padding=self.padding,
        )

        key_patches = (
            key_patches.view(
                batch_size,
                channels,
                patch_count,
                height,
                width,
            )
        )

        value_patches = F.unfold(
            value,
            kernel_size=self.window_size,
            padding=self.padding,
        )

        value_patches = (
            value_patches.view(
                batch_size,
                channels,
                patch_count,
                height,
                width,
            )
        )

        query = query.unsqueeze(
            2
        )

        attention = (
            query
            * key_patches
        ).sum(
            dim=1
        )

        attention = (
            attention
            / math.sqrt(
                channels
            )
        )

        attention = F.softmax(
            attention,
            dim=1,
        )

        matched = (
            value_patches
            * attention.unsqueeze(
                1
            )
        ).sum(
            dim=2
        )

        residual = (
            self.output_proj(
                matched
            )
        )

        output = (
            stage2_base
            + self.scale
            * residual
        )

        # Mean peak probability is useful for diagnosing whether
        # the local matching becomes sharp or stays diffuse.
        peak = (
            attention.max(
                dim=1
            )
            .values
            .mean()
        )

        return (
            output,
            peak,
        )


class CFMNetStage2CrossMatchingDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Experiment 3:
    keep the complete UNetFormer decoder and only replace the
    Stage2 branch-aware interaction with local Q/K/V matching.
    """

    def __init__(
        self,
        encoder_channels: tuple[
            int,
            int,
            int,
            int,
        ] = CFMNET_OUT_CHANNELS,
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 1,
        matching_window: int = 7,
    ) -> None:
        super().__init__(
            encoder_channels=(
                encoder_channels
            ),
            decode_channels=(
                decode_channels
            ),
            dropout=dropout,
            window_size=(
                window_size
            ),
            num_classes=(
                num_classes
            ),
        )

        stage2_channels = (
            encoder_channels[1]
        )

        if stage2_channels % 4 != 0:
            raise ValueError(
                "Stage2 channels must be "
                "divisible by four."
            )

        self.cross_matching = (
            Stage2LocalCrossMatching(
                branch_pair_channels=(
                    stage2_channels
                    // 2
                ),
                decode_channels=(
                    decode_channels
                ),
                window_size=(
                    matching_window
                ),
            )
        )

        self.last_matching_peak = None

    def forward(
        self,
        features: list[
            torch.Tensor
        ],
        stage2_branches: dict[
            str,
            torch.Tensor,
        ],
        output_size: tuple[
            int,
            int,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = features

        x4 = self.block4(
            self.pre_conv(
                stage4
            )
        )

        x3 = self.block3(
            self.fuse3(
                x4,
                stage3,
            )
        )

        x2_base = self.fuse2(
            x3,
            stage2,
        )

        (
            x2_matched,
            matching_peak,
        ) = self.cross_matching(
            stage2_base=(
                x2_base
            ),
            decoder_stage3=(
                x3
            ),
            branches=(
                stage2_branches
            ),
        )

        self.last_matching_peak = (
            matching_peak.detach()
        )

        x2 = self.block2(
            x2_matched
        )

        x1 = self.refine1(
            x2,
            stage1,
        )

        prediction = (
            self.segmentation_head(
                x1
            )
        )

        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = None

        if self.training:
            aux_size = (
                x2.shape[-2:]
            )

            aux4 = F.interpolate(
                x4,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            aux3 = F.interpolate(
                x3,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            auxiliary = (
                self.aux_head(
                    aux4
                    + aux3
                    + x2,
                    output_size=(
                        output_size
                    ),
                )
            )

        return (
            prediction,
            auxiliary,
        )

    def matching_scale(
        self,
    ) -> torch.Tensor:
        return (
            self.cross_matching
            .scale
        )


class CFMNetStage2CrossMatchingUNetFormerSOD(
    nn.Module
):
    input_keys = (
        "image",
    )

    def __init__(
        self,
        pretrained_path: str
        | Path
        | None,
        decode_channels: int = 64,
        window_size: int = 8,
        matching_window: int = 7,
    ) -> None:
        super().__init__()

        self.backbone = (
            build_cfmnet(
                pretrained_path=(
                    pretrained_path
                )
            )
        )

        self.stage2_tap = (
            Stage2BranchTap(
                self.backbone
            )
        )

        self.decoder = (
            CFMNetStage2CrossMatchingDecoder(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                decode_channels=(
                    decode_channels
                ),
                dropout=0.1,
                window_size=(
                    window_size
                ),
                num_classes=1,
                matching_window=(
                    matching_window
                ),
            )
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[
            torch.Tensor
        ],
    ]:
        output_size = (
            image.shape[-2:]
        )

        self.stage2_tap.clear()

        features = self.backbone(
            image
        )

        stage2_branches = (
            self.stage2_tap.pop()
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=features,
            stage2_branches=(
                stage2_branches
            ),
            output_size=(
                output_size
            ),
        )

        outputs: dict[
            str,
            torch.Tensor
            | list[
                torch.Tensor
            ],
        ] = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs["aux"] = [
                auxiliary
            ]

        return outputs


def build_model(
) -> CFMNetStage2CrossMatchingUNetFormerSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
        warnings.warn(
            (
                "CFMNet ImageNet-1K checkpoint "
                "was not found at "
                f"{PRETRAINED_PATH}. "
                "The backbone will train "
                "from scratch."
            ),
            RuntimeWarning,
        )

    return (
        CFMNetStage2CrossMatchingUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
            matching_window=7,
        )
    )
