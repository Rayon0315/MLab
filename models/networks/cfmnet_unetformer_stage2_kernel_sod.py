# models/networks/cfmnet_unetformer_stage2_kernel_sod.py

from __future__ import annotations

from pathlib import Path
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
    Capture the four outputs of the last CFMBlock in paper Stage 2.

    CFMNet:
        stages[0] -> Stage 1
        stages[2] -> Stage 2
        stages[4] -> Stage 3
        stages[6] -> Stage 4

    Paper Stage 2 has 192 channels:
        TCRM  48
        SSRM  48
        LDFM  48
        EGPCM 48

    Hooks expose already-computed features only.
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

        self._register(
            "tcrm",
            block.TCRM,
        )
        self._register(
            "ssrm",
            block.SSRM,
        )
        self._register(
            "ldfm",
            block.LDFM,
        )
        self._register(
            "egpcm",
            block.EGPCM,
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
                "Failed to capture CFMNet "
                "Stage2 branches: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        outputs = self._outputs
        self._outputs = {}

        return outputs


class Stage2CooperativeAdapter(
    nn.Module
):
    """
    Keep the successful Stage2-CFM path unchanged:

        TCRM + EGPCM -> semantic
        SSRM + LDFM  -> detail

        Stage3 decoder feature -> semantic guidance
        semantic -> spatial gate -> detail
        semantic + guided detail -> cooperative residual
    """

    def __init__(
        self,
        branch_channels: int = 48,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        pair_channels = (
            branch_channels
            * 2
        )

        self.semantic_proj = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.detail_proj = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.high_proj = (
            ConvBNReLU(
                decode_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.semantic_fusion = (
            ConvBNReLU(
                decode_channels * 2,
                decode_channels,
                kernel_size=3,
            )
        )

        self.saliency_gate = (
            nn.Sequential(
                nn.Conv2d(
                    decode_channels,
                    1,
                    kernel_size=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
        )

        self.cooperative_fusion = (
            ConvBNReLU(
                decode_channels * 2,
                decode_channels,
                kernel_size=3,
            )
        )

    def forward(
        self,
        high_feature: torch.Tensor,
        branch_features: dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        semantic = torch.cat(
            [
                branch_features[
                    "tcrm"
                ],
                branch_features[
                    "egpcm"
                ],
            ],
            dim=1,
        )

        detail = torch.cat(
            [
                branch_features[
                    "ssrm"
                ],
                branch_features[
                    "ldfm"
                ],
            ],
            dim=1,
        )

        semantic = (
            self.semantic_proj(
                semantic
            )
        )

        detail = (
            self.detail_proj(
                detail
            )
        )

        high_feature = (
            F.interpolate(
                high_feature,
                size=semantic.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        high_feature = (
            self.high_proj(
                high_feature
            )
        )

        semantic = (
            self.semantic_fusion(
                torch.cat(
                    [
                        semantic,
                        high_feature,
                    ],
                    dim=1,
                )
            )
        )

        gate = (
            self.saliency_gate(
                semantic
            )
        )

        guided_detail = (
            detail
            * gate
        )

        return (
            self.cooperative_fusion(
                torch.cat(
                    [
                        semantic,
                        guided_detail,
                    ],
                    dim=1,
                )
            )
        )


class DynamicDepthwiseKernelGenerator(
    nn.Module
):
    """
    Generate one sample-specific 3x3 depthwise kernel per decoder channel.

    Input:
        Stage2 functional feature [B, C_in, H2, W2]

    Output:
        kernel [B, C_out, 3, 3]

    The kernel is signed and L1-normalized spatially so it can model
    enhancement or suppression without unconstrained magnitude.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.out_channels = (
            out_channels
        )

        self.kernel_size = (
            kernel_size
        )

        self.context = (
            ConvBNReLU(
                in_channels,
                hidden_channels,
                kernel_size=3,
            )
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(
                1
            )
        )

        self.generator = (
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.ReLU6(
                    inplace=True
                ),
                nn.Conv2d(
                    hidden_channels,
                    (
                        out_channels
                        * kernel_size
                        * kernel_size
                    ),
                    kernel_size=1,
                    bias=True,
                ),
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        kernel = (
            self.context(
                x
            )
        )

        kernel = (
            self.pool(
                kernel
            )
        )

        kernel = (
            self.generator(
                kernel
            )
        )

        batch_size = (
            kernel.shape[0]
        )

        kernel = kernel.view(
            batch_size,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
        )

        kernel = torch.tanh(
            kernel
        )

        denominator = (
            kernel
            .abs()
            .sum(
                dim=(-1, -2),
                keepdim=True,
            )
            + 1e-6
        )

        return (
            kernel
            / denominator
        )


class SampleWiseDepthwiseFilter(
    nn.Module
):
    """
    Apply sample-specific depthwise kernels.

    x:
        [B, C, H, W]

    kernels:
        [B, C, K, K]

    Internally reshape the batch/channel dimensions so PyTorch grouped
    convolution performs B*C independent depthwise convolutions.
    """

    def __init__(
        self,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.kernel_size = (
            kernel_size
        )

        self.padding = (
            kernel_size
            // 2
        )

    def forward(
        self,
        x: torch.Tensor,
        kernels: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            channels,
            height,
            width,
        ) = x.shape

        if (
            kernels.shape[0]
            != batch_size
            or kernels.shape[1]
            != channels
        ):
            raise ValueError(
                "Dynamic kernel shape does not "
                "match target feature shape."
            )

        x_grouped = x.reshape(
            1,
            batch_size * channels,
            height,
            width,
        )

        weight = kernels.reshape(
            batch_size * channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )

        filtered = F.conv2d(
            x_grouped,
            weight,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=(
                batch_size
                * channels
            ),
        )

        return filtered.reshape(
            batch_size,
            channels,
            height,
            width,
        )


class CFMNetStage2KernelDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Stage2-centered kernel-guided UNetFormer.

    UNetFormer's backbone-decoder structure is kept intact.

    Stage2 has three roles:

    1. Local cooperative residual:
        Stage2 T/S/L/G -> CFM residual -> decoder Stage2.

    2. Semantic kernel:
        TCRM2 + EGPCM2 -> K_sem
        K_sem dynamically filters projected Stage3 skip.

    3. Structure kernel:
        SSRM2 + LDFM2 -> K_str
        K_str dynamically filters projected Stage1 skip.

    Stage4 is untouched.

    All three added residual scales start from zero:
        gamma3 : semantic kernel -> Stage3
        alpha2 : Stage2 local cooperative residual
        gamma1 : structure kernel -> Stage1
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

        if (
            stage2_channels
            % 4
            != 0
        ):
            raise ValueError(
                "CFMNet Stage2 channels "
                "must be divisible by four."
            )

        branch_channels = (
            stage2_channels
            // 4
        )

        pair_channels = (
            branch_channels
            * 2
        )

        self.stage2_adapter = (
            Stage2CooperativeAdapter(
                branch_channels=(
                    branch_channels
                ),
                decode_channels=(
                    decode_channels
                ),
            )
        )

        self.semantic_kernel_generator = (
            DynamicDepthwiseKernelGenerator(
                in_channels=(
                    pair_channels
                ),
                out_channels=(
                    decode_channels
                ),
                hidden_channels=(
                    decode_channels
                ),
                kernel_size=3,
            )
        )

        self.structure_kernel_generator = (
            DynamicDepthwiseKernelGenerator(
                in_channels=(
                    pair_channels
                ),
                out_channels=(
                    decode_channels
                ),
                hidden_channels=(
                    decode_channels
                ),
                kernel_size=3,
            )
        )

        self.dynamic_filter = (
            SampleWiseDepthwiseFilter(
                kernel_size=3
            )
        )

        self.stage3_kernel_scale = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

        self.stage2_cfm_scale = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

        self.stage1_kernel_scale = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

    @staticmethod
    def _normalized_weights(
        weights: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        weights = F.relu(
            weights
        )

        return (
            weights
            / (
                weights.sum()
                + epsilon
            )
        )

    def _fuse3_with_projected_skip(
        self,
        x: torch.Tensor,
        projected_stage3: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=projected_stage3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        weights = (
            self._normalized_weights(
                self.fuse3.weights,
                self.fuse3.epsilon,
            )
        )

        x = (
            weights[0]
            * projected_stage3
            + weights[1]
            * x
        )

        return (
            self.fuse3.post_conv(
                x
            )
        )

    def _refine1_with_projected_skip(
        self,
        x: torch.Tensor,
        projected_stage1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same computation as FeatureRefinementHead.forward(),
        except its pre_conv(stage1) is done before dynamic filtering.
        """

        x = F.interpolate(
            x,
            size=projected_stage1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        weights = (
            self._normalized_weights(
                self.refine1.weights,
                self.refine1.epsilon,
            )
        )

        x = (
            weights[0]
            * projected_stage1
            + weights[1]
            * x
        )

        x = (
            self.refine1.post_conv(
                x
            )
        )

        shortcut = (
            self.refine1.shortcut(
                x
            )
        )

        pixel = (
            self.refine1.pixel_attention(
                x
            )
            * x
        )

        channel = (
            self.refine1.channel_attention(
                x
            )
            * x
        )

        x = (
            pixel
            + channel
        )

        x = (
            self.refine1.proj(
                x
            )
            + shortcut
        )

        return (
            self.refine1.act(
                x
            )
        )

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

        semantic_source = torch.cat(
            [
                stage2_branches[
                    "tcrm"
                ],
                stage2_branches[
                    "egpcm"
                ],
            ],
            dim=1,
        )

        structure_source = torch.cat(
            [
                stage2_branches[
                    "ssrm"
                ],
                stage2_branches[
                    "ldfm"
                ],
            ],
            dim=1,
        )

        semantic_kernel = (
            self.semantic_kernel_generator(
                semantic_source
            )
        )

        structure_kernel = (
            self.structure_kernel_generator(
                structure_source
            )
        )

        # ------------------------------
        # Stage 4: original UNetFormer
        # ------------------------------
        x4 = (
            self.block4(
                self.pre_conv(
                    stage4
                )
            )
        )

        # ------------------------------
        # Stage 3:
        # TCRM2 + EGPCM2 -> semantic kernel
        # ------------------------------
        stage3_projected = (
            self.fuse3.pre_conv(
                stage3
            )
        )

        stage3_dynamic = (
            self.dynamic_filter(
                stage3_projected,
                semantic_kernel,
            )
        )

        stage3_projected = (
            stage3_projected
            + self.stage3_kernel_scale
            * stage3_dynamic
        )

        x3 = (
            self.block3(
                self._fuse3_with_projected_skip(
                    x4,
                    stage3_projected,
                )
            )
        )

        # ------------------------------
        # Stage 2:
        # preserve the successful local CFM path
        # ------------------------------
        x2_base = (
            self.fuse2(
                x3,
                stage2,
            )
        )

        stage2_cfm = (
            self.stage2_adapter(
                high_feature=x3,
                branch_features=(
                    stage2_branches
                ),
            )
        )

        x2 = (
            x2_base
            + self.stage2_cfm_scale
            * stage2_cfm
        )

        x2 = (
            self.block2(
                x2
            )
        )

        # ------------------------------
        # Stage 1:
        # SSRM2 + LDFM2 -> structure kernel
        # ------------------------------
        stage1_projected = (
            self.refine1.pre_conv(
                stage1
            )
        )

        stage1_dynamic = (
            self.dynamic_filter(
                stage1_projected,
                structure_kernel,
            )
        )

        stage1_projected = (
            stage1_projected
            + self.stage1_kernel_scale
            * stage1_dynamic
        )

        x1 = (
            self._refine1_with_projected_skip(
                x2,
                stage1_projected,
            )
        )

        prediction = (
            self.segmentation_head(
                x1
            )
        )

        prediction = (
            F.interpolate(
                prediction,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        )

        auxiliary = None

        if self.training:
            aux_size = (
                x2.shape[-2:]
            )

            aux4 = (
                F.interpolate(
                    x4,
                    size=aux_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

            aux3 = (
                F.interpolate(
                    x3,
                    size=aux_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

            auxiliary = (
                self.aux_head(
                    (
                        aux4
                        + aux3
                        + x2
                    ),
                    output_size=(
                        output_size
                    ),
                )
            )

        return (
            prediction,
            auxiliary,
        )

    def modulation_scales(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        return {
            "stage3_semantic_kernel": (
                self.stage3_kernel_scale
            ),
            "stage2_local_cfm": (
                self.stage2_cfm_scale
            ),
            "stage1_structure_kernel": (
                self.stage1_kernel_scale
            ),
        }


class CFMNetUNetFormerStage2KernelSOD(
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
            CFMNetStage2KernelDecoder(
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

        features = (
            self.backbone(
                image
            )
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
            outputs[
                "aux"
            ] = [
                auxiliary
            ]

        return outputs


def build_model(
) -> CFMNetUNetFormerStage2KernelSOD:
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
        CFMNetUNetFormerStage2KernelSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )


if __name__ == "__main__":
    model = build_model()

    image = torch.randn(
        2,
        3,
        352,
        352,
    )

    model.train()

    outputs = model(
        image=image
    )

    print(
        "train pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )

    print(
        "train aux:",
        [
            tuple(
                tensor.shape
            )
            for tensor
            in outputs.get(
                "aux",
                []
            )
        ],
    )

    print(
        "scales:",
        {
            name: float(
                value
                .detach()
                .cpu()
            )
            for (
                name,
                value,
            ) in (
                model
                .decoder
                .modulation_scales()
                .items()
            )
        },
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            image=image
        )

    print(
        "eval pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )
