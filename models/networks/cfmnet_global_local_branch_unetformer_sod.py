# models/networks/cfmnet_global_local_branch_unetformer_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
    ConvBNReLU,
    GlobalLocalAttention,
    GlobalLocalBlock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class Stage234BranchTap:
    """
    Capture the four CFM branches from the last CFMBlock of
    paper Stage 2/3/4.

    Stage 1 is not used here because the UNetFormer Stage-1 path
    is FeatureRefinementHead rather than GlobalLocalBlock.
    """

    STAGES = {
        "stage2": 2,
        "stage3": 4,
        "stage4": 6,
    }

    BRANCHES = {
        "tcrm": "TCRM",
        "ssrm": "SSRM",
        "ldfm": "LDFM",
        "egpcm": "EGPCM",
    }

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        self._outputs: dict[
            str,
            dict[
                str,
                torch.Tensor,
            ],
        ] = {
            stage_name: {}
            for stage_name
            in self.STAGES
        }

        self._handles = []

        for (
            stage_name,
            stage_index,
        ) in self.STAGES.items():
            block = (
                backbone
                .stages[
                    stage_index
                ]
                .blocks[-1]
            )

            for (
                branch_name,
                module_name,
            ) in self.BRANCHES.items():
                self._register(
                    stage_name=stage_name,
                    branch_name=branch_name,
                    module=getattr(
                        block,
                        module_name,
                    ),
                )

    def _register(
        self,
        stage_name: str,
        branch_name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[
                stage_name
            ][
                branch_name
            ] = output

        self._handles.append(
            module.register_forward_hook(
                hook
            )
        )

    def clear(
        self,
    ) -> None:
        for stage_name in self._outputs:
            self._outputs[
                stage_name
            ].clear()

    def pop(
        self,
    ) -> dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ]:
        expected = set(
            self.BRANCHES
        )

        for stage_name in self.STAGES:
            missing = (
                expected
                - self._outputs[
                    stage_name
                ].keys()
            )

            if missing:
                raise RuntimeError(
                    f"Failed to capture {stage_name}: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

        outputs = self._outputs

        self._outputs = {
            stage_name: {}
            for stage_name
            in self.STAGES
        }

        return outputs


class CFMRoleProjector(
    nn.Module
):
    """
    TCRM + EGPCM -> semantic/global prior
    SSRM + LDFM  -> structure/local prior
    """

    def __init__(
        self,
        stage_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        if stage_channels % 4 != 0:
            raise ValueError(
                "CFMNet stage channels must "
                "be divisible by four."
            )

        pair_channels = (
            stage_channels
            // 2
        )

        self.semantic = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.local = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

    def forward(
        self,
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

        local = torch.cat(
            [
                branches["ssrm"],
                branches["ldfm"],
            ],
            dim=1,
        )

        return (
            self.semantic(
                semantic
            ),
            self.local(
                local
            ),
        )


class BranchConditionedGlobalLocalAttention(
    GlobalLocalAttention
):
    """
    Preserve UNetFormer GlobalLocalAttention, but condition its
    two internal paths differently.

    semantic/global CFM prior:
        injected only before Q/K/V and strip-pooling global path

    structure/detail CFM prior:
        injected only before the local convolution path

    Both signed scales start at zero, so the initial behavior is
    exactly the original UNetFormer attention.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
            relative_pos_embedding=(
                relative_pos_embedding
            ),
        )

        self.global_scale = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

        self.local_scale = (
            nn.Parameter(
                torch.zeros(1)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        semantic_prior: torch.Tensor,
        local_prior: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            channels,
            height,
            width,
        ) = x.shape

        semantic_prior = F.interpolate(
            semantic_prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        local_prior = F.interpolate(
            local_prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        global_input = (
            x
            + self.global_scale
            * semantic_prior
        )

        local_input = (
            x
            + self.local_scale
            * local_prior
        )

        local = (
            self.local2(
                local_input
            )
            + self.local1(
                local_input
            )
        )

        padded = self._pad_to_window(
            global_input,
            self.ws,
        )

        (
            _,
            _,
            padded_height,
            padded_width,
        ) = padded.shape

        qkv = self.qkv(
            padded
        )

        q, k, v = rearrange(
            qkv,
            (
                "b (qkv head d) "
                "(hh ws1) (ww ws2) -> "
                "qkv (b hh ww) head "
                "(ws1 ws2) d"
            ),
            qkv=3,
            head=self.num_heads,
            d=(
                channels
                // self.num_heads
            ),
            hh=(
                padded_height
                // self.ws
            ),
            ww=(
                padded_width
                // self.ws
            ),
            ws1=self.ws,
            ws2=self.ws,
        )

        attention = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        ) * self.scale

        if self.relative_pos_embedding:
            relative_bias = (
                self.relative_position_bias_table[
                    self.relative_position_index
                    .view(-1)
                ]
            )

            relative_bias = (
                relative_bias.view(
                    self.ws
                    * self.ws,
                    self.ws
                    * self.ws,
                    -1,
                )
            )

            relative_bias = (
                relative_bias
                .permute(
                    2,
                    0,
                    1,
                )
                .contiguous()
            )

            attention = (
                attention
                + relative_bias.unsqueeze(
                    0
                )
            )

        attention = (
            attention.softmax(
                dim=-1
            )
        )

        attention = (
            attention
            @ v
        )

        attention = rearrange(
            attention,
            (
                "(b hh ww) head "
                "(ws1 ws2) d -> "
                "b (head d) "
                "(hh ws1) (ww ws2)"
            ),
            b=batch_size,
            head=self.num_heads,
            d=(
                channels
                // self.num_heads
            ),
            hh=(
                padded_height
                // self.ws
            ),
            ww=(
                padded_width
                // self.ws
            ),
            ws1=self.ws,
            ws2=self.ws,
        )

        attention = attention[
            :,
            :,
            :height,
            :width,
        ]

        attention_x = self.attn_x(
            F.pad(
                attention,
                (
                    0,
                    0,
                    0,
                    1,
                ),
                mode="reflect",
            )
        )

        attention_y = self.attn_y(
            F.pad(
                attention,
                (
                    0,
                    1,
                    0,
                    0,
                ),
                mode="reflect",
            )
        )

        out = (
            attention_x
            + attention_y
            + local
        )

        out = self._pad_out(
            out
        )

        out = self.proj(
            out
        )

        return out[
            :,
            :,
            :height,
            :width,
        ]


class BranchConditionedGlobalLocalBlock(
    GlobalLocalBlock
):
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.ReLU6,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        window_size: int = 8,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=window_size,
        )

        self.attn = (
            BranchConditionedGlobalLocalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                window_size=window_size,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        semantic_prior: torch.Tensor,
        local_prior: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            x
            + self.drop_path(
                self.attn(
                    self.norm1(x),
                    semantic_prior=(
                        semantic_prior
                    ),
                    local_prior=(
                        local_prior
                    ),
                )
            )
        )

        x = (
            x
            + self.drop_path(
                self.mlp(
                    self.norm2(x)
                )
            )
        )

        return x


class CFMNetGlobalLocalBranchDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Experiment 2:
    make CFMNet branch roles enter the matching UNetFormer path.

    Stage 4/3/2:
        TCRM + EGPCM -> Global path
        SSRM + LDFM  -> Local path

    Cross-stage fusion and Stage-1 FeatureRefinementHead stay unchanged.
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

        self.role2 = (
            CFMRoleProjector(
                encoder_channels[1],
                decode_channels,
            )
        )

        self.role3 = (
            CFMRoleProjector(
                encoder_channels[2],
                decode_channels,
            )
        )

        self.role4 = (
            CFMRoleProjector(
                encoder_channels[3],
                decode_channels,
            )
        )

        self.block4 = (
            BranchConditionedGlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

        self.block3 = (
            BranchConditionedGlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

        self.block2 = (
            BranchConditionedGlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ],
        branch_features: dict[
            str,
            dict[
                str,
                torch.Tensor,
            ],
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

        (
            semantic4,
            local4,
        ) = self.role4(
            branch_features[
                "stage4"
            ]
        )

        x4 = self.block4(
            self.pre_conv(
                stage4
            ),
            semantic_prior=semantic4,
            local_prior=local4,
        )

        (
            semantic3,
            local3,
        ) = self.role3(
            branch_features[
                "stage3"
            ]
        )

        x3 = self.block3(
            self.fuse3(
                x4,
                stage3,
            ),
            semantic_prior=semantic3,
            local_prior=local3,
        )

        (
            semantic2,
            local2,
        ) = self.role2(
            branch_features[
                "stage2"
            ]
        )

        x2 = self.block2(
            self.fuse2(
                x3,
                stage2,
            ),
            semantic_prior=semantic2,
            local_prior=local2,
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

    def role_scales(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        return {
            "stage4_global": (
                self.block4
                .attn
                .global_scale
            ),
            "stage4_local": (
                self.block4
                .attn
                .local_scale
            ),
            "stage3_global": (
                self.block3
                .attn
                .global_scale
            ),
            "stage3_local": (
                self.block3
                .attn
                .local_scale
            ),
            "stage2_global": (
                self.block2
                .attn
                .global_scale
            ),
            "stage2_local": (
                self.block2
                .attn
                .local_scale
            ),
        }


class CFMNetGlobalLocalBranchUNetFormerSOD(
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

        self.branch_tap = (
            Stage234BranchTap(
                self.backbone
            )
        )

        self.decoder = (
            CFMNetGlobalLocalBranchDecoder(
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

        self.branch_tap.clear()

        features = self.backbone(
            image
        )

        branch_features = (
            self.branch_tap.pop()
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=features,
            branch_features=(
                branch_features
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
) -> CFMNetGlobalLocalBranchUNetFormerSOD:
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
        CFMNetGlobalLocalBranchUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )
