# models/networks/cfmnet_four_role_preserving_unetformer_sod.py

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
from models.components.cfm_role_preserving import (
    FourRoleProjector,
    FourRoleRouter,
    Stage234BranchTap,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
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


class FourRoleGlobalLocalAttention(
    GlobalLocalAttention
):
    """
    UNetFormer GlobalLocalAttention with four role-preserving CFM priors.

    TCRM:
        conditions Q/K only, so it affects semantic matching.

    EGPCM:
        conditions V only, so it contributes contextual content without
        changing where Q/K choose to attend.

    SSRM:
        enters only after window attention and before horizontal/vertical
        strip pooling, matching its directional-structure role.

    LDFM:
        enters only into the local convolution path.

    No CFM branch is concatenated with another branch and no pair grouping
    is imposed. Four zero-initialized scales make the block start exactly
    from the original UNetFormer attention behavior.
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

        self.role_router = FourRoleRouter()

    @staticmethod
    def _resize_prior(
        prior: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if prior.shape[-2:] == (
            height,
            width,
        ):
            return prior

        return F.interpolate(
            prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def _window_tokens(
        self,
        feature: torch.Tensor,
        padded_height: int,
        padded_width: int,
        channels: int,
    ) -> torch.Tensor:
        return rearrange(
            feature,
            (
                "b (head d) "
                "(hh ws1) (ww ws2) -> "
                "(b hh ww) head "
                "(ws1 ws2) d"
            ),
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

    def forward(
        self,
        x: torch.Tensor,
        tcrm_prior: torch.Tensor,
        ssrm_prior: torch.Tensor,
        ldfm_prior: torch.Tensor,
        egpcm_prior: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            channels,
            height,
            width,
        ) = x.shape

        tcrm_prior = self._resize_prior(
            tcrm_prior,
            height,
            width,
        )
        ssrm_prior = self._resize_prior(
            ssrm_prior,
            height,
            width,
        )
        ldfm_prior = self._resize_prior(
            ldfm_prior,
            height,
            width,
        )
        egpcm_prior = self._resize_prior(
            egpcm_prior,
            height,
            width,
        )

        routed = self.role_router(
            x=x,
            tcrm=tcrm_prior,
            ssrm=ssrm_prior,
            ldfm=ldfm_prior,
            egpcm=egpcm_prior,
        )

        local = (
            self.local2(
                routed["local_input"]
            )
            + self.local1(
                routed["local_input"]
            )
        )

        qk_padded = self._pad_to_window(
            routed["qk_input"],
            self.ws,
        )

        value_padded = self._pad_to_window(
            routed["value_input"],
            self.ws,
        )

        (
            _,
            _,
            padded_height,
            padded_width,
        ) = qk_padded.shape

        qkv_qk = self.qkv(
            qk_padded
        )
        q_map, k_map, _ = qkv_qk.chunk(
            3,
            dim=1,
        )

        qkv_value = self.qkv(
            value_padded
        )
        _, _, v_map = qkv_value.chunk(
            3,
            dim=1,
        )

        q = self._window_tokens(
            q_map,
            padded_height,
            padded_width,
            channels,
        )
        k = self._window_tokens(
            k_map,
            padded_height,
            padded_width,
            channels,
        )
        v = self._window_tokens(
            v_map,
            padded_height,
            padded_width,
            channels,
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

        attention = attention.softmax(
            dim=-1
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

        directional = (
            attention
            + routed[
                "directional_delta"
            ]
        )

        attention_x = self.attn_x(
            F.pad(
                directional,
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
                directional,
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

    def role_scales(
        self,
    ) -> dict[str, torch.Tensor]:
        return self.role_router.scales()


class FourRoleGlobalLocalBlock(
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

        self.attn = FourRoleGlobalLocalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        roles: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = (
            x
            + self.drop_path(
                self.attn(
                    self.norm1(x),
                    tcrm_prior=(
                        roles["tcrm"]
                    ),
                    ssrm_prior=(
                        roles["ssrm"]
                    ),
                    ldfm_prior=(
                        roles["ldfm"]
                    ),
                    egpcm_prior=(
                        roles["egpcm"]
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


class CFMNetFourRolePreservingDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Preserve all four CFM branch identities at Stage 2/3/4.

    Base top-down fusion still uses the standard fused CFMNet stage
    features. The four raw CFM branches are additional role-specific
    conditions inside the matching UNetFormer block:

        TCRM  -> Q/K semantic matching
        SSRM  -> directional strip-pooling path
        LDFM  -> local convolution path
        EGPCM -> attention value/context path

    Stage 1 FeatureRefinementHead and all cross-stage fusion remain
    unchanged from CFMNetUNetFormerDecoder.
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
            window_size=window_size,
            num_classes=num_classes,
        )

        self.role2 = FourRoleProjector(
            encoder_channels[1],
            decode_channels,
        )
        self.role3 = FourRoleProjector(
            encoder_channels[2],
            decode_channels,
        )
        self.role4 = FourRoleProjector(
            encoder_channels[3],
            decode_channels,
        )

        self.block4 = FourRoleGlobalLocalBlock(
            dim=decode_channels,
            num_heads=8,
            window_size=window_size,
        )
        self.block3 = FourRoleGlobalLocalBlock(
            dim=decode_channels,
            num_heads=8,
            window_size=window_size,
        )
        self.block2 = FourRoleGlobalLocalBlock(
            dim=decode_channels,
            num_heads=8,
            window_size=window_size,
        )

    def forward(
        self,
        features: list[torch.Tensor],
        branch_features: dict[
            str,
            dict[str, torch.Tensor],
        ],
        output_size: tuple[int, int],
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

        roles4 = self.role4(
            branch_features["stage4"]
        )
        x4 = self.block4(
            self.pre_conv(stage4),
            roles=roles4,
        )

        roles3 = self.role3(
            branch_features["stage3"]
        )
        x3 = self.block3(
            self.fuse3(
                x4,
                stage3,
            ),
            roles=roles3,
        )

        roles2 = self.role2(
            branch_features["stage2"]
        )
        x2 = self.block2(
            self.fuse2(
                x3,
                stage2,
            ),
            roles=roles2,
        )

        x1 = self.refine1(
            x2,
            stage1,
        )

        prediction = self.segmentation_head(
            x1
        )

        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = None

        if self.training:
            aux_size = x2.shape[-2:]

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

            auxiliary = self.aux_head(
                aux4
                + aux3
                + x2,
                output_size=output_size,
            )

        return (
            prediction,
            auxiliary,
        )

    def role_scales(
        self,
    ) -> dict[str, torch.Tensor]:
        output = {}

        for stage_name, block in (
            ("stage4", self.block4),
            ("stage3", self.block3),
            ("stage2", self.block2),
        ):
            for role_name, scale in (
                block.attn
                .role_scales()
                .items()
            ):
                output[
                    f"{stage_name}_{role_name}"
                ] = scale

        return output


class CFMNetFourRolePreservingUNetFormerSOD(
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

        self.backbone = build_cfmnet(
            pretrained_path=pretrained_path
        )

        self.branch_tap = Stage234BranchTap(
            self.backbone
        )

        self.decoder = (
            CFMNetFourRolePreservingDecoder(
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
        | list[torch.Tensor],
    ]:
        output_size = image.shape[-2:]

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
            branch_features=branch_features,
            output_size=output_size,
        )

        outputs: dict[
            str,
            torch.Tensor
            | list[torch.Tensor],
        ] = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs["aux"] = [
                auxiliary
            ]

        return outputs


def build_model(
) -> CFMNetFourRolePreservingUNetFormerSOD:
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

    return CFMNetFourRolePreservingUNetFormerSOD(
        pretrained_path=pretrained_path,
        decode_channels=64,
        window_size=8,
    )
