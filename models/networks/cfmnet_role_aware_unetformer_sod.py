# models/networks/cfmnet_role_aware_unetformer_sod.py

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


class RoleAwareBranchTap:
    """
    Capture only the CFM branches needed by the role-aware decoder.

    Stage 1:
        SSRM + LDFM
        -> local structure/detail recovery

    Stage 2:
        TCRM + EGPCM
        SSRM + LDFM
        -> semantic-structure matching hub

    Stage 3:
        TCRM + EGPCM
        -> semantic correction

    Stage 4:
        TCRM + EGPCM
        -> global channel prior

    CFMNet stage layout:
        stages[0] -> paper Stage 1
        stages[2] -> paper Stage 2
        stages[4] -> paper Stage 3
        stages[6] -> paper Stage 4
    """

    SPEC = {
        "stage1": (
            0,
            (
                "ssrm",
                "ldfm",
            ),
        ),
        "stage2": (
            2,
            (
                "tcrm",
                "ssrm",
                "ldfm",
                "egpcm",
            ),
        ),
        "stage3": (
            4,
            (
                "tcrm",
                "egpcm",
            ),
        ),
        "stage4": (
            6,
            (
                "tcrm",
                "egpcm",
            ),
        ),
    }

    BACKBONE_NAMES = {
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
            in self.SPEC
        }

        self._handles = []

        for (
            stage_name,
            (
                stage_index,
                branch_names,
            ),
        ) in self.SPEC.items():
            stage = (
                backbone
                .stages[
                    stage_index
                ]
            )

            block = (
                stage
                .blocks[-1]
            )

            for branch_name in (
                branch_names
            ):
                module = getattr(
                    block,
                    self.BACKBONE_NAMES[
                        branch_name
                    ],
                )

                self._register(
                    stage_name=(
                        stage_name
                    ),
                    branch_name=(
                        branch_name
                    ),
                    module=module,
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
        for stage_name in (
            self._outputs
        ):
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
        for (
            stage_name,
            (
                _stage_index,
                branch_names,
            ),
        ) in self.SPEC.items():
            missing = (
                set(
                    branch_names
                )
                - self._outputs[
                    stage_name
                ].keys()
            )

            if missing:
                raise RuntimeError(
                    "Failed to capture "
                    f"{stage_name} branches: "
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
            in self.SPEC
        }

        return outputs


class Stage4GlobalSemanticPrior(
    nn.Module
):
    """
    Stage 4 is low-resolution and semantic-rich.

    TCRM4 + EGPCM4 are compressed into a global channel prior
    that conditions the Stage-3 semantic correction branch.
    """

    def __init__(
        self,
        pair_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        self.proj = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        hidden_channels = max(
            decode_channels // 2,
            16,
        )

        self.channel_prior = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(
                    1
                ),
                nn.Conv2d(
                    decode_channels,
                    hidden_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.ReLU6(
                    inplace=True
                ),
                nn.Conv2d(
                    hidden_channels,
                    decode_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
        )

    def forward(
        self,
        tcrm: torch.Tensor,
        egpcm: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.cat(
            [
                tcrm,
                egpcm,
            ],
            dim=1,
        )

        semantic = self.proj(
            semantic
        )

        return self.channel_prior(
            semantic
        )


class Stage3SemanticCorrection(
    nn.Module
):
    """
    Stage 3 provides mature semantic localization.

    Its TCRM + EGPCM feature is corrected by the global
    Stage-4 channel prior before being injected into the
    normal UNetFormer Stage-3 skip.

    The residual strength is signed and zero-initialized.
    """

    def __init__(
        self,
        pair_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        self.semantic_proj = (
            nn.Sequential(
                ConvBNReLU(
                    pair_channels,
                    decode_channels,
                    kernel_size=1,
                ),
                ConvBNReLU(
                    decode_channels,
                    decode_channels,
                    kernel_size=3,
                ),
            )
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        projected_stage3: torch.Tensor,
        tcrm: torch.Tensor,
        egpcm: torch.Tensor,
        global_prior: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.cat(
            [
                tcrm,
                egpcm,
            ],
            dim=1,
        )

        semantic = self.semantic_proj(
            semantic
        )

        semantic = (
            semantic
            * global_prior
        )

        return (
            projected_stage3
            + self.scale
            * semantic
        )


class Stage2SemanticStructureMatching(
    nn.Module
):
    """
    Stage 2 is treated as the semantic-structure negotiation hub.

    Semantic:
        TCRM2 + EGPCM2

    Structure/detail:
        SSRM2 + LDFM2

    Decoder Stage-3 feature:
        semantic query

    The matching map is explicit cosine similarity between
    the decoder query and Stage-2 semantic representation.
    It controls only the new residual branch; the original
    Stage-2 fused feature remains untouched.
    """

    def __init__(
        self,
        branch_pair_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        self.semantic_proj = (
            ConvBNReLU(
                branch_pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.structure_proj = (
            ConvBNReLU(
                branch_pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.query_proj = (
            ConvBNReLU(
                decode_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.residual_fusion = (
            nn.Sequential(
                ConvBNReLU(
                    decode_channels
                    * 3,
                    decode_channels,
                    kernel_size=3,
                ),
                ConvBNReLU(
                    decode_channels,
                    decode_channels,
                    kernel_size=3,
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
                branches[
                    "tcrm"
                ],
                branches[
                    "egpcm"
                ],
            ],
            dim=1,
        )

        structure = torch.cat(
            [
                branches[
                    "ssrm"
                ],
                branches[
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

        structure = (
            self.structure_proj(
                structure
            )
        )

        query = F.interpolate(
            decoder_stage3,
            size=semantic.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        query = (
            self.query_proj(
                query
            )
        )

        query_norm = F.normalize(
            query,
            p=2,
            dim=1,
            eps=1e-6,
        )

        semantic_norm = F.normalize(
            semantic,
            p=2,
            dim=1,
            eps=1e-6,
        )

        cosine = (
            query_norm
            * semantic_norm
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Map cosine similarity from [-1, 1] to [0, 1].
        affinity = (
            cosine
            + 1.0
        ) * 0.5

        matched_structure = (
            structure
            * affinity
        )

        residual = (
            self.residual_fusion(
                torch.cat(
                    [
                        query,
                        semantic,
                        matched_structure,
                    ],
                    dim=1,
                )
            )
        )

        output = (
            stage2_base
            + self.scale
            * residual
        )

        return (
            output,
            affinity,
        )


class Stage1UncertaintyDetailRecovery(
    nn.Module
):
    """
    Stage 1 keeps the highest-resolution local information.

    SSRM1 + LDFM1 provide structure/detail evidence.
    The already-supervised UNetFormer auxiliary saliency
    prediction supplies uncertainty:

        U = 1 - |2 * sigmoid(P2) - 1|

    Only uncertain locations receive the new detail residual.
    """

    def __init__(
        self,
        pair_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        self.detail_proj = (
            nn.Sequential(
                ConvBNReLU(
                    pair_channels,
                    decode_channels,
                    kernel_size=1,
                ),
                ConvBNReLU(
                    decode_channels,
                    decode_channels,
                    kernel_size=3,
                ),
            )
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        projected_stage1: torch.Tensor,
        ssrm: torch.Tensor,
        ldfm: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        detail = torch.cat(
            [
                ssrm,
                ldfm,
            ],
            dim=1,
        )

        detail = self.detail_proj(
            detail
        )

        uncertainty = F.interpolate(
            uncertainty,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        correction = (
            detail
            * uncertainty
        )

        return (
            projected_stage1
            + self.scale
            * correction
        )


class CFMNetRoleAwareUNetFormerDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Role-aware UNetFormer for CFMNet.

    Stage 4:
        TCRM4 + EGPCM4
        -> global semantic channel prior

    Stage 3:
        TCRM3 + EGPCM3
        -> semantic correction under Stage-4 prior

    Stage 2:
        decoder Stage-3 semantic query
        matches TCRM2 + EGPCM2
        and conditionally activates SSRM2 + LDFM2
        -> cooperative residual

    Stage 1:
        SSRM1 + LDFM1
        -> local detail residual
        weighted by uncertainty from the existing
        supervised auxiliary saliency head

    The original UNetFormer top-down backbone is preserved.
    All new residual scales start from zero.
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

        (
            c1,
            c2,
            c3,
            c4,
        ) = encoder_channels

        for channels in (
            c1,
            c2,
            c3,
            c4,
        ):
            if (
                channels
                % 4
                != 0
            ):
                raise ValueError(
                    "CFMNet stage channels "
                    "must be divisible by four."
                )

        self.stage4_prior = (
            Stage4GlobalSemanticPrior(
                pair_channels=(
                    c4 // 2
                ),
                decode_channels=(
                    decode_channels
                ),
            )
        )

        self.stage3_correction = (
            Stage3SemanticCorrection(
                pair_channels=(
                    c3 // 2
                ),
                decode_channels=(
                    decode_channels
                ),
            )
        )

        self.stage2_matching = (
            Stage2SemanticStructureMatching(
                branch_pair_channels=(
                    c2 // 2
                ),
                decode_channels=(
                    decode_channels
                ),
            )
        )

        self.stage1_detail = (
            Stage1UncertaintyDetailRecovery(
                pair_channels=(
                    c1 // 2
                ),
                decode_channels=(
                    decode_channels
                ),
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

    def _weighted_fusion_with_projected_skip(
        self,
        fusion: nn.Module,
        deep: torch.Tensor,
        projected_skip: torch.Tensor,
    ) -> torch.Tensor:
        deep = F.interpolate(
            deep,
            size=projected_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        weights = (
            self._normalized_weights(
                fusion.weights,
                fusion.epsilon,
            )
        )

        fused = (
            weights[0]
            * projected_skip
            + weights[1]
            * deep
        )

        return fusion.post_conv(
            fused
        )

    def _refine_with_projected_skip(
        self,
        deep: torch.Tensor,
        projected_stage1: torch.Tensor,
    ) -> torch.Tensor:
        deep = F.interpolate(
            deep,
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
            * deep
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

    def _native_aux_logits(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same auxiliary head already used by the baseline,
        but return logits at the native Stage-2 decoder size.
        """
        x = (
            self.aux_head.conv(
                x
            )
        )

        x = (
            self.aux_head.drop(
                x
            )
        )

        return (
            self.aux_head.conv_out(
                x
            )
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ],
        role_features: dict[
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
        ) = features

        # --------------------------------------------------
        # Stage 4:
        # Original UNetFormer decoding remains unchanged.
        # Its TCRM + EGPCM branches generate only a
        # global channel prior for Stage 3.
        # --------------------------------------------------
        x4 = (
            self.block4(
                self.pre_conv(
                    stage4
                )
            )
        )

        prior4 = (
            self.stage4_prior(
                role_features[
                    "stage4"
                ][
                    "tcrm"
                ],
                role_features[
                    "stage4"
                ][
                    "egpcm"
                ],
            )
        )

        # --------------------------------------------------
        # Stage 3:
        # semantic correction before the original
        # UNetFormer weighted fusion.
        # --------------------------------------------------
        stage3_projected = (
            self.fuse3.pre_conv(
                stage3
            )
        )

        stage3_projected = (
            self.stage3_correction(
                projected_stage3=(
                    stage3_projected
                ),
                tcrm=(
                    role_features[
                        "stage3"
                    ][
                        "tcrm"
                    ]
                ),
                egpcm=(
                    role_features[
                        "stage3"
                    ][
                        "egpcm"
                    ]
                ),
                global_prior=(
                    prior4
                ),
            )
        )

        x3 = (
            self.block3(
                self._weighted_fusion_with_projected_skip(
                    fusion=(
                        self.fuse3
                    ),
                    deep=x4,
                    projected_skip=(
                        stage3_projected
                    ),
                )
            )
        )

        # --------------------------------------------------
        # Stage 2:
        # first preserve the baseline weighted fusion,
        # then inject a semantic-structure matching residual.
        # --------------------------------------------------
        x2_base = (
            self.fuse2(
                x3,
                stage2,
            )
        )

        (
            x2_matched,
            affinity2,
        ) = (
            self.stage2_matching(
                stage2_base=(
                    x2_base
                ),
                decoder_stage3=(
                    x3
                ),
                branches=(
                    role_features[
                        "stage2"
                    ]
                ),
            )
        )

        x2 = (
            self.block2(
                x2_matched
            )
        )

        # --------------------------------------------------
        # Reuse the baseline supervised auxiliary pathway
        # to estimate Stage-2 uncertainty.
        # --------------------------------------------------
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

        aux_feature = (
            aux4
            + aux3
            + x2
        )

        coarse_logits = (
            self._native_aux_logits(
                aux_feature
            )
        )

        coarse_prob = (
            torch.sigmoid(
                coarse_logits
            )
        )

        uncertainty = (
            1.0
            - torch.abs(
                2.0
                * coarse_prob
                - 1.0
            )
        )

        # --------------------------------------------------
        # Stage 1:
        # only local SSRM/LDFM evidence is introduced,
        # and only at uncertain locations.
        # --------------------------------------------------
        stage1_projected = (
            self.refine1.pre_conv(
                stage1
            )
        )

        stage1_projected = (
            self.stage1_detail(
                projected_stage1=(
                    stage1_projected
                ),
                ssrm=(
                    role_features[
                        "stage1"
                    ][
                        "ssrm"
                    ]
                ),
                ldfm=(
                    role_features[
                        "stage1"
                    ][
                        "ldfm"
                    ]
                ),
                uncertainty=(
                    uncertainty
                ),
            )
        )

        x1 = (
            self._refine_with_projected_skip(
                deep=x2,
                projected_stage1=(
                    stage1_projected
                ),
            )
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
            auxiliary = F.interpolate(
                coarse_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        diagnostics = {
            "stage2_affinity": (
                affinity2
            ),
            "stage1_uncertainty": (
                uncertainty
            ),
        }

        return (
            prediction,
            auxiliary,
            diagnostics,
        )

    def role_scales(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        return {
            "stage3_semantic_correction": (
                self.stage3_correction
                .scale
            ),
            "stage2_matching": (
                self.stage2_matching
                .scale
            ),
            "stage1_detail_recovery": (
                self.stage1_detail
                .scale
            ),
        }


class CFMNetRoleAwareUNetFormerSOD(
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
            RoleAwareBranchTap(
                self.backbone
            )
        )

        self.decoder = (
            CFMNetRoleAwareUNetFormerDecoder(
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

        features = (
            self.backbone(
                image
            )
        )

        role_features = (
            self.branch_tap.pop()
        )

        (
            prediction,
            auxiliary,
            _diagnostics,
        ) = self.decoder(
            features=features,
            role_features=(
                role_features
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
) -> CFMNetRoleAwareUNetFormerSOD:
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
        CFMNetRoleAwareUNetFormerSOD(
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
        "role scales:",
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
                .role_scales()
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
