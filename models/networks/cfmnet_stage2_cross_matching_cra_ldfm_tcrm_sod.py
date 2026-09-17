# models/networks/cfmnet_stage2_cross_matching_cra_ldfm_tcrm_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch

from models.backbones.cfmnet import CFMNET_OUT_CHANNELS
from models.components.cfm_stage2_region_adapter import CFMStage2RegionAdapter
from models.networks.cfmnet_stage2_cross_matching_unetformer_sod import (
    PRETRAINED_PATH,
    CFMNetStage2CrossMatchingUNetFormerSOD,
)


class CFMNetStage2CrossMatchingCRALDFMTCRMSOD(CFMNetStage2CrossMatchingUNetFormerSOD):
    """
    CRA Step 2: Step 1 + M60 statistics calibrate Stage2 TCRM.

    The fused CFMNet feature pyramid and UNetFormer decoder are unchanged.
    Only captured Stage2 CFM branch outputs are region-conditioned before
    they enter the existing local Q/K/V cross matching module.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decode_channels: int = 64,
        window_size: int = 8,
        matching_window: int = 7,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=decode_channels,
            window_size=window_size,
            matching_window=matching_window,
        )

        stage2_branch_channels = CFMNET_OUT_CHANNELS[1] // 4

        self.region_adapter = CFMStage2RegionAdapter(
            branch_channels=stage2_branch_channels,
            enable_ldfm=True,
            enable_tcrm=True,
            enable_egpcm=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        output_size = image.shape[-2:]

        self.stage2_tap.clear()

        features = self.backbone(image)
        stage2_branches = self.stage2_tap.pop()

        adapted_branches = self.region_adapter(
            branches=stage2_branches,
            image=image,
            mean_60=mean_60,
        )

        prediction, auxiliary = self.decoder(
            features=features,
            stage2_branches=adapted_branches,
            output_size=output_size,
        )

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs["aux"] = [auxiliary]

        return outputs


def build_model() -> CFMNetStage2CrossMatchingCRALDFMTCRMSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
        warnings.warn(
            (
                "CFMNet ImageNet-1K checkpoint was not found at "
                f"{PRETRAINED_PATH}. The backbone will train from scratch."
            ),
            RuntimeWarning,
        )

    return CFMNetStage2CrossMatchingCRALDFMTCRMSOD(
        pretrained_path=pretrained_path,
        decode_channels=64,
        window_size=8,
        matching_window=7,
    )
