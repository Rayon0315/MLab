from __future__ import annotations

from pathlib import Path
import warnings

from models.backbones.cfmnet import CFMNET_OUT_CHANNELS
from models.components.cfm_structure_refinement import (
    BoundaryRestrictedRefinement,
    Stage1StructureTap,
)
from models.networks.cfmnet_global_local_branch_unetformer_sod import (
    PRETRAINED_PATH,
    CFMNetGlobalLocalBranchUNetFormerSOD,
)


class CFMNetGlobalLocalBoundaryRefineSOD(
    CFMNetGlobalLocalBranchUNetFormerSOD
):
    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decode_channels: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=decode_channels,
            window_size=window_size,
        )
        branch_channels = CFMNET_OUT_CHANNELS[0] // 4
        self.stage1_structure_tap = Stage1StructureTap(self.backbone)
        self.boundary_refinement = BoundaryRestrictedRefinement(
            branch_channels=branch_channels,
            hidden_channels=64,
            boundary_kernel=5,
        )

    def forward(self, image):
        self.stage1_structure_tap.clear()
        outputs = super().forward(image=image)
        branches = self.stage1_structure_tap.pop()
        refined, _ = self.boundary_refinement(
            logits=outputs["pred"],
            branches=branches,
        )
        outputs["pred"] = refined
        return outputs


def build_model():
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )
    if pretrained_path is None:
        warnings.warn(
            f"CFMNet checkpoint not found at {PRETRAINED_PATH}.",
            RuntimeWarning,
        )
    return CFMNetGlobalLocalBoundaryRefineSOD(
        pretrained_path=pretrained_path,
        decode_channels=64,
        window_size=8,
    )
