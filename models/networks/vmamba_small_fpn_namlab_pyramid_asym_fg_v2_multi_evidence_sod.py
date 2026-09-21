from __future__ import annotations

from pathlib import Path

from models.components.namlab_multi_evidence import (
    NAMLabMultiEvidenceHybrid,
)
from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD,
)


class VMambaSmallFPNNAMLabPyramidAsymFGV2MultiEvidenceSOD(
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD
):
    """
    Asymmetric FG Support v2 with a controlled NAMLab encoder replacement.

    Unchanged:
        - VMamba-S backbone
        - Stage1 NAMLab detail path
        - Stage2 RegionConditionedLocalReconstruction
        - Stage3/4 BidirectionalRegionVisualInteraction
        - FPN
        - PyramidContextBlock
        - AsymmetricForegroundSupportV2
        - outputs / loss interface

    Changed:
        - Stage2/3/4 HybridRegionScaleEncoder
          -> MultiEvidenceRegionScaleEncoder
    """

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
        use_cdc: bool = False,
        use_oad: bool = False,
        initial_cdc_scale: float = 0.05,
        initial_oad_scale: float = 0.05,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decoder_channels=decoder_channels,
        )

        # Current validated NAMLab Hybrid uses context_scale = 0.1.
        # Keep that exact value so the ordinary local-context path is not
        # silently changed when CDC/OAD are added.
        self.namlab_hybrid = (
            NAMLabMultiEvidenceHybrid(
                stage_channels=tuple(
                    self.backbone.out_channels
                ),
                initial_context_scale=0.1,
                use_cdc=use_cdc,
                use_oad=use_oad,
                initial_cdc_scale=initial_cdc_scale,
                initial_oad_scale=initial_oad_scale,
            )
        )
