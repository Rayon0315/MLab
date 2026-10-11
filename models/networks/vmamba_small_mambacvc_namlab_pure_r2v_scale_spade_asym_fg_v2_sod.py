from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from models.components.mambacvc_namlab_pure_r2v_scale_spade import (
    RegionMeanToVisualScaleSPADEGuidance,
)
from models.networks.vmamba_small_mambacvc_asym_fg_v2_sod import (
    PRETRAINED_PATH,
)
from models.networks.vmamba_small_mambacvc_namlab_pure_r2v_local_asym_fg_v2_sod import (
    VMambaSmallMambaCVCNAMLabPureR2VLocalAsymFGV2SOD,
)


class VMambaSmallMambaCVCNAMLabPureR2VScaleSPADEAsymFGV2SOD(
    VMambaSmallMambaCVCNAMLabPureR2VLocalAsymFGV2SOD
):
    """Pure R->V + scale-only SPADE-style NAM modulation + FG-v2.

    Relative to the validated Pure R->V model, only the guidance block changes:
    encoded R predicts a bounded spatial/channel scale for the selected R->V
    evidence. There is no raw-R reconstruction input and no beta branch.
    """

    def __init__(
        self,
        pretrained_path: str | Path | None,
        *,
        backbone: nn.Module | None = None,
        channels: int = 32,
        guidance_inner_channels: int = 16,
        guidance_heads: int = 4,
        guidance_window: int = 3,
        foreground_metric_channels: int = 16,
        foreground_support_scale: float = 4.0,
        foreground_residual_scale: float = 0.0,
        detach_foreground_guide: bool = True,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            backbone=backbone,
            channels=channels,
            guidance_inner_channels=guidance_inner_channels,
            guidance_heads=guidance_heads,
            guidance_window=guidance_window,
            foreground_metric_channels=foreground_metric_channels,
            foreground_support_scale=foreground_support_scale,
            foreground_residual_scale=foreground_residual_scale,
            detach_foreground_guide=detach_foreground_guide,
        )

        guidance_kwargs = dict(
            channels=channels,
            inner_channels=guidance_inner_channels,
            num_heads=guidance_heads,
            window_size=guidance_window,
        )
        self.guidance3 = RegionMeanToVisualScaleSPADEGuidance(
            **guidance_kwargs
        )
        self.guidance2 = RegionMeanToVisualScaleSPADEGuidance(
            **guidance_kwargs
        )
        self.guidance1 = RegionMeanToVisualScaleSPADEGuidance(
            **guidance_kwargs
        )


def build_model() -> VMambaSmallMambaCVCNAMLabPureR2VScaleSPADEAsymFGV2SOD:
    return VMambaSmallMambaCVCNAMLabPureR2VScaleSPADEAsymFGV2SOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
        guidance_inner_channels=16,
        guidance_heads=4,
        guidance_window=3,
        foreground_metric_channels=16,
        foreground_support_scale=4.0,
        foreground_residual_scale=0.0,
        detach_foreground_guide=True,
    )
