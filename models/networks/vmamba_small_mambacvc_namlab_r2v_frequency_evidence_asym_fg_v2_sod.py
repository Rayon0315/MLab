from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from models.components.mambacvc_namlab_r2v_frequency_evidence import (
    RegionMeanToVisualFrequencyEvidenceGuidance,
)
from models.networks.vmamba_small_mambacvc_asym_fg_v2_sod import (
    PRETRAINED_PATH,
)
from models.networks.vmamba_small_mambacvc_namlab_r2v_local_asym_fg_v2_sod import (
    VMambaSmallMambaCVCNAMLabR2VLocalAsymFGV2SOD,
)


class VMambaSmallMambaCVCNAMLabR2VFrequencyEvidenceAsymFGV2SOD(
    VMambaSmallMambaCVCNAMLabR2VLocalAsymFGV2SOD
):
    """Validated R->V NAM fusion with frequency-evidence source weighting.

    The parent network forward path is unchanged. Only guidance1/2/3 are
    replaced. Each guidance block keeps the validated [F, R, R->F] sources,
    then uses fixed low/high-frequency evidence to produce three spatial
    softmax reliability weights before the existing reconstruction.
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
        frequency_hidden_channels: int = 16,
        low_frequency_radius: float = 0.25,
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
            frequency_hidden_channels=frequency_hidden_channels,
            low_frequency_radius=low_frequency_radius,
        )
        self.guidance3 = RegionMeanToVisualFrequencyEvidenceGuidance(
            **guidance_kwargs
        )
        self.guidance2 = RegionMeanToVisualFrequencyEvidenceGuidance(
            **guidance_kwargs
        )
        self.guidance1 = RegionMeanToVisualFrequencyEvidenceGuidance(
            **guidance_kwargs
        )


def build_model() -> VMambaSmallMambaCVCNAMLabR2VFrequencyEvidenceAsymFGV2SOD:
    return VMambaSmallMambaCVCNAMLabR2VFrequencyEvidenceAsymFGV2SOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
        guidance_inner_channels=16,
        guidance_heads=4,
        guidance_window=3,
        frequency_hidden_channels=16,
        low_frequency_radius=0.25,
        foreground_metric_channels=16,
        foreground_support_scale=4.0,
        foreground_residual_scale=0.0,
        detach_foreground_guide=True,
    )
