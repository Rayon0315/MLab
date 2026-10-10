from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.components.mambacvc_namlab_direct_fusion import (
    RegionMeanDirectFusion,
)
from models.networks.vmamba_small_mambacvc_asym_fg_v2_sod import (
    PRETRAINED_PATH,
    VMambaSmallMambaCVCAsymFGV2SOD,
)


class VMambaSmallMambaCVCNAMLabDirectFusionAsymFGV2SOD(
    VMambaSmallMambaCVCAsymFGV2SOD
):
    """MambaCVC + direct NAM feature fusion + FG-v2.

    NAM M60 is encoded with the same RegionMeanPreservingEncoder used in the
    prior LocalCross experiments, but all attention/relation modeling is
    removed. At fn1/fn2/fn3 the visual feature is concatenated with the
    encoded NAM region feature and reconstructed before entering LC-VCM.
    """

    input_keys = ("image", "mean_60")
    mean_hierarchies = (60,)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        *,
        backbone: nn.Module | None = None,
        channels: int = 32,
        foreground_metric_channels: int = 16,
        foreground_support_scale: float = 4.0,
        foreground_residual_scale: float = 0.0,
        detach_foreground_guide: bool = True,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            backbone=backbone,
            channels=channels,
            foreground_metric_channels=foreground_metric_channels,
            foreground_support_scale=foreground_support_scale,
            foreground_residual_scale=foreground_residual_scale,
            detach_foreground_guide=detach_foreground_guide,
        )

        self.guidance3 = RegionMeanDirectFusion(channels=channels)
        self.guidance2 = RegionMeanDirectFusion(channels=channels)
        self.guidance1 = RegionMeanDirectFusion(channels=channels)

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, object]:
        input_size = image.shape[-2:]
        stage1, stage2, stage3, stage4 = self.backbone(image)

        fe1 = self.normalize1(stage1)
        fe2 = self.normalize2(stage2)
        fe3 = self.normalize3(stage3)
        fe4 = self.normalize4(stage4)

        # Original MambaCVC GC path.
        fn3 = self.gc3(fe3, fe4)
        fn2 = self.gc2(fe2, fn3)
        fn1 = self.gc1(fe1, fn2)

        # NAM direct-fusion ablation: encoded M60 is fused with each GC output.
        gn3 = self.guidance3(fn3, mean_60)
        gn2 = self.guidance2(fn2, mean_60)
        gn1 = self.guidance1(fn1, mean_60)

        # Original MambaCVC LC decoder path.
        fd4 = self.dec4(fe4)
        lc3 = self.lc3(gn3, fd4)
        fd3 = self.dec3(lc3)
        lc2 = self.lc2(gn2, fd3)
        fd2 = self.dec2(lc2)
        lc1 = self.lc1(gn1, fd2)
        fd1 = self.dec1(lc1)

        # Auxiliary supervision stays unchanged.
        s2 = self._predict(fd2, self.head2, fe1.shape[-2:])
        s3 = self._predict(fd3, self.head3, fe2.shape[-2:])
        s4 = self._predict(fd4, self.head4, fe3.shape[-2:])

        # Same FG-v2 tail used by the established no-NAM/R->V variants.
        guide3 = self.head3(fd3)
        fg = self.foreground_support(
            high_feature=fd3,
            low_feature=fd1,
            guide_logits=guide3,
        )

        s1 = self._predict(
            fg["refined_feature"],
            self.head1,
            input_size,
        )

        return {
            "pred": s1,
            "fg_similarity": self._upsample_to_input(
                fg["fg_similarity"], input_size
            ),
            "fg_support_logits": self._upsample_to_input(
                fg["fg_support_logits"], input_size
            ),
            "fg_support": self._upsample_to_input(
                fg["fg_support"], input_size
            ),
            "fg_guide_probability": self._upsample_to_input(
                fg["guide_foreground_probability"], input_size
            ),
            "aux": [s2, s3, s4],
        }


def build_model() -> VMambaSmallMambaCVCNAMLabDirectFusionAsymFGV2SOD:
    return VMambaSmallMambaCVCNAMLabDirectFusionAsymFGV2SOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
        foreground_metric_channels=16,
        foreground_support_scale=4.0,
        foreground_residual_scale=0.0,
        detach_foreground_guide=True,
    )
