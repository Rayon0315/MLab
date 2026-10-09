from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.asymmetric_foreground_support_v2 import (
    AsymmetricForegroundSupportV2,
)
from models.networks.vmamba_small_mambacvc_sod import (
    PRETRAINED_PATH,
    VMambaSmallMambaCVCSOD,
)


class VMambaSmallMambaCVCAsymFGV2SOD(VMambaSmallMambaCVCSOD):
    """MambaCVC + Asymmetric Foreground Support v2, without NAMLab.

    The original MambaCVC GC-VCM -> LC-VCM path is kept unchanged. FG-v2 is
    attached only at the decoder tail:

        high_feature = fd3
        low_feature  = fd1
        guide_logits = head3(fd3)

    This is the no-NAM ablation paired with the existing
    MambaCVC + NAM LocalCross + FG-v2 model.
    """

    input_keys = ("image",)

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
        )

        self.foreground_support = AsymmetricForegroundSupportV2(
            channels=channels,
            metric_channels=foreground_metric_channels,
            initial_support_scale=foreground_support_scale,
            initial_residual_scale=foreground_residual_scale,
            eps=1e-6,
            detach_guide=detach_foreground_guide,
        )

    @staticmethod
    def _upsample_to_input(
        tensor: torch.Tensor,
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(
            tensor,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        input_size = image.shape[-2:]
        stage1, stage2, stage3, stage4 = self.backbone(image)

        fe1 = self.normalize1(stage1)
        fe2 = self.normalize2(stage2)
        fe3 = self.normalize3(stage3)
        fe4 = self.normalize4(stage4)

        # Original MambaCVC global compensation path.
        fn3 = self.gc3(fe3, fe4)
        fn2 = self.gc2(fe2, fn3)
        fn1 = self.gc1(fe1, fn2)

        # Original MambaCVC local compensation decoder path. No NAM guidance.
        fd4 = self.dec4(fe4)
        lc3 = self.lc3(fn3, fd4)
        fd3 = self.dec3(lc3)
        lc2 = self.lc2(fn2, fd3)
        fd2 = self.dec2(lc2)
        lc1 = self.lc1(fn1, fd2)
        fd1 = self.dec1(lc1)

        # Keep MambaCVC auxiliary supervision unchanged.
        s2 = self._predict(fd2, self.head2, fe1.shape[-2:])
        s3 = self._predict(fd3, self.head3, fe2.shape[-2:])
        s4 = self._predict(fd4, self.head4, fe3.shape[-2:])

        # FG-v2 semantic support at the decoder tail.
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


def build_model() -> VMambaSmallMambaCVCAsymFGV2SOD:
    return VMambaSmallMambaCVCAsymFGV2SOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
        foreground_metric_channels=16,
        foreground_support_scale=4.0,
        foreground_residual_scale=0.0,
        detach_foreground_guide=True,
    )
