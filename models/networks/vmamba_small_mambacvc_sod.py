from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.mambacvc_blocks import (
    ConvBNReLU,
    DecoderBlock,
    GCVCM,
    LCVCM,
    SaliencyHead,
)

if TYPE_CHECKING:
    from models.backbones.vmamba import VMambaSmallBackbone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vmamba"
    / "vssm_small_0229_ckpt_epoch_222.pth"
)


class VMambaSmallMambaCVCSOD(nn.Module):
    """MambaCVC framework reproduced on MLab's unmodified VMamba-S.

    Paper topology:
        stage features -> 32-channel normalization
        -> GC-VCM: 3 -> 2 -> 1 (top-down)
        -> Dec-4
        -> LC-VCM-3 -> Dec-3
        -> LC-VCM-2 -> Dec-2
        -> LC-VCM-1 -> Dec-1
        -> four scale-matched saliency predictions

    Difference intentionally kept for a fair MLab backbone comparison:
        The paper modifies the VMamba stem to output H/2 at Enc-1.
        This model leaves MLab VMamba-S unchanged (Enc-1 is H/4).
        Prediction maps are therefore resized to the same semantic target
        levels: S1=input, S2=stage1, S3=stage2, S4=stage3.
    """

    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        *,
        backbone: nn.Module | None = None,
        channels: int = 32,
    ) -> None:
        super().__init__()

        if backbone is None:
            from models.backbones.vmamba import vmamba_small

            backbone = vmamba_small(pretrained_path=pretrained_path)

        self.backbone = backbone
        out_channels = tuple(self.backbone.out_channels)
        if len(out_channels) != 4:
            raise ValueError("MambaCVC requires four backbone stages")

        # Yellow blocks under Enc-1..4 in Fig.2 are 3x3 Conv + BN + ReLU.
        self.normalize1 = ConvBNReLU(out_channels[0], channels, 3)
        self.normalize2 = ConvBNReLU(out_channels[1], channels, 3)
        self.normalize3 = ConvBNReLU(out_channels[2], channels, 3)
        self.normalize4 = ConvBNReLU(out_channels[3], channels, 3)

        self.gc3 = GCVCM(channels)
        self.gc2 = GCVCM(channels)
        self.gc1 = GCVCM(channels)

        self.dec4 = DecoderBlock(channels)
        self.lc3 = LCVCM(channels, groups=4)
        self.dec3 = DecoderBlock(channels)
        self.lc2 = LCVCM(channels, groups=4)
        self.dec2 = DecoderBlock(channels)
        self.lc1 = LCVCM(channels, groups=4)
        self.dec1 = DecoderBlock(channels)

        self.head1 = SaliencyHead(channels)
        self.head2 = SaliencyHead(channels)
        self.head3 = SaliencyHead(channels)
        self.head4 = SaliencyHead(channels)

    @staticmethod
    def _predict(
        feature: torch.Tensor,
        head: nn.Module,
        size: tuple[int, int],
    ) -> torch.Tensor:
        feature = F.interpolate(
            feature,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return head(feature)

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        input_size = image.shape[-2:]
        stage1, stage2, stage3, stage4 = self.backbone(image)

        fe1 = self.normalize1(stage1)
        fe2 = self.normalize2(stage2)
        fe3 = self.normalize3(stage3)
        fe4 = self.normalize4(stage4)

        # Fig.2 GC path: fe4 -> GC3 -> GC2 -> GC1.
        fn3 = self.gc3(fe3, fe4)
        fn2 = self.gc2(fe2, fn3)
        fn1 = self.gc1(fe1, fn2)

        # Fig.2 decoder/LC path: Dec4 -> LC3 -> Dec3 -> ... -> Dec1.
        fd4 = self.dec4(fe4)
        lc3 = self.lc3(fn3, fd4)
        fd3 = self.dec3(lc3)
        lc2 = self.lc2(fn2, fd3)
        fd2 = self.dec2(lc2)
        lc1 = self.lc1(fn1, fd2)
        fd1 = self.dec1(lc1)

        # In the paper each decoder map is upsampled before a blue 3x3 head.
        # With the original MLab H/4 stem, target exact semantic levels rather
        # than hard-coding x2, keeping S1 full-resolution for train.py/test.py.
        s1 = self._predict(fd1, self.head1, input_size)
        s2 = self._predict(fd2, self.head2, fe1.shape[-2:])
        s3 = self._predict(fd3, self.head3, fe2.shape[-2:])
        s4 = self._predict(fd4, self.head4, fe3.shape[-2:])

        return {
            "pred": s1,
            "aux": [s2, s3, s4],
        }


def build_model() -> VMambaSmallMambaCVCSOD:
    return VMambaSmallMambaCVCSOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("VMamba MambaCVC smoke test requires CUDA")

    model = build_model().cuda().eval()
    image = torch.randn(1, 3, 352, 352, device="cuda")
    with torch.no_grad():
        outputs = model(image=image)
    print("pred:", outputs["pred"].shape)
    print("aux:", [tensor.shape for tensor in outputs["aux"]])
