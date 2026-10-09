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
from models.components.mambacvc_namlab_local_cross import (
    RegionMeanLocalCrossGuidance,
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


class VMambaSmallMambaCVCNAMLabLocalCrossSOD(nn.Module):
    """VMamba-S + MambaCVC + NAM region-mean local cross guidance.

    Data flow:
        VMamba -> 32ch normalization -> GC-VCM chain
        -> unified NAM M60 local bidirectional guidance at fn1/fn2/fn3
        -> LC-VCM decoder chain -> four supervised saliency heads.

    The three guidance blocks use exactly the same architecture but independent
    parameters. No stage-specific semantic/detail roles are assigned.
    """

    input_keys = ("image", "mean_60")
    mean_hierarchies = (60,)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        *,
        backbone: nn.Module | None = None,
        channels: int = 32,
        guidance_inner_channels: int = 16,
        guidance_heads: int = 4,
        guidance_window: int = 3,
    ) -> None:
        super().__init__()

        if backbone is None:
            from models.backbones.vmamba import vmamba_small

            backbone = vmamba_small(pretrained_path=pretrained_path)

        self.backbone = backbone
        out_channels = tuple(self.backbone.out_channels)
        if len(out_channels) != 4:
            raise ValueError("MambaCVC requires four backbone stages")

        self.normalize1 = ConvBNReLU(out_channels[0], channels, 3)
        self.normalize2 = ConvBNReLU(out_channels[1], channels, 3)
        self.normalize3 = ConvBNReLU(out_channels[2], channels, 3)
        self.normalize4 = ConvBNReLU(out_channels[3], channels, 3)

        self.gc3 = GCVCM(channels)
        self.gc2 = GCVCM(channels)
        self.gc1 = GCVCM(channels)

        guidance_kwargs = dict(
            channels=channels,
            inner_channels=guidance_inner_channels,
            num_heads=guidance_heads,
            window_size=guidance_window,
        )
        self.guidance3 = RegionMeanLocalCrossGuidance(**guidance_kwargs)
        self.guidance2 = RegionMeanLocalCrossGuidance(**guidance_kwargs)
        self.guidance1 = RegionMeanLocalCrossGuidance(**guidance_kwargs)

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

        # Original MambaCVC global path is kept unchanged.
        fn3 = self.gc3(fe3, fe4)
        fn2 = self.gc2(fe2, fn3)
        fn1 = self.gc1(fe1, fn2)

        # New GC->LC transition: all three scales use the same NAM M60
        # region-preserving encoder + local bidirectional cross guidance.
        gn3 = self.guidance3(fn3, mean_60)
        gn2 = self.guidance2(fn2, mean_60)
        gn1 = self.guidance1(fn1, mean_60)

        # Original LC decoder path is kept unchanged except for receiving
        # the NAM-reconstructed features instead of the raw GC outputs.
        fd4 = self.dec4(fe4)
        lc3 = self.lc3(gn3, fd4)
        fd3 = self.dec3(lc3)
        lc2 = self.lc2(gn2, fd3)
        fd2 = self.dec2(lc2)
        lc1 = self.lc1(gn1, fd2)
        fd1 = self.dec1(lc1)

        s1 = self._predict(fd1, self.head1, input_size)
        s2 = self._predict(fd2, self.head2, fe1.shape[-2:])
        s3 = self._predict(fd3, self.head3, fe2.shape[-2:])
        s4 = self._predict(fd4, self.head4, fe3.shape[-2:])

        return {
            "pred": s1,
            "aux": [s2, s3, s4],
        }


def build_model() -> VMambaSmallMambaCVCNAMLabLocalCrossSOD:
    return VMambaSmallMambaCVCNAMLabLocalCrossSOD(
        pretrained_path=PRETRAINED_PATH,
        channels=32,
        guidance_inner_channels=16,
        guidance_heads=4,
        guidance_window=3,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("VMamba MambaCVC NAMLab smoke test requires CUDA")

    model = build_model().cuda().eval()
    image = torch.randn(1, 3, 352, 352, device="cuda")
    mean_60 = torch.rand(1, 3, 352, 352, device="cuda")
    with torch.no_grad():
        outputs = model(image=image, mean_60=mean_60)
    print("pred:", outputs["pred"].shape)
    print("aux:", [tensor.shape for tensor in outputs["aux"]])
