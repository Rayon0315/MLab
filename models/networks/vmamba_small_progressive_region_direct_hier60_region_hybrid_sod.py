# models/networks/vmamba_small_progressive_region_direct_hier60_region_hybrid_sod.py

from __future__ import annotations

from pathlib import Path

import torch

from models.backbones.vmamba import (
    VMambaSmallBackbone,
    vmamba_small,
)

from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vmamba"
    / "vssm_small_0229_ckpt_epoch_222.pth"
)


class VMambaSmallProgressiveRegionDirectHier60RegionHybridSOD(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """
    VMamba-S backbone variant of the current validated
    Hier60 Hybrid Region Direct model.

    Controlled replacement:

        MambaVision-S
            ->
        VMamba-S [s2l15]

    Unchanged:
        - Hier60 region assignment
        - Stage1 RGB-M60 detail encoder
        - Stage2/3/4 Hybrid Region Mean encoders
        - region/visual interaction
        - progressive decoder
        - global semantic stream
        - boundary refinement
        - prediction heads
        - model inputs
        - loss/training protocol

    Both backbones expose:

        Stage1:  96 channels, stride 4
        Stage2: 192 channels, stride 8
        Stage3: 384 channels, stride 16
        Stage4: 768 channels, stride 32
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        # Build the exact current Hybrid Region Direct network,
        # but do not load MambaVision pretrained weights.
        #
        # This preserves all validated downstream modules and
        # their initialization/structure.
        super().__init__(
            pretrained_path=None,
        )

        # MambaVision-S and VMamba-S have exactly the same
        # four output channel widths:
        #
        #   96 / 192 / 384 / 768
        #
        # Therefore every existing region encoder,
        # interaction block and decoder can be reused
        # without modification.
        self.backbone: VMambaSmallBackbone = (
            vmamba_small(
                pretrained_path=pretrained_path,
            )
        )


def build_model(
) -> VMambaSmallProgressiveRegionDirectHier60RegionHybridSOD:
    return (
        VMambaSmallProgressiveRegionDirectHier60RegionHybridSOD(
            pretrained_path=PRETRAINED_PATH,
        )
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device("cuda")

    model = build_model()
    model = model.to(device)
    model.eval()

    image = torch.randn(
        1,
        3,
        352,
        352,
        device=device,
    )

    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )

    if "aux" in outputs:
        print(
            "aux:",
            [
                tensor.shape
                for tensor in outputs["aux"]
            ],
        )

    print(
        "backbone channels:",
        model.backbone.out_channels,
    )

    print(
        "backbone strides:",
        model.backbone.out_strides,
    )