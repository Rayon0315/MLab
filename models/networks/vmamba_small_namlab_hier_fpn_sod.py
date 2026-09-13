from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.backbones.vmamba import (
    VMambaSmallBackbone,
    vmamba_small,
)
from models.components.namlab_hybrid import NAMLabHybrid
from models.components.progressive_decoder import (
    HierarchicalFPNDecoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vmamba"
    / "vssm_small_0229_ckpt_epoch_222.pth"
)


class VMambaSmallNAMLabHierFPNSOD(nn.Module):
    """
    VMamba-S + NAMLab Hybrid + Hierarchical FPN.

    Backbone:
        VMamba-S [96, 192, 384, 768]

    Feature enhancement:
        NAMLabHybrid

    Decoder:
        Stage4: 768 -> 384
        Stage3: 384 -> 384 -> 192
        Stage2: 192 -> 192 -> 128
        Stage1:  96 -> 128

    Fusion remains vanilla FPN addition.

    Excluded on purpose:
        PyramidContext
        persistent Stage4 global branch
        selective/gated fusion
        concat reconstruction
        boundary refinement
        dictionary routing
        disagreement refinement
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__()

        self.backbone: VMambaSmallBackbone = vmamba_small(
            pretrained_path=pretrained_path,
        )

        stage_channels = tuple(
            self.backbone.out_channels
        )

        self.namlab_hybrid = NAMLabHybrid(
            stage_channels=stage_channels,
            initial_context_scale=0.1,
        )

        self.decoder = HierarchicalFPNDecoder(
            in_channels=stage_channels,
            decoder_channels=(
                128,
                192,
                384,
                384,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]

        features = self.backbone(
            image
        )

        features = self.namlab_hybrid(
            features=features,
            image=image,
            mean_60=mean_60,
        )

        return self.decoder(
            features=features,
            output_size=input_size,
        )


def build_model() -> VMambaSmallNAMLabHierFPNSOD:
    return VMambaSmallNAMLabHierFPNSOD(
        pretrained_path=PRETRAINED_PATH,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    model = build_model().to(
        device
    )
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
        "decoder channels:",
        model.decoder.decoder_channels,
    )
