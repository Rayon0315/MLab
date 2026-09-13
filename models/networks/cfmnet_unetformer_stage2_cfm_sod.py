# models/networks/cfmnet_unetformer_stage2_cfm_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cfmnet import CFMNET_OUT_CHANNELS, build_cfmnet
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
    ConvBNReLU,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class Stage2BranchTap:
    """
    Capture the four branch outputs from the last CFMBlock of paper Stage 2.

    CFMNet stores modules as:
        0 BasicStage(96)
        1 Downsample
        2 BasicStage(192)   <- paper Stage 2
        3 Downsample
        4 BasicStage(384)
        5 Downsample
        6 BasicStage(768)

    Stage 2 has 192 channels, so each CFM branch has 48 channels.
    Hooks only expose already-computed features; they do not add backbone FLOPs.
    """

    def __init__(self, backbone: nn.Module) -> None:
        stage2_last_block = backbone.stages[2].blocks[-1]
        self._outputs: dict[str, torch.Tensor] = {}
        self._handles = []

        self._register("tcrm", stage2_last_block.TCRM)
        self._register("ssrm", stage2_last_block.SSRM)
        self._register("ldfm", stage2_last_block.LDFM)
        self._register("egpcm", stage2_last_block.EGPCM)

    def _register(self, name: str, module: nn.Module) -> None:
        def hook(_module, _inputs, output):
            self._outputs[name] = output

        self._handles.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        self._outputs.clear()

    def pop(self) -> dict[str, torch.Tensor]:
        expected = {"tcrm", "ssrm", "ldfm", "egpcm"}
        missing = expected - self._outputs.keys()
        if missing:
            raise RuntimeError(
                "Failed to capture CFMNet Stage2 branches: "
                + ", ".join(sorted(missing))
            )

        outputs = self._outputs
        self._outputs = {}
        return outputs


class Stage2CooperativeAdapter(nn.Module):
    """
    Convert CFMNet Stage-2 heterogeneous subspaces into one SOD-aware residual.

    Semantic stream:
        TCRM + EGPCM

    Structure/detail stream:
        SSRM + LDFM

    High-level Stage-3 decoder semantics first refine the semantic stream.
    The semantic stream then generates a spatial gate for structure/detail.
    """

    def __init__(
        self,
        branch_channels: int = 48,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        pair_channels = branch_channels * 2

        self.semantic_proj = ConvBNReLU(
            pair_channels,
            decode_channels,
            kernel_size=1,
        )
        self.detail_proj = ConvBNReLU(
            pair_channels,
            decode_channels,
            kernel_size=1,
        )
        self.high_proj = ConvBNReLU(
            decode_channels,
            decode_channels,
            kernel_size=1,
        )

        self.semantic_fusion = ConvBNReLU(
            decode_channels * 2,
            decode_channels,
            kernel_size=3,
        )

        self.saliency_gate = nn.Sequential(
            nn.Conv2d(
                decode_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.cooperative_fusion = ConvBNReLU(
            decode_channels * 2,
            decode_channels,
            kernel_size=3,
        )

    def forward(
        self,
        high_feature: torch.Tensor,
        branch_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        semantic = torch.cat(
            [
                branch_features["tcrm"],
                branch_features["egpcm"],
            ],
            dim=1,
        )
        detail = torch.cat(
            [
                branch_features["ssrm"],
                branch_features["ldfm"],
            ],
            dim=1,
        )

        semantic = self.semantic_proj(semantic)
        detail = self.detail_proj(detail)

        high_feature = F.interpolate(
            high_feature,
            size=semantic.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_feature = self.high_proj(high_feature)

        semantic = self.semantic_fusion(
            torch.cat(
                [semantic, high_feature],
                dim=1,
            )
        )

        gate = self.saliency_gate(semantic)
        guided_detail = detail * gate

        cooperative = self.cooperative_fusion(
            torch.cat(
                [semantic, guided_detail],
                dim=1,
            )
        )
        return cooperative


class CFMNetStage2AwareDecoder(CFMNetUNetFormerDecoder):
    """
    Original CFMNet + UNetFormer-style decoder, with one Stage-2 intervention.

    Baseline:
        x2_base = fuse2(x3, stage2)

    Added path:
        Stage2 TCRM/EGPCM -> semantic
        Stage2 SSRM/LDFM -> structure/detail
        x3 -> high-level semantic guidance

        x2 = x2_base + alpha * cfm_feature

    alpha starts at zero, so training begins from the original decoder path.
    """

    def __init__(
        self,
        encoder_channels=CFMNET_OUT_CHANNELS,
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes,
        )

        stage2_channels = encoder_channels[1]
        if stage2_channels % 4 != 0:
            raise ValueError("CFMNet Stage2 channels must be divisible by 4")

        self.cfm_adapter = Stage2CooperativeAdapter(
            branch_channels=stage2_channels // 4,
            decode_channels=decode_channels,
        )

        # Same progressive-activation idea used inside CFMNet itself.
        self.cfm_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        features: list[torch.Tensor],
        stage2_branches: dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stage1, stage2, stage3, stage4 = features

        x4 = self.block4(
            self.pre_conv(stage4)
        )

        x3 = self.block3(
            self.fuse3(x4, stage3)
        )

        x2_base = self.fuse2(
            x3,
            stage2,
        )

        cfm_feature = self.cfm_adapter(
            high_feature=x3,
            branch_features=stage2_branches,
        )

        x2 = x2_base + self.cfm_scale * cfm_feature
        x2 = self.block2(x2)

        x1 = self.refine1(
            x2,
            stage1,
        )

        prediction = self.segmentation_head(x1)
        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = None
        if self.training:
            aux_size = x2.shape[-2:]
            aux4 = F.interpolate(
                x4,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )
            aux3 = F.interpolate(
                x3,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            auxiliary = self.aux_head(
                aux4 + aux3 + x2,
                output_size=output_size,
            )

        return prediction, auxiliary


class CFMNetUNetFormerStage2CFMSOD(nn.Module):
    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decode_channels: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__()

        self.backbone = build_cfmnet(
            pretrained_path=pretrained_path
        )

        self.stage2_tap = Stage2BranchTap(
            self.backbone
        )

        self.decoder = CFMNetStage2AwareDecoder(
            encoder_channels=CFMNET_OUT_CHANNELS,
            decode_channels=decode_channels,
            dropout=0.1,
            window_size=window_size,
            num_classes=1,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        output_size = image.shape[-2:]

        self.stage2_tap.clear()
        features = self.backbone(image)
        stage2_branches = self.stage2_tap.pop()

        prediction, auxiliary = self.decoder(
            features=features,
            stage2_branches=stage2_branches,
            output_size=output_size,
        )

        outputs = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs["aux"] = [auxiliary]

        return outputs


def build_model() -> CFMNetUNetFormerStage2CFMSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
        warnings.warn(
            "CFMNet ImageNet-1K checkpoint was not found at "
            f"{PRETRAINED_PATH}. The backbone will train from scratch.",
            RuntimeWarning,
        )

    return CFMNetUNetFormerStage2CFMSOD(
        pretrained_path=pretrained_path,
        decode_channels=64,
        window_size=8,
    )


if __name__ == "__main__":
    model = build_model()
    image = torch.randn(2, 3, 352, 352)

    model.train()
    outputs = model(image=image)
    print("train pred:", outputs["pred"].shape)
    print("train aux:", [x.shape for x in outputs.get("aux", [])])
    print("cfm_scale:", model.decoder.cfm_scale.detach().item())

    model.eval()
    with torch.no_grad():
        outputs = model(image=image)
    print("eval pred:", outputs["pred"].shape)
