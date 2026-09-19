"""NAMLab evidence integrated *inside* the successful saliency task stream.

Motivation
----------
The current best model uses three outer streams at F2/F3:
    base + IRAM frequency + saliency-task stream

A previous experiment appended region-mean as a fourth outer stream.  That
helped SaliencyAdapter-only, but hurt the full IRAM+Adapter model.  This file
keeps the successful three-stream outer interface unchanged and lets NAMLab
own the *region evidence inside the task stream* instead.

The actual validated ``NAMLabHybrid`` component is reused.  Its enhancement
relative to the original fused CFMNet pyramid is interpreted as NAM-specific
region evidence:
    delta_i = NAMLabHybrid(F)_i - F_i

Only F2/F3 deltas are converted to the common 128-channel task grid.  No
backbone internals and no UNetFormer internals are touched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import (
    ConvBNAct,
    FrequencyReconstructionAdapter,
)
from models.components.cfm_iram_strong_common import (
    ThreeStreamFusion,
    resize_like,
)
from models.components.namlab_hybrid import NAMLabHybrid


STAGE_CHANNELS = (96, 192, 384, 768)


class NAMLabDeltaEvidence(nn.Module):
    """Turn the validated NAMLabHybrid correction into one task feature.

    NAMLabHybrid already performs region/visual interaction.  Subtracting the
    original fused feature removes the duplicated base representation and
    keeps the information introduced by NAMLab itself.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()

        self.namlab = NAMLabHybrid(
            stage_channels=STAGE_CHANNELS,
            initial_context_scale=0.1,
        )

        half = channels // 2

        self.project2 = ConvBNAct(
            STAGE_CHANNELS[1],
            half,
            kernel_size=1,
        )
        self.project3 = ConvBNAct(
            STAGE_CHANNELS[2],
            channels - half,
            kernel_size=1,
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                channels,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
            ),
        )

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        f1, f2, f3, f4 = features

        n1, n2, n3, n4 = self.namlab(
            features=(f1, f2, f3, f4),
            image=image,
            mean_60=mean_60,
        )

        delta2 = n2 - f2
        delta3 = n3 - f3

        nam2 = self.project2(delta2)
        nam3 = resize_like(
            self.project3(delta3),
            f2,
        )

        return self.fuse(
            torch.cat(
                [nam2, nam3],
                dim=1,
            )
        )


class NAMLabInternalSaliencyAdapterStream(nn.Module):
    """SaliencyAdapterStream with NAMLab inserted at the region-evidence level.

    Modes
    -----
    replace:
        semantic + NAM-region + structure
        The original learned region branch is replaced.  This is the strongest
        test of whether explicit NAMLab region evidence is the better region
        representation.

    hybrid:
        semantic + fuse(original-region, NAM-region) + structure
        Both region sources jointly reconstruct a single region branch.

    augment:
        semantic + original-region + structure + NAM-region
        NAM becomes a fourth *internal task* branch, but the outer best-model
        interface remains base + frequency + one 128-channel task stream.
    """

    VALID_MODES = ("replace", "hybrid", "augment")

    def __init__(
        self,
        channels: int = 128,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode={mode!r}; expected one of {self.VALID_MODES}."
            )

        self.mode = mode

        # Keep the successful SaliencyAdapter feature construction unchanged.
        self.project = nn.ModuleList(
            [
                ConvBNAct(c, 64, kernel_size=1)
                for c in STAGE_CHANNELS
            ]
        )

        self.semantic = ConvBNAct(
            128,
            channels,
            kernel_size=3,
        )

        self.region = nn.Sequential(
            ConvBNAct(
                64 + channels,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
            ),
        )

        self.structure = ConvBNAct(
            128,
            channels,
            kernel_size=3,
        )

        self.nam_region = NAMLabDeltaEvidence(
            channels=channels,
        )

        if mode == "hybrid":
            self.region_hybrid = nn.Sequential(
                ConvBNAct(
                    2 * channels,
                    channels,
                    kernel_size=1,
                ),
                ConvBNAct(
                    channels,
                    channels,
                    kernel_size=3,
                ),
            )
        else:
            self.region_hybrid = None

        reconstruct_inputs = (
            4 * channels
            if mode == "augment"
            else 3 * channels
        )

        self.reconstruct = nn.Sequential(
            ConvBNAct(
                reconstruct_inputs,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
            ),
        )

        self.coarse_head = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

        self.saliency2 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
        )
        self.saliency3 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
        )

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(
                self.project,
                features,
            )
        ]

        semantic = self.semantic(
            torch.cat(
                [
                    resize_like(p3, p2),
                    resize_like(p4, p2),
                ],
                dim=1,
            )
        )

        region = self.region(
            torch.cat(
                [
                    p2,
                    F.avg_pool2d(
                        semantic,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                ],
                dim=1,
            )
        )

        high1 = p1 - F.avg_pool2d(
            p1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        high2 = p2 - F.avg_pool2d(
            p2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        structure = self.structure(
            torch.cat(
                [
                    resize_like(high1, p2),
                    high2,
                ],
                dim=1,
            )
        )

        nam_region = self.nam_region(
            features=features,
            image=image,
            mean_60=mean_60,
        )

        if self.mode == "replace":
            task_inputs = [
                semantic,
                nam_region,
                structure,
            ]
        elif self.mode == "hybrid":
            region = self.region_hybrid(
                torch.cat(
                    [region, nam_region],
                    dim=1,
                )
            )
            task_inputs = [
                semantic,
                region,
                structure,
            ]
        else:
            task_inputs = [
                semantic,
                region,
                structure,
                nam_region,
            ]

        task = self.reconstruct(
            torch.cat(
                task_inputs,
                dim=1,
            )
        )

        return (
            self.saliency2(task),
            self.saliency3(
                resize_like(task, p3)
            ),
            self.coarse_head(task),
        )


class IRAMNAMLabInternalNeck(nn.Module):
    """Keep the successful outer three-stream neck exactly in spirit.

    Outer streams remain:
        base + IRAM frequency + one saliency-task stream

    NAMLab is consumed *inside* the task stream, so adding NAM does not widen
    the outer fusion from three to four streams.
    """

    def __init__(self, mode: str = "hybrid") -> None:
        super().__init__()

        self.x_stream = NAMLabInternalSaliencyAdapterStream(
            mode=mode,
        )

        self.freq2 = FrequencyReconstructionAdapter(192)
        self.freq3 = FrequencyReconstructionAdapter(384)

        # Strong intervention, identical to the successful IRAM+Adapter setup.
        with torch.no_grad():
            self.freq2.scale.fill_(1.0)
            self.freq3.scale.fill_(1.0)

        self.fuse2 = ThreeStreamFusion(192)
        self.fuse3 = ThreeStreamFusion(384)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        f1, f2, f3, f4 = features

        sal2, sal3, coarse = self.x_stream(
            features=features,
            image=image,
            mean_60=mean_60,
        )

        out2 = self.fuse2(
            f2,
            self.freq2(f2),
            sal2,
        )
        out3 = self.fuse3(
            f3,
            self.freq3(f3),
            sal3,
        )

        return [f1, out2, out3, f4], coarse
