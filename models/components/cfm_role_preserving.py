from __future__ import annotations

import torch
import torch.nn as nn


BRANCH_ORDER = (
    "tcrm",
    "ssrm",
    "ldfm",
    "egpcm",
)


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class Stage234BranchTap:
    """Capture the last CFMBlock's four branch outputs at Stage 2/3/4."""

    STAGES = {
        "stage2": 2,
        "stage3": 4,
        "stage4": 6,
    }

    BACKBONE_NAMES = {
        "tcrm": "TCRM",
        "ssrm": "SSRM",
        "ldfm": "LDFM",
        "egpcm": "EGPCM",
    }

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        self._outputs: dict[
            str,
            dict[str, torch.Tensor],
        ] = {
            stage_name: {}
            for stage_name in self.STAGES
        }

        self._handles = []

        for stage_name, stage_index in self.STAGES.items():
            block = backbone.stages[stage_index].blocks[-1]

            for branch_name, module_name in self.BACKBONE_NAMES.items():
                self._register(
                    stage_name=stage_name,
                    branch_name=branch_name,
                    module=getattr(
                        block,
                        module_name,
                    ),
                )

    def _register(
        self,
        stage_name: str,
        branch_name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[stage_name][branch_name] = output

        self._handles.append(
            module.register_forward_hook(hook)
        )

    def clear(self) -> None:
        for stage_name in self._outputs:
            self._outputs[stage_name].clear()

    def pop(
        self,
    ) -> dict[
        str,
        dict[str, torch.Tensor],
    ]:
        expected = set(BRANCH_ORDER)

        for stage_name in self.STAGES:
            missing = expected - self._outputs[stage_name].keys()

            if missing:
                raise RuntimeError(
                    f"Failed to capture {stage_name}: "
                    + ", ".join(sorted(missing))
                )

        outputs = self._outputs

        self._outputs = {
            stage_name: {}
            for stage_name in self.STAGES
        }

        return outputs


class FourRoleProjector(nn.Module):
    """Project each CFM branch independently into decoder channels."""

    def __init__(
        self,
        stage_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        if stage_channels % 4 != 0:
            raise ValueError(
                "CFMNet stage channels must be divisible by four."
            )

        branch_channels = stage_channels // 4

        self.projectors = nn.ModuleDict(
            {
                name: _ConvBNReLU(
                    branch_channels,
                    decode_channels,
                )
                for name in BRANCH_ORDER
            }
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.projectors[name](
                branches[name]
            )
            for name in BRANCH_ORDER
        }


class FourRoleRouter(nn.Module):
    """
    Keep four CFM roles separate instead of mixing branch features.

    TCRM:
        semantic matching signal -> conditions Q/K.

    SSRM:
        directional structure signal -> corrects the directional
        strip-pooling path after window attention.

    LDFM:
        local detail signal -> conditions the local convolution path.

    EGPCM:
        contextual content signal -> conditions attention values V.

    All four scales start at zero, so the initial decoder behavior is
    the original UNetFormer block driven only by the fused CFM stage
    feature. Branch roles are learned as residual conditions.
    """

    def __init__(self) -> None:
        super().__init__()

        self.tcrm_scale = nn.Parameter(
            torch.zeros(1)
        )
        self.ssrm_scale = nn.Parameter(
            torch.zeros(1)
        )
        self.ldfm_scale = nn.Parameter(
            torch.zeros(1)
        )
        self.egpcm_scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        tcrm: torch.Tensor,
        ssrm: torch.Tensor,
        ldfm: torch.Tensor,
        egpcm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "qk_input": (
                x
                + self.tcrm_scale
                * tcrm
            ),
            "value_input": (
                x
                + self.egpcm_scale
                * egpcm
            ),
            "local_input": (
                x
                + self.ldfm_scale
                * ldfm
            ),
            "directional_delta": (
                self.ssrm_scale
                * ssrm
            ),
        }

    def scales(
        self,
    ) -> dict[str, torch.Tensor]:
        return {
            "tcrm": self.tcrm_scale,
            "ssrm": self.ssrm_scale,
            "ldfm": self.ldfm_scale,
            "egpcm": self.egpcm_scale,
        }
