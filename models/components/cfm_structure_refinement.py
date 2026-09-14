from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1StructureTap:
    def __init__(self, backbone: nn.Module) -> None:
        block = backbone.stages[0].blocks[-1]
        self._outputs: dict[str, torch.Tensor] = {}
        self._handles = [
            block.SSRM.register_forward_hook(self._make_hook("ssrm")),
            block.LDFM.register_forward_hook(self._make_hook("ldfm")),
        ]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._outputs[name] = output
        return hook

    def clear(self) -> None:
        self._outputs.clear()

    def pop(self) -> dict[str, torch.Tensor]:
        missing = {"ssrm", "ldfm"} - self._outputs.keys()
        if missing:
            raise RuntimeError(
                "Failed to capture Stage1 structure branches: "
                + ", ".join(sorted(missing))
            )
        outputs = self._outputs
        self._outputs = {}
        return outputs


class Stage1StructureEncoder(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                branch_channels * 2,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.proj(
            torch.cat(
                [branches["ssrm"], branches["ldfm"]],
                dim=1,
            )
        )


class BoundaryRestrictedRefinement(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        hidden_channels: int = 64,
        boundary_kernel: int = 5,
    ) -> None:
        super().__init__()
        if boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be odd.")

        self.boundary_kernel = boundary_kernel
        self.structure_encoder = Stage1StructureEncoder(
            branch_channels=branch_channels,
            hidden_channels=hidden_channels,
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def _boundary_band(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits.detach().float())
        pad = self.boundary_kernel // 2
        dilation = F.max_pool2d(
            p,
            kernel_size=self.boundary_kernel,
            stride=1,
            padding=pad,
        )
        erosion = -F.max_pool2d(
            -p,
            kernel_size=self.boundary_kernel,
            stride=1,
            padding=pad,
        )
        return (dilation - erosion).clamp(0.0, 1.0).to(logits.dtype)

    def forward(
        self,
        logits: torch.Tensor,
        branches: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structure = self.structure_encoder(branches)
        correction = self.correction_head(structure)
        correction = F.interpolate(
            correction,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        band = self._boundary_band(logits)
        refined = logits + self.scale * band * correction
        return refined, band


class StructureConsistencyHead(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.structure_encoder = Stage1StructureEncoder(
            branch_channels=branch_channels,
            hidden_channels=hidden_channels,
        )
        self.head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        logits = self.head(self.structure_encoder(branches))
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class DirectionalThinStructureRecovery(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        hidden_channels: int = 64,
        support_kernel: int = 9,
    ) -> None:
        super().__init__()
        if support_kernel % 2 == 0:
            raise ValueError("support_kernel must be odd.")

        self.support_kernel = support_kernel
        self.structure_encoder = Stage1StructureEncoder(
            branch_channels=branch_channels,
            hidden_channels=hidden_channels,
        )
        self.horizontal = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def _thin_gate_and_weights(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(logits.detach().float())
        k = self.support_kernel
        pad = k // 2

        h_support = F.avg_pool2d(
            p,
            kernel_size=(1, k),
            stride=1,
            padding=(0, pad),
        )
        v_support = F.avg_pool2d(
            p,
            kernel_size=(k, 1),
            stride=1,
            padding=(pad, 0),
        )
        compact = F.avg_pool2d(
            p,
            kernel_size=k,
            stride=1,
            padding=pad,
        )
        directional = torch.maximum(h_support, v_support)
        thinness = (directional - compact).clamp_min(0.0)
        near_fg = F.max_pool2d(
            p,
            kernel_size=k,
            stride=1,
            padding=pad,
        )
        gate = (2.0 * thinness * near_fg).clamp(0.0, 1.0)

        denom = h_support + v_support + 1e-6
        h_weight = h_support / denom
        v_weight = v_support / denom

        return (
            gate.to(logits.dtype),
            h_weight.to(logits.dtype),
            v_weight.to(logits.dtype),
        )

    def forward(
        self,
        logits: torch.Tensor,
        branches: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structure = self.structure_encoder(branches)
        h = self.horizontal(structure)
        v = self.vertical(structure)

        h = F.interpolate(
            h,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        v = F.interpolate(
            v,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        gate, h_weight, v_weight = self._thin_gate_and_weights(logits)
        correction = h_weight * h + v_weight * v
        refined = logits + self.scale * gate * correction
        return refined, gate
