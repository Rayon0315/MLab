"""Leave-one-evidence-out ablations for the validated SaliencyAdapterStream.

The full adapter is kept structurally unchanged. Every ablation still computes all
three branches and keeps the original reconstruction/fusion widths, parameter
count, auxiliary head, and downstream IRAM/UNetFormer interface. The selected
evidence is replaced by an all-zero tensor before it can influence the task
representation.

For the semantic ablation, the zeroed semantic tensor is also passed to the
region branch. This is intentional: in the full adapter semantic evidence enters
both the final three-way reconstruction and the region branch through the pooled
semantic context. Zeroing it at both sites removes semantic evidence rather than
leaking it through the region path.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from models.components.cfm_iram_saliency_adapter import SaliencyAdapterStream
from models.components.cfm_iram_strong_common import resize_like


class SaliencyAdapterLeaveOneOutStream(SaliencyAdapterStream):
    """Exact full adapter with one evidence family silenced.

    Args:
        ablated: one of ``semantic``, ``region``, or ``structure``.

    Notes:
        * Branch modules are still instantiated and executed. This preserves the
          model architecture, parameter count, and nearly identical forward cost.
        * The ablated branch has no influence on ``task`` and therefore receives
          no task gradient, which is the intended leave-one-evidence-out behavior.
        * The outer IRAM branch, ThreeStreamFusion, coarse auxiliary supervision,
          backbone, and UNetFormer decoder remain unchanged.
    """

    VALID_ABLATIONS = {"semantic", "region", "structure"}

    def __init__(self, channels: int = 128, ablated: str = "semantic") -> None:
        super().__init__(channels=channels)
        if ablated not in self.VALID_ABLATIONS:
            raise ValueError(
                f"ablated must be one of {sorted(self.VALID_ABLATIONS)}, got {ablated!r}"
            )
        self.ablated = ablated

    @staticmethod
    def _zero_like(feature: torch.Tensor) -> torch.Tensor:
        # zeros_like intentionally severs the selected evidence path while
        # preserving shape/dtype/device and the downstream reconstruction graph.
        return torch.zeros_like(feature)

    def forward(self, features):
        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(self.project, features)
        ]

        # --------------------------- semantic evidence ---------------------------
        semantic_raw = self.semantic(
            torch.cat(
                [
                    resize_like(p3, p2),
                    resize_like(p4, p2),
                ],
                dim=1,
            )
        )

        semantic = (
            self._zero_like(semantic_raw)
            if self.ablated == "semantic"
            else semantic_raw
        )

        # In the validated adapter, pooled semantic context conditions the region
        # stream. For w/o semantic we must also remove that route, otherwise the
        # supposedly ablated evidence leaks through ``region``.
        region_raw = self.region(
            torch.cat(
                [
                    p2,
                    F.avg_pool2d(
                        semantic,
                        7,
                        stride=1,
                        padding=3,
                    ),
                ],
                dim=1,
            )
        )

        region = (
            self._zero_like(region_raw)
            if self.ablated == "region"
            else region_raw
        )

        # -------------------------- structural evidence --------------------------
        high1 = p1 - F.avg_pool2d(
            p1,
            3,
            stride=1,
            padding=1,
        )
        high2 = p2 - F.avg_pool2d(
            p2,
            3,
            stride=1,
            padding=1,
        )

        structure_raw = self.structure(
            torch.cat(
                [
                    resize_like(high1, p2),
                    high2,
                ],
                dim=1,
            )
        )

        structure = (
            self._zero_like(structure_raw)
            if self.ablated == "structure"
            else structure_raw
        )

        # Reconstruction is EXACTLY the same 3*C -> C operation as the full
        # adapter. No channel reduction, new fusion, residual scale, or gate is
        # introduced by the ablation.
        task = self.reconstruct(
            torch.cat(
                [
                    semantic,
                    region,
                    structure,
                ],
                dim=1,
            )
        )

        return (
            self.saliency2(task),
            self.saliency3(resize_like(task, p3)),
            self.coarse_head(task),
        )
