from __future__ import annotations

from pathlib import Path

import torch

from models.components.region_semantic_graph import RegionSemanticGraph
from models.networks.smt_t_pafr_shallow_ecm_sod import (
    PRETRAINED_PATH,
    SMTTinyPAFRShallowECMSOD,
)


class SMTTinyPAFRShallowECMRSGSOD(SMTTinyPAFRShallowECMSOD):
    """SMT-T + PAFR + Shallow ECM + Stage3 Region Semantic Graph.

    NAMLab region partitions define graph nodes only. Stage3 supplies regional
    semantics, Stage4 supplies global context, and RSG performs 1/2-hop
    region-level semantic reasoning before CRM3. Shallow ECM remains on S1/S2.
    """

    input_keys = ("image", "region_partition")

    def __init__(
        self,
        pretrained_path: str | Path | None,
        graph_dim: int = 128,
        graph_heads: int = 4,
        initial_reinject_scale: float = 0.1,
    ) -> None:
        super().__init__(pretrained_path=pretrained_path)

        self.region_semantic_graph = RegionSemanticGraph(
            stage3_channels=self.backbone.out_channels[2],
            stage4_channels=self.backbone.out_channels[3],
            token_dim=graph_dim,
            num_heads=graph_heads,
            initial_reinject_scale=initial_reinject_scale,
        )

    def forward(
        self,
        image: torch.Tensor,
        region_partition: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]

        stage1, stage2, stage3, stage4 = self.backbone(image)

        stage3 = self.region_semantic_graph(
            stage3=stage3,
            stage4=stage4,
            region_partition=region_partition,
        )

        stage5 = self.pool4(stage4)
        stage5 = self.stage5_projection(stage5)
        stage5 = self.stage5(stage5)

        opm_feature = self.opm(stage5)

        decoded5, _ = self.crm5(opm_feature, stage5)

        x = self._resize_local(decoded5, stage4.shape[-2:])
        decoded4, _ = self.crm4(x, stage4)

        x = self._resize_local(decoded4, stage3.shape[-2:])
        decoded3, _ = self.crm3(x, stage3)

        x = self._resize_local(decoded3, stage2.shape[-2:])
        decoded2, _ = self.crm2(x, stage2)

        x = self._resize_local(decoded2, stage1.shape[-2:])
        decoded1, _ = self.crm1(x, stage1)

        shallow_structure = self.ecm((stage1, stage2))
        decoded1 = self.structure_fusion(
            torch.cat((decoded1, shallow_structure), dim=1)
        )

        opm_logits = self._resize_local(self.out_opm(opm_feature), input_size)
        side5_logits = self._resize_local(self.out5(decoded5), input_size)
        side4_logits = self._resize_local(self.out4(decoded4), input_size)
        side3_logits = self._resize_local(self.out3(decoded3), input_size)
        side2_logits = self._resize_local(self.out2(decoded2), input_size)
        side1_logits = self._resize_local(self.out1(decoded1), input_size)

        final_logits = self.refunet(side1_logits)

        pafr_logits = [
            final_logits,
            side1_logits,
            side2_logits,
            side3_logits,
            side4_logits,
            side5_logits,
            opm_logits,
        ]

        return {
            "pred": final_logits,
            "aux": pafr_logits[1:],
            "pafr_logits": pafr_logits,
        }


def build_model() -> SMTTinyPAFRShallowECMRSGSOD:
    return SMTTinyPAFRShallowECMRSGSOD(
        pretrained_path=PRETRAINED_PATH,
    )
