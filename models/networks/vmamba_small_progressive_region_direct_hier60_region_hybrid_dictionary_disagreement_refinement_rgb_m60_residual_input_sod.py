from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.vmamba_small_progressive_region_direct_hier60_region_hybrid_dictionary_disagreement_refinement_sod import (
    PRETRAINED_PATH,
    VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementRefinementSOD,
)


class VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementRefinementRGBM60ResidualInputSOD(
    VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementRefinementSOD
):
    """
    Current best model with explicit input decomposition:

        [RGB, M60, RGB - M60] -> 9 channels -> VMamba patch embedding

    M60 is normalized into the same ImageNet space as RGB.

    The pretrained VMamba patch embedding is expanded from 3 to 9 input
    channels. The original RGB weights are copied into channels 0:3 and the
    extra six channel weights are initialized to zero, so the initial backbone
    behavior is exactly the original pretrained RGB path.

    Everything after the backbone remains unchanged.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        num_prototypes: int = 12,
        dictionary_dim: int = 128,
        dictionary_temperature: float = 0.2,
        routing_strength: float = 0.5,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            num_prototypes=num_prototypes,
            dictionary_dim=dictionary_dim,
            dictionary_temperature=dictionary_temperature,
            routing_strength=routing_strength,
        )

        self._expand_patch_embed_to_nine_channels()

    def _expand_patch_embed_to_nine_channels(self) -> None:
        old_conv = self.backbone.patch_embed[0]

        if not isinstance(old_conv, nn.Conv2d):
            raise TypeError(
                "Expected VMamba patch_embed[0] to be nn.Conv2d."
            )

        new_conv = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )

        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, 0:3].copy_(old_conv.weight)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.backbone.patch_embed[0] = new_conv

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]

        mean60_normalized = self.region_hierarchy.normalize_mean_map(
            mean_60
        )
        region_residual = image - mean60_normalized

        backbone_input = torch.cat(
            [
                image,
                mean60_normalized,
                region_residual,
            ],
            dim=1,
        )

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            backbone_input
        )

        raw_stage2 = stage2
        raw_stage3 = stage3
        raw_stage4 = stage4

        type_field = self.latent_type_dictionary(
            stage2=raw_stage2,
            stage3=raw_stage3,
            stage4=raw_stage4,
        )

        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_hierarchy(
            image=image,
            mean_20=mean_60,
            mean_40=mean_60,
            mean_60=mean_60,
        )

        (
            region1,
            region2,
            region3,
            region4,
        ) = self.region_encoder(
            detail_region=detail_region,
            fine_region=fine_region,
            middle_region=middle_region,
            coarse_region=coarse_region,
            stage1_size=stage1.shape[-2:],
            stage2_size=stage2.shape[-2:],
            stage3_size=stage3.shape[-2:],
            stage4_size=stage4.shape[-2:],
        )

        stage4 = self.stage4_region_interaction(
            visual_feature=stage4,
            region_feature=region4,
        )

        stage3 = self.stage3_region_interaction(
            visual_feature=stage3,
            region_feature=region3,
        )

        stage2 = self.stage2_region_reconstruction(
            visual_feature=stage2,
            region_feature=region2,
        )

        stage1 = self.stage1_detail_reconstruction(
            visual_feature=stage1,
            detail_feature=region1,
        )

        decoded4 = self.deep_projection(stage4)
        decoded4 = self.context4(decoded4)
        prediction4 = self.pred4(decoded4)

        global3 = self.global3(
            stage4,
            target_size=stage3.shape[-2:],
        )

        (
            routed_stage3,
            routed_decoded4,
            routed_global3,
        ) = self.dictionary_router3(
            low_feature=stage3,
            high_feature=decoded4,
            global_feature=global3,
            type_field=type_field,
        )

        decoded3 = self.fusion3(
            low_feature=routed_stage3,
            high_feature=routed_decoded4,
            global_feature=routed_global3,
        )

        prediction3 = self.pred3(decoded3)
        decoded3_reduced = self.reduce3(decoded3)

        global2 = self.global2(
            stage4,
            target_size=stage2.shape[-2:],
        )

        (
            routed_stage2,
            routed_decoded3,
            routed_global2,
        ) = self.dictionary_router2(
            low_feature=stage2,
            high_feature=decoded3_reduced,
            global_feature=global2,
            type_field=type_field,
        )

        decoded2 = self.fusion2(
            low_feature=routed_stage2,
            high_feature=routed_decoded3,
            global_feature=routed_global2,
        )

        prediction2 = self.pred2(decoded2)
        decoded2_reduced = self.reduce2(decoded2)

        stage1_feature = self.stage1_adapter(stage1)

        global1 = self.global1(
            stage4,
            target_size=stage1.shape[-2:],
        )

        (
            routed_stage1,
            routed_decoded2,
            routed_global1,
        ) = self.dictionary_router1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
            type_field=type_field,
        )

        decoded1 = self.fusion1(
            low_feature=routed_stage1,
            high_feature=routed_decoded2,
            global_feature=routed_global1,
        )

        stage2_boundary = self.stage2_boundary_adapter(
            stage2
        )

        decoded1 = self.boundary_refinement(
            shallow_feature=stage1_feature,
            semantic_feature=stage2_boundary,
            saliency_feature=decoded1,
        )

        coarse_prediction = self.pred1(
            decoded1
        )

        refined_prediction = self.disagreement_refinement(
            saliency_feature=decoded1,
            shallow_feature=stage1_feature,
            semantic_feature=stage2_boundary,
            coarse_logit=coarse_prediction,
        )

        refined_prediction = F.interpolate(
            refined_prediction,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": refined_prediction,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementRefinementRGBM60ResidualInputSOD:
    return (
        VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementRefinementRGBM60ResidualInputSOD(
            pretrained_path=PRETRAINED_PATH,
            num_prototypes=12,
            dictionary_dim=128,
            dictionary_temperature=0.2,
            routing_strength=0.5,
        )
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("VMamba smoke test requires CUDA.")

    device = torch.device("cuda")

    model = build_model().to(device)
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
        "patch input channels:",
        model.backbone.patch_embed[0].in_channels,
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
