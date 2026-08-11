# models/networks/mambavision_small_progressive_region_direct_dictionary_routing_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.multiscale_saliency_dictionary import (
    DictionaryStreamRouter,
    SharedLatentTypeDictionary,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    MambaVisionSmallProgressiveRegionDirectSOD,
    PRETRAINED_PATH,
)


class MambaVisionSmallProgressiveRegionDirectDictionaryRoutingSOD(
    MambaVisionSmallProgressiveRegionDirectSOD
):
    """
    Progressive + Direct Region Mean + Multi-scale Latent Dictionary Routing.

    The validated Direct Region hierarchy is kept unchanged:

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M40
        Stage4 <- M20

    The dictionary branch reads RAW backbone features:

        Raw S2
        Raw S3
        Raw S4

    and projects them into one shared latent type space.

    One dataset-level dictionary is shared by all three scales:

        D in R^(K x C)

    The resulting multi-scale latent-type field never reconstructs or
    overwrites the region-conditioned backbone hierarchy.

    Instead, it only controls the information routing of the existing
    Progressive decoder:

        low stream
        top-down stream
        persistent Stage4 global stream

    Therefore Region Mean and Dictionary have separate roles:

        Region Mean:
            image-specific region appearance reconstruction

        Dictionary:
            dataset-level shared latent-type recognition

        Progressive decoder:
            type-conditioned hierarchical information selection
    """

    input_keys = (
        MambaVisionSmallProgressiveRegionDirectSOD
        .input_keys
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
        )

        stage1_channels = (
            self.backbone.out_channels[0]
        )

        stage2_channels = (
            self.backbone.out_channels[1]
        )

        stage3_channels = (
            self.backbone.out_channels[2]
        )

        stage4_channels = (
            self.backbone.out_channels[3]
        )

        # -------------------------------------------------
        # Shared multi-scale latent dictionary
        # -------------------------------------------------

        self.latent_type_dictionary = (
            SharedLatentTypeDictionary(
                stage2_channels=stage2_channels,
                stage3_channels=stage3_channels,
                stage4_channels=stage4_channels,
                dictionary_dim=dictionary_dim,
                num_prototypes=num_prototypes,
                temperature=dictionary_temperature,
            )
        )

        # -------------------------------------------------
        # Dictionary-guided Progressive routing
        # -------------------------------------------------
        #
        # The original Progressive fusion blocks remain untouched.
        #
        # Dictionary routing happens immediately before each original
        # fusion block.
        # -------------------------------------------------

        self.dictionary_router3 = (
            DictionaryStreamRouter(
                channels=stage3_channels,
                num_prototypes=num_prototypes,
                routing_strength=routing_strength,
            )
        )

        self.dictionary_router2 = (
            DictionaryStreamRouter(
                channels=stage2_channels,
                num_prototypes=num_prototypes,
                routing_strength=routing_strength,
            )
        )

        self.dictionary_router1 = (
            DictionaryStreamRouter(
                channels=128,
                num_prototypes=num_prototypes,
                routing_strength=routing_strength,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_20: torch.Tensor,
        mean_40: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = (
            image.shape[-2:]
        )

        # -------------------------------------------------
        # Backbone
        # -------------------------------------------------

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image
        )

        # Keep raw backbone features for the dictionary branch.
        #
        # Region reconstruction below produces new tensors, therefore
        # these references remain the untouched backbone outputs.
        raw_stage2 = stage2
        raw_stage3 = stage3
        raw_stage4 = stage4

        # -------------------------------------------------
        # Shared latent-type dictionary
        # -------------------------------------------------
        #
        # Raw S2 / S3 / S4 all query the SAME dictionary.
        #
        # Output:
        #
        #   type_field:
        #       [B, K, H2, W2]
        #
        # This branch does not alter any backbone feature.
        # -------------------------------------------------

        type_field = (
            self.latent_type_dictionary(
                stage2=raw_stage2,
                stage3=raw_stage3,
                stage4=raw_stage4,
            )
        )

        # -------------------------------------------------
        # Direct region hierarchy
        # -------------------------------------------------

        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_hierarchy(
            image=image,
            mean_20=mean_20,
            mean_40=mean_40,
            mean_60=mean_60,
        )

        # -------------------------------------------------
        # Region pyramid
        # -------------------------------------------------

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
            stage1_size=(
                stage1.shape[-2:]
            ),
            stage2_size=(
                stage2.shape[-2:]
            ),
            stage3_size=(
                stage3.shape[-2:]
            ),
            stage4_size=(
                stage4.shape[-2:]
            ),
        )

        # -------------------------------------------------
        # Region-conditioned backbone reconstruction
        # -------------------------------------------------
        #
        # This part is identical to the current
        # Progressive + Direct Region Mean model.
        #
        # Dictionary never enters these modules.
        # -------------------------------------------------

        stage4 = (
            self.stage4_region_interaction(
                visual_feature=stage4,
                region_feature=region4,
            )
        )

        stage3 = (
            self.stage3_region_interaction(
                visual_feature=stage3,
                region_feature=region3,
            )
        )

        stage2 = (
            self.stage2_region_reconstruction(
                visual_feature=stage2,
                region_feature=region2,
            )
        )

        stage1 = (
            self.stage1_detail_reconstruction(
                visual_feature=stage1,
                detail_feature=region1,
            )
        )

        # -------------------------------------------------
        # Stage4
        # -------------------------------------------------
        #
        # Stage4 is intentionally untouched by Dictionary.
        # -------------------------------------------------

        decoded4 = self.deep_projection(
            stage4
        )

        decoded4 = self.context4(
            decoded4
        )

        prediction4 = self.pred4(
            decoded4
        )

        # -------------------------------------------------
        # Stage3
        # -------------------------------------------------

        global3 = self.global3(
            stage4,
            target_size=(
                stage3.shape[-2:]
            ),
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

        prediction3 = self.pred3(
            decoded3
        )

        decoded3_reduced = self.reduce3(
            decoded3
        )

        # -------------------------------------------------
        # Stage2
        # -------------------------------------------------

        global2 = self.global2(
            stage4,
            target_size=(
                stage2.shape[-2:]
            ),
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

        prediction2 = self.pred2(
            decoded2
        )

        decoded2_reduced = self.reduce2(
            decoded2
        )

        # -------------------------------------------------
        # Stage1
        # -------------------------------------------------

        stage1_feature = (
            self.stage1_adapter(
                stage1
            )
        )

        global1 = self.global1(
            stage4,
            target_size=(
                stage1.shape[-2:]
            ),
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

        # -------------------------------------------------
        # Boundary refinement
        # -------------------------------------------------
        #
        # Keep the original region-conditioned Stage1 / Stage2
        # features here. Dictionary routing does not enter the
        # boundary module.
        # -------------------------------------------------

        stage2_boundary = (
            self.stage2_boundary_adapter(
                stage2
            )
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=stage1_feature,
                semantic_feature=stage2_boundary,
                saliency_feature=decoded1,
            )
        )

        prediction1 = self.pred1(
            decoded1
        )

        prediction1 = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction1,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> (
    MambaVisionSmallProgressiveRegionDirectDictionaryRoutingSOD
):
    return (
        MambaVisionSmallProgressiveRegionDirectDictionaryRoutingSOD(
            pretrained_path=PRETRAINED_PATH,
            num_prototypes=12,
            dictionary_dim=128,
            dictionary_temperature=0.2,
            routing_strength=0.5,
        )
    )