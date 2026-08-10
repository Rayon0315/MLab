# models/networks/mambavision_small_progressive_region_direct_dictionary_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.components.saliency_dictionary import (
    GlobalMetaTypeDictionary,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    MambaVisionSmallProgressiveRegionDirectSOD,
    PRETRAINED_PATH,
)


class Stage4RegionDictionaryInteraction(nn.Module):
    """
    Keep the original Stage4 region interaction and append a global
    latent meta-type dictionary immediately after it.

    Information flow:

        Stage4 visual feature
                +
        Stage4 M20 region feature
                |
                v
        Original bidirectional region-visual interaction
                |
                v
        Region-enhanced Stage4
                |
                v
        Global latent meta-type dictionary
                |
                v
        Dictionary-enhanced Stage4
    """

    def __init__(
        self,
        region_interaction: nn.Module,
        channels: int,
        num_prototypes: int = 8,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.region_interaction = region_interaction

        self.dictionary = GlobalMetaTypeDictionary(
            channels=channels,
            num_prototypes=num_prototypes,
            temperature=temperature,
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.region_interaction(
            visual_feature=visual_feature,
            region_feature=region_feature,
        )

        feature = self.dictionary(
            feature
        )

        return feature


class MambaVisionSmallProgressiveRegionDirectDictionarySOD(
    MambaVisionSmallProgressiveRegionDirectSOD
):
    """
    Progressive + Direct Region Mean + Global Meta-Type Dictionary.

    The original Progressive + Direct Region Mean network is preserved.

    The only structural change is:

        Stage4 region-conditioned feature
            -> global latent meta-type dictionary
            -> original Stage4 decoder/global-semantic paths

    Dictionary properties:

        - dataset-level global learnable prototypes
        - K latent meta-types
        - no predefined foreground/background prototype split
        - no semantic class labels
        - no GT-based prototype generation
        - no additional dictionary loss
        - soft feature-to-prototype assignment
        - prototype-to-feature reconstruction
        - residual latent reconstruction
        - identical train/test inference path

    All other components remain inherited from:

        mambavision_small_progressive_region_direct_sod.py
    """

    input_keys = (
        MambaVisionSmallProgressiveRegionDirectSOD
        .input_keys
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        num_prototypes: int = 8,
        dictionary_temperature: float = 0.1,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
        )

        stage4_channels = (
            self.backbone.out_channels[3]
        )

        # -------------------------------------------------
        # Controlled Stage4 dictionary insertion
        # -------------------------------------------------
        #
        # Original:
        #
        #   Stage4
        #       -> Stage4 region interaction
        #       -> deep_projection / global3 / global2 / global1
        #
        # New:
        #
        #   Stage4
        #       -> Stage4 region interaction
        #       -> global meta-type dictionary
        #       -> deep_projection / global3 / global2 / global1
        #
        # Everything else remains inherited from the current
        # Progressive + Direct Region Mean network.
        # -------------------------------------------------

        self.stage4_region_interaction = (
            Stage4RegionDictionaryInteraction(
                region_interaction=(
                    self.stage4_region_interaction
                ),
                channels=stage4_channels,
                num_prototypes=num_prototypes,
                temperature=(
                    dictionary_temperature
                ),
            )
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectDictionarySOD:
    return (
        MambaVisionSmallProgressiveRegionDirectDictionarySOD(
            pretrained_path=PRETRAINED_PATH,
            num_prototypes=8,
            dictionary_temperature=0.1,
        )
    )