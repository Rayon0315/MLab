# models/components/multiscale_saliency_dictionary.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
)


class SharedLatentTypeDictionary(nn.Module):
    """
    Shared dataset-level latent type dictionary.

    Raw backbone Stage2 / Stage3 / Stage4 features are projected into
    the same latent space and matched against one shared prototype bank.

    The dictionary does not reconstruct or modify backbone features.
    It only produces a multi-scale latent-type assignment field.

    Information flow:

        S2 -> projection -> assignment A2
        S3 -> projection -> assignment A3
        S4 -> projection -> assignment A4

                         shared dictionary D

        A2 + up(A3) + up(A4)
                    |
                    v
          shared latent-type field

    Each spatial location contains a probability distribution over K
    latent visual types.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        dictionary_dim: int = 128,
        num_prototypes: int = 12,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()

        if num_prototypes < 2:
            raise ValueError(
                "num_prototypes must be at least 2."
            )

        if temperature <= 0.0:
            raise ValueError(
                "temperature must be positive."
            )

        self.dictionary_dim = dictionary_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self.stage2_projection = ConvNormAct(
            stage2_channels,
            dictionary_dim,
            kernel_size=1,
            padding=0,
        )

        self.stage3_projection = ConvNormAct(
            stage3_channels,
            dictionary_dim,
            kernel_size=1,
            padding=0,
        )

        self.stage4_projection = ConvNormAct(
            stage4_channels,
            dictionary_dim,
            kernel_size=1,
            padding=0,
        )

        # One shared dataset-level latent dictionary.
        #
        # The prototypes have no predefined meanings such as:
        #
        #   foreground
        #   background
        #   airplane
        #   ship
        #
        # Their latent meanings are learned only from the SOD objective.
        self.prototypes = nn.Parameter(
            torch.empty(
                num_prototypes,
                dictionary_dim,
            )
        )

        self._init_parameters()

    def _init_parameters(
        self,
    ) -> None:
        nn.init.trunc_normal_(
            self.prototypes,
            std=0.02,
        )

    def _assign(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match a feature map to the shared prototype dictionary.

        Args:
            feature:
                [B, C, H, W]

        Returns:
            assignment:
                [B, K, H, W]
        """

        feature = F.normalize(
            feature,
            p=2,
            dim=1,
        )

        dictionary = F.normalize(
            self.prototypes,
            p=2,
            dim=1,
        )

        similarity = torch.einsum(
            "bchw,kc->bkhw",
            feature,
            dictionary,
        )

        similarity = (
            similarity
            / self.temperature
        )

        assignment = torch.softmax(
            similarity,
            dim=1,
        )

        return assignment

    def forward(
        self,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stage2:
                Raw Stage2 backbone feature.

            stage3:
                Raw Stage3 backbone feature.

            stage4:
                Raw Stage4 backbone feature.

        Returns:
            shared_type_field:
                [B, K, H2, W2]
        """

        stage2_latent = (
            self.stage2_projection(
                stage2
            )
        )

        stage3_latent = (
            self.stage3_projection(
                stage3
            )
        )

        stage4_latent = (
            self.stage4_projection(
                stage4
            )
        )

        assignment2 = self._assign(
            stage2_latent
        )

        assignment3 = self._assign(
            stage3_latent
        )

        assignment4 = self._assign(
            stage4_latent
        )

        target_size = (
            assignment2.shape[-2:]
        )

        assignment3 = F.interpolate(
            assignment3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        assignment4 = F.interpolate(
            assignment4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # All three assignments live in the same K-dimensional
        # categorical space because they query the same prototype bank.
        #
        # Equal averaging intentionally avoids introducing another
        # learnable scale-selection module in this first experiment.
        shared_type_field = (
            assignment2
            + assignment3
            + assignment4
        ) / 3.0

        return shared_type_field


class DictionaryStreamRouter(nn.Module):
    """
    Use the latent-type field only to route existing decoder streams.

    The router does not generate new semantic features and does not
    add dictionary vectors into the region-conditioned hierarchy.

    It independently scales:

        low-level stream
        top-down stream
        persistent global stream

    based on the current latent-type distribution.

    All routing convolutions are zero-initialized, therefore:

        initial routing weight = 1

    and the module initially preserves the original Progressive fusion
    behavior exactly.
    """

    def __init__(
        self,
        channels: int,
        num_prototypes: int,
        routing_strength: float = 0.5,
    ) -> None:
        super().__init__()

        self.routing_strength = routing_strength

        self.low_router = nn.Conv2d(
            num_prototypes,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.high_router = nn.Conv2d(
            num_prototypes,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.global_router = nn.Conv2d(
            num_prototypes,
            channels,
            kernel_size=1,
            bias=True,
        )

        self._init_routers()

    def _init_routers(
        self,
    ) -> None:
        for router in (
            self.low_router,
            self.high_router,
            self.global_router,
        ):
            nn.init.zeros_(
                router.weight
            )

            if router.bias is not None:
                nn.init.zeros_(
                    router.bias
                )

    def _route(
        self,
        feature: torch.Tensor,
        type_field: torch.Tensor,
        router: nn.Conv2d,
    ) -> torch.Tensor:
        type_field = F.interpolate(
            type_field,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        routing_logits = router(
            type_field
        )

        # Centered at 1.0:
        #
        #   logits = 0 -> scale = 1
        #
        # With routing_strength = 0.5:
        #
        #   scale is approximately within [0.5, 1.5].
        routing_scale = (
            1.0
            + self.routing_strength
            * torch.tanh(
                routing_logits
            )
        )

        return (
            feature
            * routing_scale
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        global_feature: torch.Tensor,
        type_field: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        low_feature = self._route(
            feature=low_feature,
            type_field=type_field,
            router=self.low_router,
        )

        high_feature = self._route(
            feature=high_feature,
            type_field=type_field,
            router=self.high_router,
        )

        global_feature = self._route(
            feature=global_feature,
            type_field=type_field,
            router=self.global_router,
        )

        return (
            low_feature,
            high_feature,
            global_feature,
        )