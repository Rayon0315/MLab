# models/components/saliency_dictionary.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class GlobalMetaTypeDictionary(nn.Module):
    """
    Dataset-level latent meta-type dictionary.

    The dictionary contains K learnable prototypes in the same latent
    space as the input feature. Each spatial token softly assigns to
    the prototypes, reconstructs a dictionary representation, and the
    reconstruction is fused back into the original feature through a
    zero-initialized residual branch.

    No GT mask is used inside this module. The prototypes are learned
    only from the existing SOD objective through backpropagation.
    """

    def __init__(
        self,
        channels: int,
        num_prototypes: int = 8,
        temperature: float = 0.1,
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

        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        # Dataset-level learnable latent meta-types.
        self.prototypes = nn.Parameter(
            torch.empty(
                num_prototypes,
                channels,
            )
        )

        # Normalize the Stage4 latent space before prototype matching.
        self.feature_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )

        # Normalize every prototype while preserving a learnable
        # channel-wise affine transformation.
        self.prototype_norm = nn.LayerNorm(
            channels
        )

        # Representation reconstruction:
        #
        #   normalized feature
        #   prototype reconstruction
        #   reconstruction error
        #
        # are jointly used to construct the dictionary residual.
        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        self.output_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self._init_parameters()

    def _init_parameters(
        self,
    ) -> None:
        nn.init.trunc_normal_(
            self.prototypes,
            std=0.02,
        )

        # Start from the exact original network behavior.
        #
        # At initialization:
        #
        #   output = feature + 0
        #
        # so the dictionary branch does not randomly damage the
        # already validated Progressive + Direct Region structure.
        nn.init.zeros_(
            self.output_projection.weight
        )

        if self.output_projection.bias is not None:
            nn.init.zeros_(
                self.output_projection.bias
            )

    def _compute_assignment(
        self,
        feature_tokens: torch.Tensor,
        dictionary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_tokens:
                [B, N, C]

            dictionary:
                [K, C]

        Returns:
            assignment:
                [B, N, K]
        """

        query = F.normalize(
            feature_tokens,
            p=2,
            dim=-1,
        )

        key = F.normalize(
            dictionary,
            p=2,
            dim=-1,
        )

        similarity = torch.matmul(
            query,
            key.transpose(0, 1),
        )

        similarity = (
            similarity
            / self.temperature
        )

        assignment = torch.softmax(
            similarity,
            dim=-1,
        )

        return assignment

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature:
                Stage4 region-enhanced feature
                [B, C, H, W]

        Returns:
            dictionary-enhanced Stage4 feature
                [B, C, H, W]
        """

        (
            batch_size,
            channels,
            height,
            width,
        ) = feature.shape

        if channels != self.channels:
            raise ValueError(
                "Dictionary channel mismatch: "
                f"expected {self.channels}, got {channels}."
            )

        # -------------------------------------------------
        # Normalize Stage4 latent representation
        # -------------------------------------------------

        normalized_feature = self.feature_norm(
            feature
        )

        # [B, C, H, W]
        # ->
        # [B, HW, C]
        feature_tokens = (
            normalized_feature
            .flatten(2)
            .transpose(1, 2)
        )

        # -------------------------------------------------
        # Global latent meta-type dictionary
        # -------------------------------------------------

        # [K, C]
        dictionary = self.prototype_norm(
            self.prototypes
        )

        # -------------------------------------------------
        # Feature-to-prototype assignment
        # -------------------------------------------------

        # [B, HW, K]
        assignment = self._compute_assignment(
            feature_tokens=feature_tokens,
            dictionary=dictionary,
        )

        # -------------------------------------------------
        # Prototype-to-feature reconstruction
        # -------------------------------------------------

        # [B, HW, K] @ [K, C]
        # ->
        # [B, HW, C]
        reconstruction_tokens = torch.matmul(
            assignment,
            dictionary,
        )

        # [B, HW, C]
        # ->
        # [B, C, H, W]
        reconstruction = (
            reconstruction_tokens
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        # -------------------------------------------------
        # Reconstruction discrepancy
        # -------------------------------------------------

        reconstruction_error = (
            normalized_feature
            - reconstruction
        )

        # -------------------------------------------------
        # Dictionary-conditioned latent reconstruction
        # -------------------------------------------------

        fused = torch.cat(
            [
                normalized_feature,
                reconstruction,
                reconstruction_error,
            ],
            dim=1,
        )

        residual = self.fusion(
            fused
        )

        residual = self.output_projection(
            residual
        )

        return feature + residual