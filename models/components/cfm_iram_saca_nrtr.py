"""NAM Region Token Reasoning (NRTR).

NAMLab is used as an adaptive perceptual partition operator, not as a dense
feature branch.

The partition now comes directly from official NAMLab MAT -> NPZ output:
    region_partition[y, x] = true NAM region ID

original CFMNet F2 + NAM partition
    -> soft region association
    -> region tokens (deep mean/std + region mean RGB + geometry)
    -> 2-layer global Region Transformer + adjacency bias

complete IRAM + SACA host F2 -> pixel queries
refined region tokens        -> keys / values
    -> soft pixel-to-region readout + NAM association bias
    -> zero-initialized residual correction of host F2

SACA itself is never modified.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import (
    ConvBNAct,
)


class NAMRegionTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=192,
        token_dim=128,
        max_regions=96,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.max_regions = max_regions

        self.semantic_projection = (
            nn.Sequential(
                ConvBNAct(
                    in_channels,
                    token_dim,
                    kernel_size=1,
                ),
                ConvBNAct(
                    token_dim,
                    token_dim,
                    kernel_size=3,
                    groups=token_dim,
                ),
            )
        )

        # deep mean + deep std + mean RGB + geometry
        descriptor_dim = (
            token_dim * 2
            + 3
            + 5
        )

        self.token_fusion = (
            nn.Sequential(
                nn.Linear(
                    descriptor_dim,
                    token_dim * 2,
                ),
                nn.GELU(),
                nn.Linear(
                    token_dim * 2,
                    token_dim,
                ),
                nn.LayerNorm(
                    token_dim
                ),
            )
        )

    def _build_association(
        self,
        region_partition,
        f1_size,
        f2_size,
    ):
        if region_partition.ndim == 4:
            if (
                region_partition.shape[1]
                != 1
            ):
                raise ValueError(
                    "region_partition 4D input "
                    "must have one channel"
                )

            region_partition = (
                region_partition[:, 0]
            )

        if region_partition.ndim != 3:
            raise ValueError(
                "region_partition must be "
                "[B,H,W] or [B,1,H,W]"
            )

        labels_f1 = (
            F.interpolate(
                region_partition
                .unsqueeze(1)
                .float(),
                size=f1_size,
                mode="nearest",
            )[:, 0]
            .long()
        )

        if (
            labels_f1.min().item()
            < 0
        ):
            raise ValueError(
                "region_partition contains "
                "negative IDs"
            )

        max_id = int(
            labels_f1.max().item()
        )

        if (
            max_id
            >= self.max_regions
        ):
            raise ValueError(
                f"Official NAM partition region id "
                f"{max_id} >= "
                f"max_regions={self.max_regions}. "
                "Expected fallback-M60 partitions "
                "to stay below this safety ceiling. "
                "Inspect the NPZ/MAT conversion."
            )

        # Dynamic K per batch. This keeps region-space computation genuinely
        # variable-cardinality while still allowing ordinary batched tensors.
        batch_region_count = (
            max_id + 1
        )

        one_hot_f1 = (
            F.one_hot(
                labels_f1,
                num_classes=(
                    batch_region_count
                ),
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .float()
        )

        association_map = (
            F.adaptive_avg_pool2d(
                one_hot_f1,
                output_size=f2_size,
            )
        )

        association_map = (
            association_map
            / association_map.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(
                1e-6
            )
        )

        association = (
            association_map
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        valid = (
            association.sum(
                dim=1
            )
            > 1e-6
        )

        return (
            labels_f1,
            one_hot_f1,
            association,
            valid,
        )

    @staticmethod
    def _build_adjacency(
        labels_f1,
        region_count,
    ):
        batch_size = (
            labels_f1.shape[0]
        )

        adjacency = torch.zeros(
            batch_size,
            region_count,
            region_count,
            dtype=torch.bool,
            device=labels_f1.device,
        )

        pairs = (
            (
                labels_f1[
                    :,
                    :,
                    :-1,
                ],
                labels_f1[
                    :,
                    :,
                    1:,
                ],
            ),
            (
                labels_f1[
                    :,
                    :-1,
                    :,
                ],
                labels_f1[
                    :,
                    1:,
                    :,
                ],
            ),
        )

        for first, second in pairs:
            different = (
                first != second
            )

            for batch_index in range(
                batch_size
            ):
                a = first[
                    batch_index
                ][different[batch_index]]

                b = second[
                    batch_index
                ][different[batch_index]]

                if a.numel() == 0:
                    continue

                flat = adjacency[
                    batch_index
                ].view(-1)

                flat[
                    (
                        a * region_count
                        + b
                    ).unique()
                ] = True

                flat[
                    (
                        b * region_count
                        + a
                    ).unique()
                ] = True

        return adjacency

    def forward(
        self,
        feature_f2,
        mean_60,
        region_partition,
        f1_size,
    ):
        batch_size, _, height, width = (
            feature_f2.shape
        )

        (
            labels_f1,
            one_hot_f1,
            association,
            valid,
        ) = self._build_association(
            region_partition,
            f1_size=f1_size,
            f2_size=(
                height,
                width,
            ),
        )

        region_count = (
            association.shape[-1]
        )

        adjacency = (
            self._build_adjacency(
                labels_f1,
                region_count=(
                    region_count
                ),
            )
        )

        x = (
            self.semantic_projection(
                feature_f2
            )
            .flatten(2)
            .transpose(1, 2)
            .float()
        )

        association_float = (
            association.float()
        )

        mass = (
            association_float
            .sum(
                dim=1
            )
            .clamp_min(
                1e-6
            )
        )

        semantic_mean = (
            torch.einsum(
                "bnk,bnd->bkd",
                association_float,
                x,
            )
            / mass.unsqueeze(-1)
        )

        semantic_second = (
            torch.einsum(
                "bnk,bnd->bkd",
                association_float,
                x.square(),
            )
            / mass.unsqueeze(-1)
        )

        semantic_std = torch.sqrt(
            (
                semantic_second
                - semantic_mean.square()
            ).clamp_min(
                0.0
            )
            + 1e-6
        )

        # Region-mean image is only used as an appearance descriptor here.
        # It is not passed through another dense evidence branch.
        mean_f1 = F.interpolate(
            mean_60.float(),
            size=f1_size,
            mode="nearest",
        )

        mean_flat = (
            mean_f1
            .flatten(2)
            .transpose(1, 2)
        )

        one_hot_flat = (
            one_hot_f1
            .flatten(2)
            .transpose(1, 2)
            .float()
        )

        mass_f1 = (
            one_hot_flat
            .sum(
                dim=1
            )
            .clamp_min(
                1e-6
            )
        )

        appearance = (
            torch.einsum(
                "bnk,bnd->bkd",
                one_hot_flat,
                mean_flat,
            )
            / mass_f1.unsqueeze(-1)
        )

        f1_height, f1_width = (
            f1_size
        )

        yy, xx = torch.meshgrid(
            torch.linspace(
                0,
                1,
                f1_height,
                device=(
                    feature_f2.device
                ),
            ),
            torch.linspace(
                0,
                1,
                f1_width,
                device=(
                    feature_f2.device
                ),
            ),
            indexing="ij",
        )

        coordinates = (
            torch.stack(
                [
                    xx,
                    yy,
                ],
                dim=-1,
            )
            .view(
                1,
                -1,
                2,
            )
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        center = (
            torch.einsum(
                "bnk,bnd->bkd",
                one_hot_flat,
                coordinates,
            )
            / mass_f1.unsqueeze(-1)
        )

        coordinate_second = (
            torch.einsum(
                "bnk,bnd->bkd",
                one_hot_flat,
                coordinates.square(),
            )
            / mass_f1.unsqueeze(-1)
        )

        coordinate_std = torch.sqrt(
            (
                coordinate_second
                - center.square()
            ).clamp_min(
                0.0
            )
            + 1e-6
        )

        area_ratio = (
            mass_f1
            / float(
                f1_height
                * f1_width
            )
        ).unsqueeze(-1)

        geometry = torch.cat(
            [
                area_ratio,
                center,
                coordinate_std,
            ],
            dim=-1,
        )

        descriptor = torch.cat(
            [
                semantic_mean,
                semantic_std,
                appearance,
                geometry,
            ],
            dim=-1,
        )

        tokens = (
            self.token_fusion(
                descriptor
            )
            * valid.unsqueeze(-1)
        )

        return {
            "tokens": tokens,
            "association": (
                association
            ),
            "valid": valid,
            "adjacency": adjacency,
        }


class RegionSelfAttention(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=4,
    ):
        super().__init__()

        if (
            dim % num_heads
            != 0
        ):
            raise ValueError(
                "dim must be divisible "
                "by num_heads"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (
            dim // num_heads
        )
        self.scale = (
            self.head_dim ** -0.5
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=True,
        )

        self.out = nn.Linear(
            dim,
            dim,
            bias=True,
        )

        self.adjacency_bias = (
            nn.Parameter(
                torch.ones(
                    num_heads
                )
            )
        )

    def forward(
        self,
        x,
        valid,
        adjacency,
    ):
        batch_size, region_count, _ = (
            x.shape
        )

        qkv = (
            self.qkv(
                x
            )
            .view(
                batch_size,
                region_count,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q, key, value = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        score = (
            torch.matmul(
                q,
                key.transpose(
                    -2,
                    -1,
                ),
            )
            * self.scale
        )

        score = (
            score
            + adjacency[
                :,
                None,
            ].to(
                score.dtype
            )
            * self.adjacency_bias[
                None,
                :,
                None,
                None,
            ].to(
                score.dtype
            )
        )

        score = score.masked_fill(
            ~valid[
                :,
                None,
                None,
                :,
            ],
            -1e4,
        )

        attention = torch.softmax(
            score.float(),
            dim=-1,
        ).to(
            q.dtype
        )

        output = (
            torch.matmul(
                attention,
                value,
            )
            .transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                region_count,
                self.dim,
            )
        )

        return (
            self.out(
                output
            )
            * valid.unsqueeze(-1)
        )


class RegionTransformerLayer(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=4,
        mlp_ratio=2,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attention = (
            RegionSelfAttention(
                dim,
                num_heads,
            )
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                dim,
                dim * mlp_ratio,
            ),
            nn.GELU(),
            nn.Linear(
                dim * mlp_ratio,
                dim,
            ),
        )

    def forward(
        self,
        x,
        valid,
        adjacency,
    ):
        x = (
            x
            + self.attention(
                self.norm1(
                    x
                ),
                valid=valid,
                adjacency=(
                    adjacency
                ),
            )
        )

        x = (
            x
            + self.ffn(
                self.norm2(
                    x
                )
            )
            * valid.unsqueeze(-1)
        )

        return (
            x
            * valid.unsqueeze(-1)
        )


class RegionTransformer(nn.Module):
    def __init__(
        self,
        dim=128,
        depth=2,
        num_heads=4,
    ):
        super().__init__()

        self.layers = (
            nn.ModuleList(
                [
                    RegionTransformerLayer(
                        dim,
                        num_heads,
                        2,
                    )
                    for _ in range(
                        depth
                    )
                ]
            )
        )

        self.norm = nn.LayerNorm(
            dim
        )

    def forward(
        self,
        x,
        valid,
        adjacency,
    ):
        for layer in self.layers:
            x = layer(
                x,
                valid=valid,
                adjacency=adjacency,
            )

        return (
            self.norm(
                x
            )
            * valid.unsqueeze(-1)
        )


class NAMRegionTokenReasoning(nn.Module):
    def __init__(
        self,
        feature_channels=192,
        token_dim=128,
        max_regions=96,
        depth=2,
        num_heads=4,
    ):
        super().__init__()

        self.token_dim = token_dim

        self.tokenizer = (
            NAMRegionTokenizer(
                feature_channels,
                token_dim,
                max_regions,
            )
        )

        self.region_transformer = (
            RegionTransformer(
                token_dim,
                depth,
                num_heads,
            )
        )

        self.region_head = nn.Linear(
            token_dim,
            1,
        )

        self.query = nn.Conv2d(
            feature_channels,
            token_dim,
            kernel_size=1,
            bias=False,
        )

        self.key = nn.Linear(
            token_dim,
            token_dim,
            bias=False,
        )

        self.value = nn.Linear(
            token_dim,
            token_dim,
            bias=False,
        )

        # NAM ownership is a soft prior, not a hard constraint.
        self.association_bias = (
            nn.Parameter(
                torch.tensor(
                    1.0
                )
            )
        )

        self.context_refine = (
            nn.Sequential(
                ConvBNAct(
                    token_dim,
                    token_dim,
                    kernel_size=3,
                    groups=token_dim,
                ),
                ConvBNAct(
                    token_dim,
                    token_dim,
                    kernel_size=1,
                ),
            )
        )

        self.delta = nn.Conv2d(
            token_dim,
            feature_channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(
            self.delta.weight
        )

        nn.init.zeros_(
            self.delta.bias
        )

    def forward(
        self,
        original_f2,
        host_f2,
        mean_60,
        region_partition,
        f1_size,
    ):
        data = self.tokenizer(
            feature_f2=original_f2,
            mean_60=mean_60,
            region_partition=(
                region_partition
            ),
            f1_size=f1_size,
        )

        tokens = (
            self.region_transformer(
                data["tokens"],
                valid=data["valid"],
                adjacency=(
                    data["adjacency"]
                ),
            )
        )

        region_logits = (
            self.region_head(
                tokens
            )[..., 0]
        )

        query = self.query(
            host_f2
        )

        batch_size, _, height, width = (
            query.shape
        )

        query = (
            query
            .flatten(2)
            .transpose(1, 2)
        )

        key = self.key(
            tokens
        )

        value = self.value(
            tokens
        )

        score = torch.einsum(
            "bnd,bkd->bnk",
            query,
            key,
        ) / math.sqrt(
            float(
                self.token_dim
            )
        )

        score = (
            score
            + self.association_bias.to(
                score.dtype
            )
            * data[
                "association"
            ].to(
                score.dtype
            )
        )

        score = score.masked_fill(
            ~data["valid"][
                :,
                None,
                :,
            ],
            -1e4,
        )

        readout = torch.softmax(
            score.float(),
            dim=-1,
        ).to(
            value.dtype
        )

        context = torch.einsum(
            "bnk,bkd->bnd",
            readout,
            value,
        )

        context = (
            context
            .transpose(1, 2)
            .reshape(
                batch_size,
                self.token_dim,
                height,
                width,
            )
        )

        context = (
            self.context_refine(
                context
            )
        )

        output = (
            host_f2
            + self.delta(
                context
            )
        )

        return output, {
            "region_logits": (
                region_logits
            ),
            "region_assoc": (
                data["association"]
            ),
            "region_valid": (
                data["valid"]
            ),
        }
