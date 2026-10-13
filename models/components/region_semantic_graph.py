from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionSemanticGraph(nn.Module):
    """NAM-guided Stage3 region graph with Stage4 global conditioning.

    NAMLab supplies only the region partition. Stage3 supplies region semantics,
    Stage4 supplies global context, and graph attention connects 1/2-hop NAM
    regions before a gated dense re-injection into Stage3.
    """

    def __init__(
        self,
        stage3_channels: int,
        stage4_channels: int,
        token_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        initial_reinject_scale: float = 0.1,
    ) -> None:
        super().__init__()

        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.stage3_channels = int(stage3_channels)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.attention_scale = self.head_dim ** -0.5

        self.region_projection = nn.Sequential(
            nn.Linear(self.stage3_channels * 2, self.token_dim, bias=False),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(stage4_channels, self.token_dim, bias=False),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
        )

        self.attention_norm = nn.LayerNorm(self.token_dim)
        self.query_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.key_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.global_query = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.global_key = nn.Linear(self.token_dim, self.token_dim, bias=False)

        self.position_bias = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, self.num_heads),
        )
        self.message_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)

        hidden_dim = max(self.token_dim, int(self.token_dim * ffn_ratio))
        self.ffn_norm = nn.LayerNorm(self.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

        self.context_projection = nn.Linear(
            self.token_dim,
            self.stage3_channels,
            bias=False,
        )
        self.reinject_gate = nn.Conv2d(
            self.stage3_channels * 2,
            self.stage3_channels,
            kernel_size=1,
            bias=True,
        )
        self.reinject_scale = nn.Parameter(
            torch.tensor(float(initial_reinject_scale))
        )

    @staticmethod
    def _resize_partition(
        region_partition: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if region_partition.ndim == 4:
            if region_partition.shape[1] != 1:
                raise ValueError(
                    "4D region_partition must have a singleton channel dimension"
                )
            region_partition = region_partition[:, 0]
        if region_partition.ndim != 3:
            raise ValueError(
                "region_partition must be [B,H,W] or [B,1,H,W], "
                f"got {tuple(region_partition.shape)}"
            )

        if tuple(region_partition.shape[-2:]) == tuple(target_size):
            return region_partition.long()

        resized = F.interpolate(
            region_partition.unsqueeze(1).float(),
            size=target_size,
            mode="nearest",
        )
        return resized[:, 0].long()

    @staticmethod
    def build_candidate_mask(
        compact_labels: torch.Tensor,
        num_regions: int,
    ) -> torch.Tensor:
        """Return symmetric 1-hop + 2-hop NAM-region adjacency without self edges."""
        if compact_labels.ndim != 2:
            raise ValueError("compact_labels must be a 2D region-id map")

        device = compact_labels.device
        one_hop = torch.zeros(
            num_regions,
            num_regions,
            dtype=torch.bool,
            device=device,
        )

        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if compact_labels.shape[1] > 1:
            pairs.append(
                (compact_labels[:, :-1].reshape(-1), compact_labels[:, 1:].reshape(-1))
            )
        if compact_labels.shape[0] > 1:
            pairs.append(
                (compact_labels[:-1, :].reshape(-1), compact_labels[1:, :].reshape(-1))
            )

        for left, right in pairs:
            different = left != right
            if different.any():
                left = left[different]
                right = right[different]
                one_hop[left, right] = True
                one_hop[right, left] = True

        two_hop = (
            one_hop.to(torch.float32)
            @ one_hop.to(torch.float32)
        ) > 0

        candidate = one_hop | two_hop
        candidate.fill_diagonal_(False)
        return candidate

    @staticmethod
    def _pool_regions(
        feature: torch.Tensor,
        compact_labels: torch.Tensor,
        num_regions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Avg/max pool Stage3 pixels and return normalized region centroids."""
        channels, height, width = feature.shape
        flat_feature = feature.permute(1, 2, 0).reshape(-1, channels)
        flat_labels = compact_labels.reshape(-1)

        counts = torch.bincount(
            flat_labels,
            minlength=num_regions,
        ).to(dtype=flat_feature.dtype).clamp_min(1.0)

        summed = flat_feature.new_zeros((num_regions, channels))
        summed.index_add_(0, flat_labels, flat_feature)
        average = summed / counts.unsqueeze(1)

        max_values = flat_feature.new_full(
            (num_regions, channels),
            torch.finfo(flat_feature.dtype).min,
        )
        expanded_labels = flat_labels.unsqueeze(1).expand(-1, channels)
        max_values.scatter_reduce_(
            0,
            expanded_labels,
            flat_feature,
            reduce="amax",
            include_self=True,
        )

        y = torch.linspace(
            -1.0,
            1.0,
            steps=height,
            device=feature.device,
            dtype=feature.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            steps=width,
            device=feature.device,
            dtype=feature.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        centroid_sums = feature.new_zeros((num_regions, 2))
        centroid_sums.index_add_(0, flat_labels, coordinates)
        centroids = centroid_sums / counts.unsqueeze(1)

        return torch.cat((average, max_values), dim=1), centroids

    def _reason_single(
        self,
        stage3: torch.Tensor,
        stage4_global: torch.Tensor,
        compact_labels: torch.Tensor,
        num_regions: int,
    ) -> torch.Tensor:
        pooled, centroids = self._pool_regions(
            stage3,
            compact_labels,
            num_regions,
        )
        tokens = self.region_projection(pooled)
        global_token = self.global_projection(stage4_global)

        candidate = self.build_candidate_mask(
            compact_labels,
            num_regions=num_regions,
        )

        normalized = self.attention_norm(tokens)
        q = self.query_projection(normalized)
        k = self.key_projection(normalized)
        v = self.value_projection(normalized)

        q = q + self.global_query(global_token).unsqueeze(0)
        k = k + self.global_key(global_token).unsqueeze(0)

        q = q.reshape(num_regions, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(num_regions, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(num_regions, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale

        relative = centroids.unsqueeze(1) - centroids.unsqueeze(0)
        position_bias = self.position_bias(relative).permute(2, 0, 1)
        scores = scores + position_bias

        mask = candidate.unsqueeze(0)
        scores = scores.masked_fill(~mask, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        message = torch.matmul(attention, v)
        message = message.transpose(0, 1).reshape(num_regions, self.token_dim)
        tokens = tokens + self.message_projection(message)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))

        dense_context = self.context_projection(tokens)[compact_labels]
        dense_context = dense_context.permute(2, 0, 1).contiguous()
        gate = torch.sigmoid(
            self.reinject_gate(
                torch.cat((stage3, dense_context), dim=0).unsqueeze(0)
            )[0]
        )

        return stage3 + self.reinject_scale * gate * dense_context

    def forward(
        self,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
        region_partition: torch.Tensor,
    ) -> torch.Tensor:
        if stage3.ndim != 4 or stage4.ndim != 4:
            raise ValueError("stage3 and stage4 must be BCHW tensors")
        if stage3.shape[0] != stage4.shape[0]:
            raise ValueError("stage3/stage4 batch sizes must match")

        resized_partition = self._resize_partition(
            region_partition,
            target_size=stage3.shape[-2:],
        )
        if resized_partition.shape[0] != stage3.shape[0]:
            raise ValueError("partition batch size must match feature batch size")

        stage4_global = F.adaptive_avg_pool2d(stage4, 1).flatten(1)
        outputs: list[torch.Tensor] = []

        for batch_index in range(stage3.shape[0]):
            labels = resized_partition[batch_index]
            _, inverse = torch.unique(
                labels,
                sorted=True,
                return_inverse=True,
            )
            compact_labels = inverse.reshape_as(labels)
            num_regions = int(compact_labels.max().item()) + 1

            outputs.append(
                self._reason_single(
                    stage3=stage3[batch_index],
                    stage4_global=stage4_global[batch_index],
                    compact_labels=compact_labels,
                    num_regions=num_regions,
                )
            )

        return torch.stack(outputs, dim=0)
