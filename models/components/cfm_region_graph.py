"""Reusable NAMLab region-graph reasoning for dense visual features.

The module is deliberately task-agnostic:

    dense feature + integer region partition
        -> region nodes
        -> NAM adjacency graph
        -> graph attention message passing
        -> region-to-pixel broadcast
        -> reconstructed dense feature

NAMLab decides region membership and graph topology. Learned feature relations
decide the content and strength of messages passed over that topology.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    """Local copy to keep RegionGraphBlock independent from CFM/SACA modules."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class RegionGraphAttentionLayer(nn.Module):
    """Sparse multi-head graph attention over an explicit region graph."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

        hidden_channels = int(channels * mlp_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    @staticmethod
    def _edge_softmax(
        logits: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Softmax over incoming edges independently for every head."""
        edge_count, num_heads = logits.shape
        index = dst[:, None].expand(edge_count, num_heads)

        # Keep reduction numerically stable under AMP/BF16.
        logits_fp32 = logits.float()

        max_per_node = torch.full(
            (num_nodes, num_heads),
            -torch.inf,
            dtype=torch.float32,
            device=logits.device,
        )
        max_per_node.scatter_reduce_(
            dim=0,
            index=index,
            src=logits_fp32,
            reduce="amax",
            include_self=True,
        )

        exp_logits = torch.exp(logits_fp32 - max_per_node[dst])

        denominator = torch.zeros(
            (num_nodes, num_heads),
            dtype=torch.float32,
            device=logits.device,
        )
        denominator.index_add_(0, dst, exp_logits)

        return (
            exp_logits
            / denominator[dst].clamp_min(1e-6)
        ).to(dtype=logits.dtype)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            nodes: [K, C]
            edge_index: [2, E], source row then destination row.
        """
        residual = nodes
        x = self.norm1(nodes)
        num_nodes = x.shape[0]

        q = self.q(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        k = self.k(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        v = self.v(x).view(
            num_nodes,
            self.num_heads,
            self.head_dim,
        )

        src = edge_index[0]
        dst = edge_index[1]

        logits = (
            q[dst] * k[src]
        ).sum(dim=-1) * self.scale

        attention = self._edge_softmax(
            logits=logits,
            dst=dst,
            num_nodes=num_nodes,
        )

        # AMP can keep QK/softmax in float32 while V is bfloat16.
        # Message aggregation requires source and destination tensors to
        # share an exact dtype, so align attention to the value branch.
        attention = attention.to(dtype=v.dtype)
        messages = attention[..., None] * v[src]

        aggregated = torch.zeros(
            (
                num_nodes,
                self.num_heads,
                self.head_dim,
            ),
            dtype=v.dtype,
            device=v.device,
        )
        aggregated.index_add_(0, dst, messages)
        aggregated = aggregated.reshape(num_nodes, self.channels)

        nodes = residual + self.proj(aggregated)
        nodes = nodes + self.mlp(self.norm2(nodes))
        return nodes


class RegionGraphBlock(nn.Module):
    """Convert a dense feature map into a region-structured dense feature map."""

    def __init__(
        self,
        channels: int,
        graph_depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        if graph_depth < 1:
            raise ValueError("graph_depth must be >= 1")

        self.channels = channels
        self.graph_layers = nn.ModuleList(
            [
                RegionGraphAttentionLayer(
                    channels=channels,
                    num_heads=num_heads,
                )
                for _ in range(graph_depth)
            ]
        )

        # Intentionally no outer identity residual. Downstream consumers receive
        # a representation reconstructed jointly from local pixel features and
        # region-graph context, so region information has structural influence.
        self.fusion = nn.Sequential(
            ConvBNAct(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
            ),
        )

    @staticmethod
    def _compact_partition(
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap present IDs to contiguous 0..K-1 after nearest resize."""
        shape = labels.shape
        unique_ids, inverse = torch.unique(
            labels.reshape(-1),
            sorted=True,
            return_inverse=True,
        )
        return inverse.view(shape), unique_ids

    @staticmethod
    def _pool_nodes(
        feature: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool dense feature pixels into graph nodes."""
        channels = feature.shape[0]
        flat_labels = labels.reshape(-1)
        flat_feature = feature.permute(1, 2, 0).reshape(-1, channels)
        num_regions = int(flat_labels.max().item()) + 1

        # Accumulate in FP32 for stable region means under AMP/BF16.
        node_sum = torch.zeros(
            (num_regions, channels),
            dtype=torch.float32,
            device=feature.device,
        )
        node_sum.index_add_(0, flat_labels, flat_feature.float())

        counts = torch.zeros(
            (num_regions, 1),
            dtype=torch.float32,
            device=feature.device,
        )
        counts.index_add_(
            0,
            flat_labels,
            torch.ones(
                (flat_labels.numel(), 1),
                dtype=torch.float32,
                device=feature.device,
            ),
        )

        nodes = node_sum / counts.clamp_min(1.0)
        return nodes.to(dtype=feature.dtype)

    @staticmethod
    def _build_edges(
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Build undirected 4-neighbour region adjacency plus self-loops."""
        num_regions = int(labels.max().item()) + 1

        left = labels[:, :-1].reshape(-1)
        right = labels[:, 1:].reshape(-1)
        horizontal = torch.stack([left, right], dim=1)

        top = labels[:-1, :].reshape(-1)
        bottom = labels[1:, :].reshape(-1)
        vertical = torch.stack([top, bottom], dim=1)

        pairs = torch.cat([horizontal, vertical], dim=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

        if pairs.numel() > 0:
            pairs = torch.cat([pairs, pairs.flip(dims=[1])], dim=0)

        node_ids = torch.arange(
            num_regions,
            device=labels.device,
            dtype=torch.long,
        )
        self_edges = torch.stack([node_ids, node_ids], dim=1)

        pairs = torch.cat([pairs, self_edges], dim=0)
        pairs = torch.unique(pairs, dim=0)
        return pairs.transpose(0, 1).contiguous()

    @staticmethod
    def _broadcast_nodes(
        nodes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast each updated node feature to pixels in its region."""
        height, width = labels.shape
        channels = nodes.shape[1]
        dense = nodes[labels.reshape(-1)]
        return (
            dense.view(height, width, channels)
            .permute(2, 0, 1)
            .contiguous()
        )

    def _forward_single(
        self,
        feature: torch.Tensor,
        partition: torch.Tensor,
    ) -> torch.Tensor:
        labels, _ = self._compact_partition(partition)
        nodes = self._pool_nodes(feature, labels)
        edge_index = self._build_edges(labels)

        for layer in self.graph_layers:
            nodes = layer(nodes, edge_index)

        return self._broadcast_nodes(nodes, labels)

    def forward(
        self,
        feature: torch.Tensor,
        region_partition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature: [B, C, H, W]
            region_partition: [B, H0, W0] or [B, 1, H0, W0]
        """
        if region_partition.ndim == 4:
            if region_partition.shape[1] != 1:
                raise ValueError(
                    "4D region_partition must have exactly one channel"
                )
            region_partition = region_partition[:, 0]

        if region_partition.ndim != 3:
            raise ValueError(
                "region_partition must be [B,H,W] or [B,1,H,W]"
            )

        if region_partition.shape[0] != feature.shape[0]:
            raise ValueError(
                "feature and region_partition batch sizes must match"
            )

        labels = F.interpolate(
            region_partition.unsqueeze(1).float(),
            size=feature.shape[-2:],
            mode="nearest",
        )[:, 0].long()

        if labels.min().item() < 0:
            raise ValueError("region_partition contains negative IDs")

        contexts = [
            self._forward_single(
                feature=feature[batch_index],
                partition=labels[batch_index],
            )
            for batch_index in range(feature.shape[0])
        ]
        region_context = torch.stack(contexts, dim=0)

        return self.fusion(
            torch.cat([feature, region_context], dim=1)
        )
