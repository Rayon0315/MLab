from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.mambacvc_blocks import ConvBNReLU


class PixelLayerNorm(nn.Module):
    """LayerNorm over channels independently at every spatial location."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.permute(0, 2, 3, 1)
        feature = F.layer_norm(
            feature,
            (feature.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return feature.permute(0, 3, 1, 2).contiguous()


class RegionMeanPreservingEncoder(nn.Module):
    """Encode NAM M60 while preserving its piecewise region structure.

    This mirrors the validated NAMLab Hybrid region encoder idea:
      nearest resize -> 1x1 projection -> channel-only residual refinement
      + lightweight depthwise local context.

    Input mean_60 is the RGB region-mean image in [0, 1].
    """

    def __init__(
        self,
        channels: int = 32,
        bottleneck_ratio: float = 0.5,
        context_scale: float = 0.1,
    ) -> None:
        super().__init__()

        hidden = max(16, int(channels * bottleneck_ratio))

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )

        self.input_projection = nn.Conv2d(3, channels, kernel_size=1, bias=True)
        self.preserve_norm = PixelLayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        # Preserve the stable initialization used by the validated Hybrid encoder.
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)

        self.context_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.context_norm = PixelLayerNorm(channels)
        self.context_activation = nn.GELU()
        self.context_projection = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale)))

    def forward(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if mean_60.ndim != 4 or mean_60.shape[1] != 3:
            raise ValueError(
                "mean_60 must be an RGB region-mean tensor [B,3,H,W]"
            )

        mean_60 = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )
        mean_60 = (mean_60 - self.image_mean) / self.image_std

        base = self.input_projection(mean_60)
        preserve = base + self.channel_mlp(self.preserve_norm(base))

        context = self.context_depthwise(preserve)
        context = self.context_norm(context)
        context = self.context_activation(context)
        context = self.context_projection(context)

        return preserve + self.context_scale * context


class LocalBidirectionalRegionVisualInteraction(nn.Module):
    """Bidirectional visual<->region cross attention inside a local window.

    The two streams interact symmetrically:
      Visual query  attends to Region key/value  -> visual_from_region
      Region query  attends to Visual key/value  -> region_from_visual

    NAM region-mean RGB differences are injected as a structural attention bias:
      bias(p, q) = -lambda * mean(|M60(p) - M60(q)|)

    Unlike the original global NAMLab interaction, this module never builds an
    HW x HW matrix. Complexity is O(HW * window_size^2).
    """

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()

        if inner_channels % num_heads != 0:
            raise ValueError("inner_channels must be divisible by num_heads")
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")

        self.channels = channels
        self.inner_channels = inner_channels
        self.num_heads = num_heads
        self.head_channels = inner_channels // num_heads
        self.window_size = window_size
        self.window_area = window_size * window_size
        self.padding = window_size // 2
        self.scale = self.head_channels ** -0.5

        # Pixel-wise channel normalization keeps the guidance strictly local:
        # changing a distant pixel cannot alter the query/key at this location.
        self.visual_norm = PixelLayerNorm(channels)
        self.region_norm = PixelLayerNorm(channels)

        self.visual_query = nn.Conv2d(channels, inner_channels, 1, bias=False)
        self.visual_key = nn.Conv2d(channels, inner_channels, 1, bias=False)
        self.visual_value = nn.Conv2d(channels, inner_channels, 1, bias=False)

        self.region_query = nn.Conv2d(channels, inner_channels, 1, bias=False)
        self.region_key = nn.Conv2d(channels, inner_channels, 1, bias=False)
        self.region_value = nn.Conv2d(channels, inner_channels, 1, bias=False)

        self.visual_from_region_projection = ConvBNReLU(
            inner_channels, channels, 1
        )
        self.region_from_visual_projection = ConvBNReLU(
            inner_channels, channels, 1
        )

        # log(lambda), initialized to lambda=1.0; exp keeps the bias strength > 0.
        self.region_bias_log_scale = nn.Parameter(torch.tensor(0.0))

        # Strong reconstruction: no explicit identity bypass after this fusion.
        self.reconstruction = nn.Sequential(
            ConvBNReLU(channels * 4, channels, 1),
            ConvBNReLU(channels, channels, 3),
        )

    def _reshape_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        return feature.reshape(
            b, self.num_heads, self.head_channels, h * w
        )

    def _extract_patches(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        patches = F.unfold(
            feature,
            kernel_size=self.window_size,
            padding=self.padding,
        )
        return patches.reshape(
            b,
            self.num_heads,
            self.head_channels,
            self.window_area,
            h * w,
        )

    def _valid_window_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ones = torch.ones(
            batch_size, 1, height, width, device=device, dtype=dtype
        )
        valid = F.unfold(
            ones,
            kernel_size=self.window_size,
            padding=self.padding,
        )
        valid = valid.reshape(
            batch_size, 1, self.window_area, height * width
        )
        return valid > 0.5

    def _region_similarity_bias(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        mean = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )
        b, c, h, w = mean.shape
        if c != 3:
            raise ValueError("mean_60 must contain RGB region means")

        center = mean.reshape(b, c, 1, h * w)
        patches = F.unfold(
            mean,
            kernel_size=self.window_size,
            padding=self.padding,
        ).reshape(b, c, self.window_area, h * w)

        difference = (patches - center).abs().mean(dim=1, keepdim=True)
        strength = torch.exp(self.region_bias_log_scale)
        return -strength * difference

    def _local_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = query.shape
        q = self._reshape_tokens(query)  # [B, heads, d, L]
        k = self._extract_patches(key)   # [B, heads, d, K, L]
        v = self._extract_patches(value) # [B, heads, d, K, L]

        scores = (q.unsqueeze(3) * k).sum(dim=2) * self.scale
        scores = scores + self._region_similarity_bias(
            mean_60,
            target_size=(h, w),
        )

        valid = self._valid_window_mask(
            b,
            h,
            w,
            device=query.device,
            dtype=query.dtype,
        )
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=2)

        out = (attention.unsqueeze(2) * v).sum(dim=3)
        return out.reshape(b, self.inner_channels, h, w)

    def compute_cross_updates(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_feature.shape != region_feature.shape:
            raise ValueError("visual and region features must have equal shape")

        visual = self.visual_norm(visual_feature)
        region = self.region_norm(region_feature)

        visual_from_region = self._local_cross_attention(
            query=self.visual_query(visual),
            key=self.region_key(region),
            value=self.region_value(region),
            mean_60=mean_60,
        )
        region_from_visual = self._local_cross_attention(
            query=self.region_query(region),
            key=self.visual_key(visual),
            value=self.visual_value(visual),
            mean_60=mean_60,
        )

        visual_from_region = self.visual_from_region_projection(
            visual_from_region
        )
        region_from_visual = self.region_from_visual_projection(
            region_from_visual
        )
        return visual_from_region, region_from_visual

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        visual_from_region, region_from_visual = self.compute_cross_updates(
            visual_feature,
            region_feature,
            mean_60,
        )

        fused = torch.cat(
            [
                visual_feature,
                region_feature,
                visual_from_region,
                region_from_visual,
            ],
            dim=1,
        )
        return self.reconstruction(fused)


class RegionMeanLocalCrossGuidance(nn.Module):
    """One unified GC->LC NAM region-mean guidance block."""

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.region_encoder = RegionMeanPreservingEncoder(channels=channels)
        self.interaction = LocalBidirectionalRegionVisualInteraction(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(
        self,
        feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        region = self.region_encoder(
            mean_60,
            target_size=feature.shape[-2:],
        )
        return self.interaction(
            visual_feature=feature,
            region_feature=region,
            mean_60=mean_60,
        )
