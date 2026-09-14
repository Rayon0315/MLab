from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeSaliencyCandidateVerifier(nn.Module):
    """
    Relative Saliency Candidate Verification (RSCV).

    Goal:
        verify whether each high-response candidate should remain
        salient relative to other candidates and the whole scene.

    Input:
        semantic_feature:
            high-level decoder feature, B x C x H x W

        saliency_logits:
            saliency logits after CFPS, B x 1 x Hs x Ws

    Procedure:
        1. find Top-K local saliency peaks at semantic-feature scale;
        2. use the feature at each peak as a semantic query;
        3. build a soft candidate region by cosine matching;
        4. pool object / surround / global tokens;
        5. perform one lightweight candidate self-attention block;
        6. predict candidate-level relative saliency;
        7. convert low candidate saliency into one-way suppression.

    Candidate discovery and soft-region masks are detached.
    The verifier therefore learns to judge the candidates generated
    by the current saliency model instead of reshaping the candidate
    extraction process itself.
    """

    def __init__(
        self,
        channels: int = 128,
        num_candidates: int = 4,
        num_heads: int = 4,
        ff_channels: int = 256,
        nms_kernel: int = 5,
        similarity_temperature: float = 0.2,
        initial_accept_bias: float = 2.0,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads."
            )

        if nms_kernel % 2 == 0:
            raise ValueError(
                "nms_kernel must be odd."
            )

        self.channels = channels
        self.num_candidates = num_candidates
        self.nms_kernel = nms_kernel
        self.similarity_temperature = (
            similarity_temperature
        )

        # object + surround + global + [confidence, x, y, area]
        descriptor_channels = (
            channels * 3 + 4
        )

        self.candidate_encoder = nn.Sequential(
            nn.Linear(
                descriptor_channels,
                channels,
            ),
            nn.LayerNorm(
                channels
            ),
            nn.GELU(),
        )

        self.attention_norm = nn.LayerNorm(
            channels
        )

        self.candidate_attention = (
            nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                batch_first=True,
            )
        )

        self.ffn_norm = nn.LayerNorm(
            channels
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                channels,
                ff_channels,
            ),
            nn.GELU(),
            nn.Linear(
                ff_channels,
                channels,
            ),
        )

        self.score_head = nn.Sequential(
            nn.Linear(
                channels,
                channels // 2,
            ),
            nn.GELU(),
            nn.Linear(
                channels // 2,
                1,
            ),
        )

        # Start by accepting most candidates.
        # sigmoid(2) ~= 0.88 and the initial veto penalty
        # softplus(-2) ~= 0.127.
        nn.init.zeros_(
            self.score_head[-1].weight
        )
        nn.init.constant_(
            self.score_head[-1].bias,
            initial_accept_bias,
        )

    def _extract_candidate_indices(
        self,
        probability: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        probability:
            B x 1 x H x W

        Returns:
            indices: B x K flattened spatial indices
            scores:  B x K peak probabilities
        """
        padding = self.nms_kernel // 2

        local_maximum = F.max_pool2d(
            probability,
            kernel_size=self.nms_kernel,
            stride=1,
            padding=padding,
        )

        peak_map = torch.where(
            probability
            >= local_maximum - 1e-6,
            probability,
            torch.zeros_like(
                probability
            ),
        )

        flat_peak = peak_map.flatten(
            start_dim=2
        ).squeeze(1)

        k = min(
            self.num_candidates,
            flat_peak.shape[-1],
        )

        scores, indices = torch.topk(
            flat_peak,
            k=k,
            dim=-1,
        )

        if k < self.num_candidates:
            pad_count = (
                self.num_candidates - k
            )

            indices = torch.cat(
                [
                    indices,
                    indices[:, -1:]
                    .expand(
                        -1,
                        pad_count,
                    ),
                ],
                dim=1,
            )

            scores = torch.cat(
                [
                    scores,
                    scores[:, -1:]
                    .expand(
                        -1,
                        pad_count,
                    ),
                ],
                dim=1,
            )

        return (
            indices.detach(),
            scores.detach(),
        )

    def _build_candidate_regions(
        self,
        semantic_feature: torch.Tensor,
        saliency_probability: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build B x K x H x W soft candidate regions.

        The relative cosine similarity is normalized so each
        candidate has maximum affinity 1, then restricted by the
        current saliency probability.
        """
        (
            batch_size,
            channels,
            height,
            width,
        ) = semantic_feature.shape

        normalized_feature = F.normalize(
            semantic_feature,
            dim=1,
            eps=1e-6,
        )

        flattened_feature = (
            normalized_feature.flatten(
                start_dim=2
            )
        )

        gather_index = (
            candidate_indices
            .unsqueeze(1)
            .expand(
                -1,
                channels,
                -1,
            )
        )

        candidate_queries = torch.gather(
            flattened_feature,
            dim=2,
            index=gather_index,
        ).transpose(
            1,
            2,
        )

        similarity = torch.einsum(
            "bkc,bchw->bkhw",
            candidate_queries,
            normalized_feature,
        )

        maximum = similarity.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        relative_affinity = torch.exp(
            (
                similarity
                - maximum
            )
            / self.similarity_temperature
        )

        candidate_regions = (
            relative_affinity
            * saliency_probability
        )

        return candidate_regions.detach()

    @staticmethod
    def _weighted_pool(
        feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        feature:
            B x C x H x W
        weight:
            B x K x H x W

        returns:
            B x K x C
        """
        numerator = torch.einsum(
            "bchw,bkhw->bkc",
            feature,
            weight,
        )

        denominator = weight.sum(
            dim=(-2, -1),
            keepdim=False,
        ).unsqueeze(-1).clamp_min(
            1e-6
        )

        return (
            numerator
            / denominator
        )

    def forward(
        self,
        semantic_feature: torch.Tensor,
        saliency_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            batch_size,
            channels,
            height,
            width,
        ) = semantic_feature.shape

        if channels != self.channels:
            raise ValueError(
                "semantic feature channel mismatch: "
                f"{channels} != {self.channels}"
            )

        # Candidate discovery follows the current CFPS output,
        # but does not backpropagate through peak selection.
        saliency_probability = torch.sigmoid(
            saliency_logits.detach()
        )

        saliency_probability = F.interpolate(
            saliency_probability,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        (
            candidate_indices,
            candidate_peak_scores,
        ) = self._extract_candidate_indices(
            saliency_probability
        )

        candidate_regions = (
            self._build_candidate_regions(
                semantic_feature=semantic_feature,
                saliency_probability=(
                    saliency_probability
                ),
                candidate_indices=(
                    candidate_indices
                ),
            )
        )

        object_token = self._weighted_pool(
            semantic_feature,
            candidate_regions,
        )

        surround_weight = (
            1.0
            - candidate_regions
        )

        surround_token = self._weighted_pool(
            semantic_feature,
            surround_weight,
        )

        global_token = semantic_feature.mean(
            dim=(-2, -1)
        ).unsqueeze(1).expand(
            -1,
            self.num_candidates,
            -1,
        )

        area = candidate_regions.mean(
            dim=(-2, -1)
        ).unsqueeze(-1)

        y = (
            candidate_indices
            // width
        ).float()

        x = (
            candidate_indices
            % width
        ).float()

        if width > 1:
            x = (
                2.0
                * x
                / float(width - 1)
                - 1.0
            )
        else:
            x = torch.zeros_like(x)

        if height > 1:
            y = (
                2.0
                * y
                / float(height - 1)
                - 1.0
            )
        else:
            y = torch.zeros_like(y)

        metadata = torch.stack(
            [
                candidate_peak_scores,
                x,
                y,
                area.squeeze(-1),
            ],
            dim=-1,
        )

        descriptor = torch.cat(
            [
                object_token,
                surround_token,
                global_token,
                metadata,
            ],
            dim=-1,
        )

        candidate_token = (
            self.candidate_encoder(
                descriptor
            )
        )

        normalized = self.attention_norm(
            candidate_token
        )

        attended, _ = (
            self.candidate_attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
        )

        candidate_token = (
            candidate_token
            + attended
        )

        candidate_token = (
            candidate_token
            + self.ffn(
                self.ffn_norm(
                    candidate_token
                )
            )
        )

        candidate_logits = (
            self.score_head(
                candidate_token
            ).squeeze(-1)
        )

        # Candidate-level veto:
        #   -log(sigmoid(score)) = softplus(-score) >= 0
        #
        # It is directly interpretable as a rejection penalty and
        # requires no extra suppression-scale hyperparameter.
        rejection_penalty = F.softplus(
            -candidate_logits
        )

        candidate_suppression = (
            candidate_regions
            * rejection_penalty[
                :,
                :,
                None,
                None,
            ]
        )

        suppression = (
            candidate_suppression.amax(
                dim=1,
                keepdim=True,
            )
        )

        suppression = F.interpolate(
            suppression,
            size=saliency_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        refined_logits = (
            saliency_logits
            - suppression
        )

        return {
            "refined_logits": refined_logits,
            "candidate_logits": candidate_logits,
            "candidate_regions": candidate_regions,
            "candidate_peak_scores": (
                candidate_peak_scores
            ),
            "candidate_suppression": (
                suppression
            ),
        }
