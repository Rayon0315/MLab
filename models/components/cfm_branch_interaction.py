from __future__ import annotations

import torch
import torch.nn as nn


BRANCH_ORDER = (
    "tcrm",
    "ssrm",
    "ldfm",
    "egpcm",
)


def _stack_branches(
    branches: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [branches[name] for name in BRANCH_ORDER],
        dim=1,
    )


def _concat_branches(
    branch_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        list(branch_tensor.unbind(dim=1)),
        dim=1,
    )


class Stage4BranchTap:
    """Capture TCRM / SSRM / LDFM / EGPCM from the last Stage-4 CFMBlock."""

    BRANCHES = {
        "tcrm": "TCRM",
        "ssrm": "SSRM",
        "ldfm": "LDFM",
        "egpcm": "EGPCM",
    }

    def __init__(self, backbone: nn.Module) -> None:
        self._outputs: dict[str, torch.Tensor] = {}
        self._handles = []

        block = backbone.stages[6].blocks[-1]

        for branch_name, module_name in self.BRANCHES.items():
            module = getattr(block, module_name)
            self._register(branch_name, module)

    def _register(
        self,
        branch_name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[branch_name] = output

        self._handles.append(
            module.register_forward_hook(hook)
        )

    def clear(self) -> None:
        self._outputs.clear()

    def pop(self) -> dict[str, torch.Tensor]:
        expected = set(self.BRANCHES)
        missing = expected - self._outputs.keys()

        if missing:
            raise RuntimeError(
                "Failed to capture Stage4 CFM branches: "
                + ", ".join(sorted(missing))
            )

        outputs = self._outputs
        self._outputs = {}
        return outputs


class AdaptiveFourBranchFusion(nn.Module):
    """
    Spatially adaptive four-way competitive weighting for the four
    Stage-4 CFMNet branch outputs.

    The gate sees all four branches jointly and predicts four spatial
    weights with softmax competition. The weights are multiplied by 4,
    so the zero-initialized gate starts exactly as raw branch concat:

        uniform softmax = 1/4 -> scale = 1
    """

    def __init__(
        self,
        branch_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        total_channels = 4 * branch_channels
        hidden_channels = (
            hidden_channels
            if hidden_channels is not None
            else branch_channels
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                total_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                4,
                kernel_size=1,
                bias=True,
            ),
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        branches: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        stacked = _stack_branches(branches)
        raw_concat = _concat_branches(stacked)

        weights = torch.softmax(
            self.gate(raw_concat),
            dim=1,
        )

        weights = 4.0 * weights

        weighted = (
            stacked
            * weights.unsqueeze(2)
        )

        return _concat_branches(weighted)


class CompleteGraphBranchAttention(nn.Module):
    """
    Complete-graph attention over the four aligned CFMNet branches.

    At each spatial position, TCRM / SSRM / LDFM / EGPCM are treated as
    four tokens. Attention is only across the branch dimension, never
    across spatial positions. The diagonal is masked, so every branch
    receives messages only from the other three branches.

    There are six undirected branch pairs (C(4, 2)); attention remains
    directional internally, so the data can learn different i->j and
    j->i strengths without allocating six separate attention modules.

    A per-branch zero-initialized residual scale makes the module start
    exactly as raw branch concat.
    """

    def __init__(
        self,
        branch_channels: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        if branch_channels % num_heads != 0:
            raise ValueError(
                "branch_channels must be divisible by num_heads"
            )

        self.branch_channels = branch_channels
        self.num_heads = num_heads
        self.head_dim = branch_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(branch_channels)
        self.qkv = nn.Linear(
            branch_channels,
            3 * branch_channels,
            bias=qkv_bias,
        )
        self.proj = nn.Linear(
            branch_channels,
            branch_channels,
            bias=True,
        )

        self.alpha = nn.Parameter(
            torch.zeros(4)
        )

    def _qkv(
        self,
        branches: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        stacked = _stack_branches(branches)

        tokens = stacked.permute(
            0,
            3,
            4,
            1,
            2,
        )

        qkv = self.qkv(
            self.norm(tokens)
        )

        batch, height, width, _, _ = qkv.shape

        qkv = qkv.view(
            batch,
            height,
            width,
            4,
            3,
            self.num_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=4)

        q = q.permute(0, 4, 1, 2, 3, 5)
        k = k.permute(0, 4, 1, 2, 3, 5)
        v = v.permute(0, 4, 1, 2, 3, 5)

        return stacked, q, k, v

    def _attention_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        scores = (
            q
            @ k.transpose(-2, -1)
        ) * self.scale

        diagonal_mask = torch.eye(
            4,
            device=scores.device,
            dtype=torch.bool,
        ).view(1, 1, 1, 1, 4, 4)

        scores = scores.masked_fill(
            diagonal_mask,
            float("-inf"),
        )

        return torch.softmax(
            scores,
            dim=-1,
        )

    def compute_attention_weights(
        self,
        branches: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, q, k, _ = self._qkv(branches)
        return self._attention_weights(q, k)

    def forward(
        self,
        branches: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        stacked, q, k, v = self._qkv(branches)
        weights = self._attention_weights(q, k)

        message = weights @ v

        message = message.permute(
            0,
            2,
            3,
            4,
            1,
            5,
        ).contiguous()

        batch, height, width, _, _, _ = message.shape

        message = message.view(
            batch,
            height,
            width,
            4,
            self.branch_channels,
        )

        message = self.proj(message)

        message = message.permute(
            0,
            3,
            4,
            1,
            2,
        )

        updated = (
            stacked
            + self.alpha.view(1, 4, 1, 1, 1)
            * message
        )

        return _concat_branches(updated)
