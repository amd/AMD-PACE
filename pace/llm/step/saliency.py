# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import List

import torch
import torch.nn as nn

from pace.llm.step.hf_modules import find_gated_mlps
from pace.utils.logging import PACE_LLM_ASSERT


class SaliencyAccumulator:
    """
    Accumulates ModHiFi channel saliency over calibration data.

    ModHiFi scores an FFN channel by how well it alone reconstructs the layer output. The score is a function of the component similarity matrix Q, where Q[i,j] is the expected inner product of the contributions of channels i and j. For a single channel it reduces to the diagonal of Q:

        score[i] = ‖W_down[:,i]‖² · E[Φ_i²]

    This requires one running sum per channel rather than the full `d_ff × d_ff` matrix.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The model to instrument. Must have dense gated FFN blocks.
        """
        self.blocks = find_gated_mlps(model)
        PACE_LLM_ASSERT(
            len(self.blocks) > 0,
            "found no dense gate/up/down FFN blocks to calibrate. Mixture-of-experts and non-gated FFNs are not supported by STeP calibration.",
        )
        self.sum_squares: List[torch.Tensor] = [None] * len(self.blocks)
        self.token_counts: List[int] = [0] * len(self.blocks)
        self._handles = []

    def __enter__(self):
        for index, (_, block) in enumerate(self.blocks):
            self._handles.append(
                block.down_proj.register_forward_hook(self._make_hook(index))
            )
        return self

    def __exit__(self, *exc_info):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def _make_hook(self, index: int):
        def hook(module, inputs, output):
            activations = inputs[0].detach()
            if activations.dim() > 2:
                activations = activations.reshape(-1, activations.shape[-1])
            activations = activations.float()

            squares = activations.pow(2).sum(dim=0)
            running = self.sum_squares[index]
            self.sum_squares[index] = squares if running is None else running + squares
            self.token_counts[index] += activations.shape[0]

        return hook

    def score(self, index: int) -> torch.Tensor:
        """
        Finish the statistic for one block, on the device its weights are on.

        Args:
            index: Block index, in model definition order.

        Returns:
            torch.Tensor: `d_ff` saliency scores on CPU.
        """
        name, block = self.blocks[index]
        PACE_LLM_ASSERT(
            self.token_counts[index] > 0,
            f"{name} saw no calibration tokens; check that the forward passes reached every layer",
        )
        # A slice of columns at a time: reducing the whole weight at once makes several float copies of it on the device.
        norms = torch.cat(
            [
                columns.float().pow(2).sum(dim=0)
                for columns in block.down_proj.weight.detach().split(1024, dim=1)
            ]
        )
        mean_square = self.sum_squares[index] / self.token_counts[index]
        return (norms * mean_square).cpu()


def channel_order(scores: torch.Tensor) -> torch.Tensor:
    """
    Rank channels most salient first.

    Args:
        scores: Saliency per channel.

    Returns:
        torch.Tensor: int64 permutation.
    """
    return scores.argsort(descending=True)
