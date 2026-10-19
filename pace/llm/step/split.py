# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from pace.llm.step.budget import CHANNEL_ALIGNMENT
from pace.llm.step.hf_modules import find_gated_mlps
from pace.llm.step.residual import ResidualFFN
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO


class PinnedBuffer:
    """
    A grow-only pinned staging buffer for hidden states moving between the GPU and the CPU.

    Pinning is costly, so the buffer is allocated once at the largest size seen and sliced for smaller calls.
    """

    def __init__(self):
        self._storage: Optional[torch.Tensor] = None

    def view(self, shape, dtype: torch.dtype) -> torch.Tensor:
        needed = 1
        for dim in shape:
            needed *= dim
        if (
            self._storage is None
            or self._storage.dtype != dtype
            or self._storage.numel() < needed
        ):
            self._storage = torch.empty(needed, dtype=dtype, pin_memory=True)
        return self._storage[:needed].view(*shape)


@dataclass
class SplitState:
    """
    Mode and staging shared by every split FFN block.

    Drafting runs the GPU prefix alone; verification runs the prefix plus the CPU residual, which reconstructs the full model. A single shared object makes switching modes independent of layer count.

    The pinned staging is shared as well, so it holds one layer's hidden states. That is safe because the layers run in order on one stream: a layer writes the buffers only after its own input copy has landed, and the stream runs that copy after the previous layer's result copy.
    """

    verify: bool = True
    collect_timings: bool = False
    residual_seconds: float = 0.0
    staging: PinnedBuffer = field(default_factory=PinnedBuffer)
    # A host-to-device copy out of pageable memory is synchronous however it is flagged, so the residual output is staged through pinned memory to keep the transfer asynchronous.
    result_staging: PinnedBuffer = field(default_factory=PinnedBuffer)

    def reset_timings(self) -> None:
        self.residual_seconds = 0.0


class SplitFFN(nn.Module):
    """
    An FFN block whose salient channels are held on the GPU and the remainder on the CPU.

    The wrapped block retains its own forward for the GPU prefix, so architecture-specific behaviour stays with the HuggingFace module. This class owns the residual channels, the transfers and the sum.
    """

    def __init__(
        self,
        block: nn.Module,
        gpu_channels: int,
        state: SplitState,
        device: torch.device,
        activation: str,
        residual_backend: str = "auto",
    ):
        """
        Args:
            block: A dense gate/up/down FFN, weights still on CPU.
            gpu_channels: Channels to keep on the GPU. Must be a prefix.
            state: Shared draft/verify mode.
            device: GPU device for the prefix.
            activation: Activation name from the model config.
            residual_backend: Passed to ResidualFFN.
        """
        super().__init__()
        total = block.up_proj.out_features
        PACE_LLM_ASSERT(
            0 <= gpu_channels <= total,
            f"gpu_channels {gpu_channels} outside [0, {total}]",
        )

        self.block = block
        self.state = state
        self.device = device
        self.total_channels = total

        residual_weights = self._trim_to(gpu_channels)
        self.residual = ResidualFFN(
            *residual_weights, activation=activation, backend=residual_backend
        )
        self._copy_done = torch.cuda.Event() if device.type == "cuda" else None

    def _trim_to(self, gpu_channels: int):
        """
        Split the block's weights, moving the prefix to the device and returning the residual.

        Args:
            gpu_channels: Number of leading intermediate channels kept on the device.

        Returns:
            tuple: Residual gate, up and down weights on the host, with None for a projection the block does not have.
        """
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(self.block, name, None)
            PACE_LLM_ASSERT(
                projection is None or projection.bias is None,
                f"{name} has a bias, which STeP's channel split does not handle yet",
            )

        residual = []
        for name in ("gate_proj", "up_proj"):
            projection = getattr(self.block, name, None)
            if projection is None:
                residual.append(None)
                continue
            weight = projection.weight.data
            # A slice would keep the whole weight alive, so both parts are copied. The prefix goes through a host clone, as in `move_to_device`.
            residual.append(weight[gpu_channels:].clone())
            projection.weight.data = weight[:gpu_channels].clone().to(self.device)
            projection.out_features = gpu_channels

        down = self.block.down_proj
        weight = down.weight.data
        residual.append(weight[:, gpu_channels:].contiguous())
        down.weight.data = weight[:, :gpu_channels].contiguous().to(self.device)
        down.in_features = gpu_channels

        return tuple(residual)

    @property
    def gpu_channels(self) -> int:
        return self.block.up_proj.out_features

    def forward(self, hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.state.verify or self.residual.channels == 0:
            return self.block(hidden, *args, **kwargs)
        if self.gpu_channels == 0:
            return self.residual.forward(hidden.cpu()).to(self.device)
        return self._forward_coexecuted(hidden, *args, **kwargs)

    def _residual_of(self, staged):
        if not self.state.collect_timings:
            return self.residual.forward(staged)
        start = time.perf_counter()
        result = self.residual.forward(staged)
        self.state.residual_seconds += time.perf_counter() - start
        return result

    def _forward_coexecuted(self, hidden, *args, **kwargs):
        """
        Run the CPU residual concurrently with the GPU prefix.

        The prefix and the residual both depend only on this layer's input, so they may be evaluated at once.

        A single stream is used. The copy is issued first and an event recorded immediately after it, so waiting on that event releases the host once the hidden state has landed, while the prefix kernels queued behind it are still running.
        """
        PACE_LLM_ASSERT(
            self._copy_done is not None,
            f"co-execution needs a GPU device, got {self.device}",
        )
        staged = self.state.staging.view(hidden.shape, self.residual.dtype)
        staged.copy_(hidden, non_blocking=True)
        self._copy_done.record()

        prefix = self.block(hidden, *args, **kwargs)

        self._copy_done.synchronize()
        residual = self._residual_of(staged)
        returning = self.state.result_staging.view(residual.shape, residual.dtype)
        returning.copy_(residual)
        return prefix + returning.to(self.device, non_blocking=True)

    def evict(self, channels: int) -> int:
        """
        Move the least salient resident channels from the GPU to the CPU.

        Within a request eviction only goes one way, since the pressure driving it is a monotonically growing KV cache; `restore` undoes it between requests. The terminal state is the entire FFN on the CPU with only attention resident.

        A partial move must leave the residual a multiple of CHANNEL_ALIGNMENT so it continues to pack for the fast kernel. A request smaller than one aligned block is rounded up rather than down, since rounding down would stall eviction short of full offload.

        Args:
            channels: How many to move.

        Returns:
            int: How many were actually moved.
        """
        channels = min(channels, self.gpu_channels)
        if channels < self.gpu_channels:
            channels = max(
                (channels // CHANNEL_ALIGNMENT) * CHANNEL_ALIGNMENT,
                min(CHANNEL_ALIGNMENT, self.gpu_channels),
            )
        if channels <= 0:
            return 0

        keep = self.gpu_channels - channels
        moved = []
        for name in ("gate_proj", "up_proj"):
            projection = getattr(self.block, name, None)
            if projection is None:
                moved.append(None)
                continue
            weight = projection.weight.data
            moved.append(weight[keep:].to("cpu"))
            # Cloned rather than sliced: a row slice is a view, and the evicted rows' GPU memory would never be freed.
            projection.weight.data = weight[:keep].clone()
            projection.out_features = keep

        down = self.block.down_proj
        weight = down.weight.data
        moved.append(weight[:, keep:].to("cpu").contiguous())
        down.weight.data = weight[:, :keep].contiguous()
        down.in_features = keep

        self.residual.absorb(moved[0], moved[1], moved[2])
        return channels

    def restore(self, channels: int) -> int:
        """
        Return the most salient residual channels to the GPU, undoing eviction.

        Eviction is one-way while a request runs, since its KV cache only grows. Between requests that cache is gone, so the channels it displaced can come back.

        On the PACE kernels fewer channels move when needed to leave the residual a multiple of CHANNEL_ALIGNMENT, so it continues to pack. A torch residual has no such constraint and moves exactly what was asked.

        Args:
            channels: How many to move.

        Returns:
            int: How many were actually moved.
        """
        channels = max(0, min(channels, self.residual.channels))
        remaining = self.residual.channels - channels
        if remaining > 0 and self.residual.use_kernels:
            remaining = -(-remaining // CHANNEL_ALIGNMENT) * CHANNEL_ALIGNMENT
            channels = max(0, self.residual.channels - remaining)
        if channels == 0:
            return 0

        gate, up, down = self.residual.release(channels)
        for name, rows in (("gate_proj", gate), ("up_proj", up)):
            projection = getattr(self.block, name, None)
            if projection is None:
                continue
            projection.weight.data = torch.cat(
                [projection.weight.data, rows.to(self.device)], dim=0
            )
            projection.out_features = projection.weight.shape[0]

        down_proj = self.block.down_proj
        down_proj.weight.data = torch.cat(
            [down_proj.weight.data, down.to(self.device)], dim=1
        )
        down_proj.in_features = down_proj.weight.shape[1]
        return channels


def install_split(
    model: nn.Module,
    gpu_channels: int,
    device: torch.device,
    activation: str,
    residual_backend: str = "auto",
    disable_tqdm: bool = False,
) -> tuple:
    """
    Replace every dense FFN block with a GPU/CPU split version.

    Call before moving the rest of the model to the GPU. Each block's prefix goes to the device as it is split, so the host only ever adds the residual rather than the whole FFN.

    Args:
        model: Causal LM with weights on CPU.
        gpu_channels: Channels to keep on the GPU, per block.
        device: GPU device.
        activation: Activation name from the model config.
        residual_backend: Passed to ResidualFFN.
        disable_tqdm: Hide the progress bar.

    Returns:
        tuple: (SplitState, list of SplitFFN) for mode switching and eviction.
    """
    blocks = find_gated_mlps(model)
    PACE_LLM_ASSERT(len(blocks) > 0, "found no dense gate/up/down FFN blocks to split")

    state = SplitState(verify=True)
    splits: List[SplitFFN] = []
    for name, block in tqdm(
        blocks, desc="Splitting FFN blocks", unit="block", disable=disable_tqdm
    ):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        split = SplitFFN(
            block, gpu_channels, state, device, activation, residual_backend
        )
        setattr(parent, name.rsplit(".", 1)[-1], split)
        splits.append(split)

    PACE_LLM_INFO(
        f"split {len(splits)} FFN blocks: {gpu_channels} channels on {device}, {splits[0].residual.describe()} per block"
    )
    return state, splits
