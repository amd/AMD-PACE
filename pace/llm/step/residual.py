# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

from functools import partial

import torch
import torch.nn.functional as F

from pace.ops.backends.tpp import (
    _GELU_ALIASES,
    _TPP_BLOCK_SIZE,
    _pack_weight_for_block_size,
)
from pace.utils.logging import PACE_LLM_ASSERT

_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_new": partial(F.gelu, approximate="tanh"),
    "gelu_pytorch_tanh": partial(F.gelu, approximate="tanh"),
    "gelu_tanh": partial(F.gelu, approximate="tanh"),
    "relu": F.relu,
}

_ACT_LINEAR_OPS = {
    "silu": "libxsmmlinear_silu",
    "gelu": "libxsmmlinear_gelu",
    "relu": "libxsmmlinear_relu",
}


def unpack_weight(packed: torch.Tensor) -> torch.Tensor:
    """
    Invert the TPP 5D VNNI packing back to a 2D [out, in] weight.

    Args:
        packed: Weight in the layout produced by `_pack_weight_for_block_size`.

    Returns:
        torch.Tensor: Contiguous 2D weight.
    """
    PACE_LLM_ASSERT(
        packed.dim() == 5, f"expected a 5D packed weight, got {packed.dim()}D"
    )
    blocks_out, blocks_in, rows, block_size, pair = packed.shape
    restored = packed.permute(0, 3, 1, 2, 4).contiguous()
    return restored.reshape(blocks_out * block_size, blocks_in * rows * pair)


class ResidualFFN:
    """
    The CPU-resident residual of a partitioned gated FFN.

    Holds the channels that did not fit on the GPU and computes `down(act(gate(x)) * up(x))` over them. Weights are packed into the TPP 5D VNNI layout when their dimensions permit, which is the requirement for the fused `libxsmm_fused_mlp` path.

    The layout is owned by `pace.ops.backends.tpp`; this class reuses it rather than restating it.
    """

    def __init__(
        self,
        gate: Optional[torch.Tensor],
        up: torch.Tensor,
        down: torch.Tensor,
        activation: str = "silu",
        backend: str = "auto",
        block_size: int = _TPP_BLOCK_SIZE,
    ):
        """
        Args:
            gate: Gate projection [channels, d_model], or None for a plain MLP.
            up: Up projection [channels, d_model].
            down: Down projection [d_model, channels].
            activation: Activation name from the model config.
            backend: "pace", "torch", or "auto".
            block_size: TPP packing block size.
        """
        PACE_LLM_ASSERT(
            up.shape == down.T.shape,
            f"up {tuple(up.shape)} and down {tuple(down.shape)} disagree on shape",
        )
        if gate is not None:
            PACE_LLM_ASSERT(
                gate.shape == up.shape,
                f"gate {tuple(gate.shape)} and up {tuple(up.shape)} disagree on shape",
            )

        self.activation = _GELU_ALIASES.get(activation, activation)
        # The kernel exposes one GELU; the torch path can match the configured spelling.
        self.act_fn = _ACTIVATIONS.get(activation)
        self.block_size = block_size
        self.dtype = up.dtype
        self._channels = up.shape[0]

        kernels_usable = backend != "torch" and self.dtype == torch.bfloat16
        if backend == "pace":
            PACE_LLM_ASSERT(
                self.dtype == torch.bfloat16,
                f"the PACE residual kernels are bfloat16 only, got {self.dtype}",
            )
            PACE_LLM_ASSERT(
                self.activation in _ACT_LINEAR_OPS,
                f"activation {self.activation} has no PACE kernel; available: {sorted(_ACT_LINEAR_OPS)}",
            )
        PACE_LLM_ASSERT(
            self.act_fn is not None,
            f"no residual implementation for activation {activation}; available: {sorted(_ACTIVATIONS)}",
        )
        self.use_kernels = kernels_usable and self.activation in _ACT_LINEAR_OPS

        self.gate, self.up, self.down = self._store(gate, up, down)

    def _store(self, gate, up, down):
        if not self.use_kernels:
            return gate, up, down
        return (
            self._maybe_pack(gate),
            self._maybe_pack(up),
            self._maybe_pack(down),
        )

    def _maybe_pack(self, weight):
        if weight is None:
            return None
        packed = _pack_weight_for_block_size(weight.contiguous(), self.block_size)
        return packed if packed is not None else weight.contiguous()

    @property
    def channels(self) -> int:
        """Residual channels held on the CPU."""
        return self._channels

    @property
    def fused(self) -> bool:
        """Whether the single fused kernel applies."""
        if not self.use_kernels:
            return False
        weights = [self.up, self.down] + ([self.gate] if self.gate is not None else [])
        return all(w.dim() == 5 for w in weights)

    def describe(self) -> str:
        if self.fused:
            path = "libxsmm_fused_mlp (packed)"
        elif self.use_kernels:
            path = "libxsmmlinear (unpacked, degraded)"
        else:
            path = "torch"
        return f"{self._channels} channels on {self.up.device}, {path}"

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual contribution for a batch of hidden states.

        Args:
            hidden: [batch, tokens, d_model] on CPU, matching the weight dtype.

        Returns:
            torch.Tensor: [batch, tokens, d_model] to be added to the GPU prefix's output.
        """
        if self._channels == 0:
            return torch.zeros_like(hidden)
        if self.fused:
            return torch.ops.pace.libxsmm_fused_mlp(
                hidden, self.gate, self.up, self.down, None, None, None, self.activation
            )
        if self.use_kernels:
            return self._forward_kernels(hidden)
        return self._forward_torch(hidden)

    def _forward_kernels(self, hidden):
        act_op = getattr(torch.ops.pace, _ACT_LINEAR_OPS[self.activation])
        if self.gate is None:
            inter = act_op(hidden, self.up, None)
        else:
            gate_out = act_op(hidden, self.gate, None)
            inter = torch.ops.pace.libxsmmlinear_mul(hidden, gate_out, self.up, None)
        return torch.ops.pace.libxsmmlinear_plain(inter, self.down, None)

    def _forward_torch(self, hidden):
        act = self.act_fn
        if self.gate is None:
            inter = act(F.linear(hidden, self.up))
        else:
            inter = act(F.linear(hidden, self.gate)) * F.linear(hidden, self.up)
        return F.linear(inter, self.down)

    def weights_2d(self):
        """
        Return the residual weights in plain [out, in] layout.

        Returns:
            tuple: (gate, up, down), with gate None for a plain MLP.
        """

        def flat(weight):
            if weight is None:
                return None
            return unpack_weight(weight) if weight.dim() == 5 else weight

        return flat(self.gate), flat(self.up), flat(self.down)

    def absorb(
        self, gate: Optional[torch.Tensor], up: torch.Tensor, down: torch.Tensor
    ) -> None:
        """
        Prepend channels evicted from the GPU, keeping saliency order.

        Args:
            gate: Gate rows for the evicted channels, or None.
            up: Up rows for the evicted channels.
            down: Down columns for the evicted channels.
        """
        own_gate, own_up, own_down = self.weights_2d()
        merged_up = torch.cat([up, own_up], dim=0)
        merged_down = torch.cat([down, own_down], dim=1)
        merged_gate = None
        if own_gate is not None:
            PACE_LLM_ASSERT(
                gate is not None, "this FFN has a gate projection but none was given"
            )
            merged_gate = torch.cat([gate, own_gate], dim=0)

        self._channels = merged_up.shape[0]
        self.gate, self.up, self.down = self._store(merged_gate, merged_up, merged_down)

    def release(self, channels: int):
        """
        Remove the leading channels, the most salient the CPU holds, and return them.

        The inverse of `absorb`, for returning evicted channels to the GPU.

        Args:
            channels: How many to remove.

        Returns:
            tuple: (gate, up, down) for the removed channels, gate None for a plain MLP.
        """
        PACE_LLM_ASSERT(
            0 <= channels <= self._channels,
            f"cannot release {channels} of {self._channels} residual channels",
        )
        own_gate, own_up, own_down = self.weights_2d()
        released = (
            own_gate[:channels].contiguous() if own_gate is not None else None,
            own_up[:channels].contiguous(),
            own_down[:, :channels].contiguous(),
        )
        self._channels -= channels
        # Row slices are views, so the kept rows are cloned or the torch path holds on to the released ones.
        self.gate, self.up, self.down = self._store(
            own_gate[channels:].clone() if own_gate is not None else None,
            own_up[channels:].clone(),
            own_down[:, channels:].contiguous(),
        )
        return released
