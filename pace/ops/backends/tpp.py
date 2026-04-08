# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
from typing import Optional

import torch
import torch.nn as nn

import pace  # noqa: F401
from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import OperatorType, FusedOperatorType, BackendType, DataType


_TPP_BLOCK_SIZE = int(os.getenv("LIBXSMM_BLOCK_SIZE", 32))


def _pack_weight_for_block_size(weight_2d, block_size):
    """Pack a 2D weight [M, N] into 5D TPP format for a given block size."""
    M, N = weight_2d.size(0), weight_2d.size(1)
    if M % block_size != 0 or N % 64 != 0:
        return None
    w = weight_2d.reshape(M // block_size, block_size, N // 64, 32, 2)
    return torch.permute(w, (0, 2, 3, 1, 4)).contiguous()


@backend_registry.register(OperatorType.LINEAR, BackendType.TPP, [DataType.BFLOAT16])
class TPPLinear(BackendBase):

    def preprocess(self, layer):
        weight = layer.weight
        block_size = int(os.getenv("LIBXSMM_BLOCK_SIZE", 32))
        # For the optimized path, the weight should be reshaped to 5D tensor
        # with shape (M/block_size, block_size, N/64, 32, 2)
        if weight.size(0) % block_size == 0 and weight.size(1) % 64 == 0:
            weight = torch.reshape(
                weight,
                (weight.size(0) // block_size, block_size, weight.size(1) // 64, 32, 2),
            )
            layer.weight = nn.Parameter(
                torch.permute(weight, (0, 2, 3, 1, 4)).contiguous()
            )

    def preprocess_input(self, input: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input tensor to ensure it has the correct shape.
        """
        self.orig_shape = input.shape[:-1]
        if input.dim() < 3:
            for _ in range(3 - input.dim()):
                input = input.unsqueeze(0)
        elif input.dim() > 3:
            input = input.reshape(-1, input.size(-1))
        return input

    def postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the output tensor to restore the original shape.
        """
        output = output.reshape(*self.orig_shape, -1)
        return output

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_plain(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARRELU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearRelu(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_relu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARGELU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearGelu(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_gelu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARSILU, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearSilU(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        output = torch.ops.pace.libxsmmlinear_silu(input, weight, bias)
        return self.postprocess_output(output)


@backend_registry.register(
    FusedOperatorType.FUSEDLINEARMUL, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedLinearMul(TPPLinear):

    def preprocess(self, layer):
        super().preprocess(layer.linear)

    def execute(
        self,
        input: torch.Tensor,
        mul: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.preprocess_input(input)
        mul = self.preprocess_input(mul)
        output = torch.ops.pace.libxsmmlinear_mul(input, mul, weight, bias)
        return self.postprocess_output(output)


_FUSED_MLP_ACTIVATIONS = {"silu", "gelu", "relu"}
_FUSED_MLP_ENABLED = os.getenv("PACE_FUSED_MLP", "1") != "0"

_GELU_ALIASES = {"gelu_new": "gelu", "gelu_pytorch_tanh": "gelu"}

_ACT_LINEAR_OPS = {
    "silu": lambda: torch.ops.pace.libxsmmlinear_silu,
    "gelu": lambda: torch.ops.pace.libxsmmlinear_gelu,
    "relu": lambda: torch.ops.pace.libxsmmlinear_relu,
}


def _unwrap(x):
    """Unwrap single-element list/tuple from IMBPS-style chunked weights."""
    return x[0] if isinstance(x, (list, tuple)) else x


def _try_pack(weight, block_size):
    if (
        weight.dim() == 2
        and weight.size(0) % block_size == 0
        and weight.size(1) % 64 == 0
    ):
        packed = _pack_weight_for_block_size(weight.data, block_size)
        if packed is not None:
            return nn.Parameter(packed)
    return None


@backend_registry.register(
    FusedOperatorType.FUSEDMLPLINEAR, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedMLP(BackendBase):

    def preprocess(self, layer):
        for proj in [layer.gate_proj, layer.up_proj]:
            if proj is not None:
                packed = _try_pack(proj.linear.weight, _TPP_BLOCK_SIZE)
                if packed is not None:
                    proj.linear.weight = packed

        packed = _try_pack(layer.down_proj.weight, _TPP_BLOCK_SIZE)
        if packed is not None:
            layer.down_proj.weight = packed

        layer.gate_proj_weight_chunks = (
            layer.gate_proj.linear.weight if layer.gate_proj else None
        )
        layer.gate_proj_bias_chunks = (
            layer.gate_proj.linear.bias if layer.gate_proj else None
        )
        layer.up_proj_weight_chunks = layer.up_proj.linear.weight
        layer.up_proj_bias_chunks = layer.up_proj.linear.bias
        layer.down_proj_weight_chunks = layer.down_proj.weight

    def execute(
        self,
        input: torch.Tensor,
        up_proj_weights,
        up_proj_bias,
        down_proj_weights,
        down_proj_bias: Optional[torch.Tensor],
        activation: Optional[str],
        gate_proj_weights=None,
        gate_proj_bias=None,
    ) -> torch.Tensor:
        activation = _GELU_ALIASES.get(activation, activation)
        batch_shape = input.shape[:-1]
        if input.dim() < 3:
            for _ in range(3 - input.dim()):
                input = input.unsqueeze(0)

        wt_up = _unwrap(up_proj_weights)
        wt_down = _unwrap(down_proj_weights)
        ub = _unwrap(up_proj_bias)

        if gate_proj_weights is not None:
            wt_gate = _unwrap(gate_proj_weights)
            gb = _unwrap(gate_proj_bias)
        else:
            wt_gate = None
            gb = None

        if (
            _FUSED_MLP_ENABLED
            and activation in _FUSED_MLP_ACTIVATIONS
            and wt_up.dim() == 5
            and wt_down.dim() == 5
            and (wt_gate is None or wt_gate.dim() == 5)
        ):
            out = torch.ops.pace.libxsmm_fused_mlp(
                input, wt_gate, wt_up, wt_down, gb, ub, down_proj_bias, activation
            )
            return out.reshape(*batch_shape, -1)

        if wt_gate is not None:
            act_op = _ACT_LINEAR_OPS.get(activation, _ACT_LINEAR_OPS["silu"])()
            gate_out = act_op(input, wt_gate, gb)
            inter = torch.ops.pace.libxsmmlinear_mul(input, gate_out, wt_up, ub)
        else:
            act_op_fn = _ACT_LINEAR_OPS.get(activation)
            if act_op_fn is not None:
                inter = act_op_fn()(input, wt_up, ub)
            else:
                inter = torch.ops.pace.libxsmmlinear_plain(input, wt_up, ub)

        out = torch.ops.pace.libxsmmlinear_plain(inter, wt_down, down_proj_bias)
        return out.reshape(*batch_shape, -1)
