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
from pace.utils.logging import PACE_LLM_ASSERT

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

# The TPP kernels expose one GELU; the tanh-approximation spellings differ from
# it by less than bf16 resolution, so they all route to it.
_GELU_ALIASES = {
    "gelu_new": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "gelu_tanh": "gelu",
}

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


@backend_registry.register(
    FusedOperatorType.FUSEDMOE, BackendType.TPP, [DataType.BFLOAT16]
)
class TPPFusedMoE(BackendBase):
    """TPP backend for MoEExperts — packs per-expert weights to 6D VNNI tensors.

    preprocess() is called automatically by BaseModelForCausalLM.preprocess_weights()
    after all expert weights have been loaded by load_weights(). It packs
    gate_proj_weights, up_proj_weights, down_proj_weights into 6D VNNI format
    and stacks across experts into layer._fused_wt_{gate,up,down}.
    If packing fails (misaligned dimensions), preprocessing aborts via
    PACE_LLM_ASSERT with an actionable message (packing is mandatory for the
    fused backend; there is no implicit runtime fallback).
    """

    def preprocess(self, layer):
        bs = int(os.environ.get("LIBXSMM_BLOCK_SIZE", "32"))

        def _pack(weight_2d, what):
            packed = _pack_weight_for_block_size(weight_2d, bs)
            PACE_LLM_ASSERT(
                packed is not None,
                f"TPPFusedMoE: {what} packing failed — shape "
                f"{tuple(weight_2d.shape)} not aligned to block_size={bs} or "
                f"N%64!=0.",
            )
            return packed

        gate_packed, up_packed, down_packed = [], [], []
        gate_biases, up_biases, down_biases = [], [], []
        has_bias = layer.bias_available

        for e in range(layer.num_experts):
            up_packed.append(_pack(layer.up_projs[e].weight.data, f"up[{e}]"))
            down_packed.append(_pack(layer.down_projs[e].weight.data, f"down[{e}]"))
            if layer.gate:
                gate_packed.append(_pack(layer.gate_projs[e].weight.data, f"gate[{e}]"))
            if has_bias:
                up_biases.append(layer.up_projs[e].bias.data)
                down_biases.append(layer.down_projs[e].bias.data)
                if layer.gate:
                    gate_biases.append(layer.gate_projs[e].bias.data)

        # Fold same-shape shared experts into the stack as extra experts (indices
        # [num_experts, num_experts + num_shared_experts)). The operator emits
        # their (id, weight=1.0) routing entries. Shared MLPs are NATIVE, so
        # their raw weights live in the sub-op linears.
        if getattr(layer, "foldable_shared", False):
            for s, mlp in enumerate(layer.shared_mlps):
                up_packed.append(
                    _pack(mlp.up_proj.linear.weight.data, f"shared up[{s}]")
                )
                down_packed.append(
                    _pack(mlp.down_proj.weight.data, f"shared down[{s}]")
                )
                if layer.gate:
                    gate_packed.append(
                        _pack(mlp.gate_proj.linear.weight.data, f"shared gate[{s}]")
                    )
                if has_bias:
                    up_biases.append(mlp.up_proj.linear.bias.data)
                    down_biases.append(mlp.down_proj.bias.data)
                    if layer.gate:
                        gate_biases.append(mlp.gate_proj.linear.bias.data)
            layer._folded_shared = True

        layer._fused_wt_gate = torch.stack(gate_packed) if gate_packed else None
        layer._fused_wt_up = torch.stack(up_packed)
        layer._fused_wt_down = torch.stack(down_packed)
        layer._fused_gate_bias = torch.stack(gate_biases) if gate_biases else None
        layer._fused_up_bias = torch.stack(up_biases) if up_biases else None
        layer._fused_down_bias = torch.stack(down_biases) if down_biases else None

        # Free the unpacked per-expert weights (now in the 6D stack).
        for e in range(layer.num_experts):
            layer.up_projs[e].weight.data = torch.empty(0)
            layer.down_projs[e].weight.data = torch.empty(0)
            if layer.gate:
                layer.gate_projs[e].weight.data = torch.empty(0)
        if getattr(layer, "_folded_shared", False):
            for mlp in layer.shared_mlps:
                mlp.up_proj.linear.weight.data = torch.empty(0)
                mlp.down_proj.weight.data = torch.empty(0)
                if layer.gate:
                    mlp.gate_proj.linear.weight.data = torch.empty(0)

    def execute(
        self,
        output,
        hidden_states,
        topk_ids,
        topk_weights,
        activation,
        num_experts,
        wt_gate,
        wt_up,
        wt_down,
        gate_bias,
        up_bias,
        down_bias,
    ):
        """Run the expert engine over the packed weights. Routing is supplied as
        topk_ids / topk_weights; folded shared experts are already extra columns.
        """
        activation = _GELU_ALIASES.get(activation, activation)
        # Void in-place op: writes into `output`; return that buffer.
        torch.ops.pace.libxsmm_moe_engine(
            output,
            hidden_states.contiguous(),
            topk_ids,
            topk_weights,
            wt_gate,
            wt_up,
            wt_down,
            gate_bias,
            up_bias,
            down_bias,
            activation,
            num_experts,
        )
        return output


@backend_registry.register(
    OperatorType.MOEROUTER_SOFTMAX,
    BackendType.TPP,
    [DataType.FLOAT32, DataType.BFLOAT16],
)
class TPPSoftmaxRouter(BackendBase):
    """C++ top-k softmax router. Sigmoid/grouped schemes have no TPP backend and
    fall back to NATIVE via the registry."""

    def execute(self, router_logits, topk, renormalize=True, **_):
        return torch.ops.pace.moe_route_softmax(
            router_logits.contiguous(), topk, renormalize
        )
