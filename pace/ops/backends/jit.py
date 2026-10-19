# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional, Tuple

import torch

import pace  # noqa: F401
from pace.ops.base import BackendBase
from pace.ops.registry import backend_registry
from pace.ops.enum import (
    OperatorType,
    FusedOperatorType,
    BackendType,
    DataType,
)


@backend_registry.register(
    OperatorType.LINEAR, BackendType.JIT, [DataType.FLOAT32, DataType.BFLOAT16]
)
class JITLinear(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.pace.linear(input, weight, bias)


@backend_registry.register(
    OperatorType.RMSNORM,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITRMSNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        _normalized_shape,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return torch.ops.pace.rmsnorm(input, weight, eps)


@backend_registry.register(
    OperatorType.LAYERNORM,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITLayerNorm(BackendBase):

    def execute(
        self,
        input: torch.Tensor,
        _normalized_shape,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros_like(weight)
        return torch.ops.pace.layernorm(input, weight, bias, eps)


@backend_registry.register(
    FusedOperatorType.FUSED_LAYERNORM_RESIDUAL,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITFusedLayerNormResidual(BackendBase):

    def execute(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bias is None:
            bias = torch.zeros_like(weight)
        return torch.ops.pace.fused_add_layernorm(x, residual, weight, bias, eps)


@backend_registry.register(
    FusedOperatorType.FUSED_RMSNORM_RESIDUAL,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITFusedRMSNormResidual(BackendBase):

    def execute(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.pace.fused_add_rmsnorm(x, residual, weight, eps)


@backend_registry.register(
    OperatorType.RMSNORM_GATED,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITRMSNormGated(BackendBase):
    """Fused RMSNorm + SiLU gate: ``(weight * rmsnorm(x)) * silu(gate)``."""

    def execute(
        self,
        input: torch.Tensor,
        gate: torch.Tensor,
        _normalized_shape,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return torch.ops.pace.rmsnorm_gated(input, gate, weight, eps)


@backend_registry.register(
    OperatorType.GATED_DELTANET_RECURRENCE,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITGatedDeltaRecurrence(BackendBase):
    """Gated delta recurrence on the AVX-512 kernels.

    Dispatches like the NATIVE backend: a cached state plus a single query
    token runs the fused decode step, anything else runs the sequential
    prefill scan. ``query``/``key`` stay UNEXPANDED -- the kernels broadcast
    each key/query head across its group of value heads, so grouped-query
    models avoid the repeat_interleave copy. Both kernels advance ``state``
    in place and return it as the new state."""

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
        num_groups: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is not None and query.shape[1] == 1:
            state = state.to(torch.float32).contiguous()
            out = torch.ops.pace.recurrent_gated_delta_rule(
                query, key, value, g, beta, state
            )
            return out, state

        if state is None:
            state = torch.zeros(
                query.shape[0],
                value.shape[2],
                query.shape[-1],
                value.shape[-1],
                dtype=torch.float32,
                device=value.device,
            )
        else:
            state = state.to(torch.float32).contiguous()
        out = torch.ops.pace.chunk_gated_delta_rule(query, key, value, g, beta, state)
        return out, state


@backend_registry.register(
    OperatorType.GATED_DELTANET_CONV1D,
    BackendType.JIT,
    [DataType.BFLOAT16],
)
class JITCausalConv1dUpdate(BackendBase):
    """Depthwise causal conv1d + SiLU.

    A cached state plus a single token runs the in-place decode update;
    anything else (prefill and chunked continuation) runs the token-major
    prefill kernel, which also returns the next conv state. ``conv_state`` is
    shifted in place through a raw pointer, so it is made contiguous first."""

    def execute(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conv_state is not None and hidden_states.shape[-1] == 1:
            conv_state = conv_state.contiguous()
            out = torch.ops.pace.causal_conv1d_update(hidden_states, conv_state, weight)
            return out, conv_state
        return torch.ops.pace.causal_conv1d_prefill(hidden_states, conv_state, weight)
