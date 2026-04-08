# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

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
