# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RMS Norm has been adapted from the original implementation in
# https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/llama/modeling_llama.py#L59

import numbers
from typing import Optional, Union, List

import torch
from torch import Size
from torch.nn import Parameter

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, FusedOperatorType, BackendType, DataType


class LayerNorm(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.LAYERNORM

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias_available = bias
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        weight = None
        bias = None
        if elementwise_affine:
            weight = Parameter(
                torch.empty(normalized_shape, dtype=self.dtype.to_torch_dtype()),
                requires_grad=False,
            )
            if self.bias_available:
                bias = Parameter(
                    torch.empty(normalized_shape, dtype=self.dtype.to_torch_dtype()),
                    requires_grad=False,
                )

        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)

        self._forward_impl = self._forward_fallback
        if self.backend is not None and self.weight is not None:
            self._forward_impl = self._forward_backend

    def _forward_backend(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.bias_available}, "
            f"dtype={self.dtype}, backend_impl={self.backend}"
        )


class RMSNorm(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.RMSNORM

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        weight = Parameter(
            torch.ones(normalized_shape, dtype=self.dtype.to_torch_dtype()),
            requires_grad=False,
        )
        self.register_parameter("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"dtype={self.dtype}, backend_impl={self.backend}"
        )


class Gemma3RMSNorm(RMSNorm):
    """
    RMS Normalization variant for Gemma 3 that uses (1 + weight) scaling.
    Weight is initialized to zeros so (1 + weight) = 1 at initialization.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(
            normalized_shape, eps=eps, dtype=dtype, backend_impl=backend_impl
        )
        # Re-initialize weight to zeros so (1 + weight) = 1 at start
        weight = Parameter(
            torch.zeros(self.normalized_shape, dtype=self.dtype.to_torch_dtype()),
            requires_grad=False,
        )
        self.register_parameter("weight", weight)
        self.cached_weight = None

    def _cached_weight(self) -> torch.Tensor:
        if self.cached_weight is None:
            self.cached_weight = self.weight + 1.0
        return self.cached_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            x, self.normalized_shape, self._cached_weight(), self.eps
        )


class FusedLayerNormResidual(OperatorBase):
    """Fused residual add + LayerNorm: residual_out = x + residual,
    normed = layernorm(residual_out)"""

    @property
    def operator_type(self):
        return FusedOperatorType.FUSED_LAYERNORM_RESIDUAL

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias_available = bias
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        weight = None
        bias_param = None
        if elementwise_affine:
            weight = Parameter(
                torch.empty(normalized_shape, dtype=self.dtype.to_torch_dtype()),
                requires_grad=False,
            )
            if self.bias_available:
                bias_param = Parameter(
                    torch.empty(normalized_shape, dtype=self.dtype.to_torch_dtype()),
                    requires_grad=False,
                )

        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias_param)

        self._forward_impl = self._forward_fallback
        if self.backend is not None and self.weight is not None:
            self._forward_impl = self._forward_backend

    def _forward_backend(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.execute(x, residual, self.weight, self.bias, self.eps)

    def _forward_fallback(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = torch.nn.functional.layer_norm(
            residual_out, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return normed, residual_out

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(x, residual)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"dtype={self.dtype}, backend_impl={self.backend}"
        )


class FusedRMSNormResidual(OperatorBase):
    """Fused residual add + RMSNorm: residual_out = x + residual,
    normed = rmsnorm(residual_out)"""

    @property
    def operator_type(self):
        return FusedOperatorType.FUSED_RMSNORM_RESIDUAL

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        weight = Parameter(
            torch.ones(normalized_shape, dtype=self.dtype.to_torch_dtype()),
            requires_grad=False,
        )
        self.register_parameter("weight", weight)

        self._forward_impl = self._forward_fallback
        if self.backend is not None:
            self._forward_impl = self._forward_backend

    def _forward_backend(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.execute(x, residual, self.weight, self.eps)

    def _forward_fallback(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = torch.nn.functional.rms_norm(
            residual_out, self.normalized_shape, self.weight, self.eps
        )
        return normed, residual_out

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(x, residual)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"dtype={self.dtype}, backend_impl={self.backend}"
        )


class FusedGemma3RMSNormResidual(FusedRMSNormResidual):
    """Fused residual add + Gemma3 RMSNorm using (1 + weight) scaling.
    Weight is initialized to zeros so (1 + weight) = 1 at initialization."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(
            normalized_shape, eps=eps, dtype=dtype, backend_impl=backend_impl
        )
        weight = Parameter(
            torch.zeros(self.normalized_shape, dtype=self.dtype.to_torch_dtype()),
            requires_grad=False,
        )
        self.register_parameter("weight", weight)
        self._cached_weight = None

    def _get_weight(self) -> torch.Tensor:
        if self._cached_weight is None:
            self._cached_weight = self.weight + 1.0
        return self._cached_weight

    def _forward_backend(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.execute(x, residual, self._get_weight(), self.eps)

    def _forward_fallback(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = torch.nn.functional.rms_norm(
            residual_out, self.normalized_shape, self._get_weight(), self.eps
        )
        return normed, residual_out
