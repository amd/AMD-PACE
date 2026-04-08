# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************


from typing import Optional

import torch

from pace.ops.linear import Linear
from pace.ops.base import OperatorBase
from pace.utils.logging import PACE_ASSERT
from pace.ops.activations import Activation
from pace.ops.enum import FusedOperatorType, BackendType, DataType


class FusedLinearActivation(OperatorBase):
    """
    Fused Linear ReLU operator.
    """

    _supported_activations = ["relu", "gelu", "gelu_new", "gelu_pytorch_tanh", "silu"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "relu",
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        PACE_ASSERT(
            activation in self._supported_activations,
            f"Unsupported activation function: {activation}. in FusedLinearActivation. "
            f"Supported functions are: {', '.join(self._supported_activations)}.",
        )

        super().__init__(backend_impl=backend_impl, dtype=dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.bias_available = bias

        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl if self.backend is None else None,
        )
        self.activation = Activation(
            activation,
            dtype=dtype,
            backend_impl=backend_impl if self.backend is None else None,
        )

        self._forward_impl = self._forward_impl_fallback
        if self.backend is not None:
            self._forward_impl = self._forward_impl_backend

    def _forward_impl_fallback(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

    def _forward_impl_backend(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            x,
            self.linear.weight,
            self.linear.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_available}, "
            f"dtype={self.dtype}, "
            f"backend_impl={self.backend if self.backend is not None else self.linear.backend}"
        )


class FusedLinearRelu(FusedLinearActivation):

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDLINEARRELU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation="relu",
            dtype=dtype,
            backend_impl=backend_impl,
        )


class FusedLinearGelu(FusedLinearActivation):

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDLINEARGELU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation="gelu",
            dtype=dtype,
            backend_impl=backend_impl,
        )


class FusedLinearSiLU(FusedLinearActivation):

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDLINEARSILU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation="silu",
            dtype=dtype,
            backend_impl=backend_impl,
        )


class FusedLinearMul(OperatorBase):
    """
    Fused Linear Mul operator.
    """

    @property
    def operator_type(self):
        return FusedOperatorType.FUSEDLINEARMUL

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DataType] = None,
        backend_impl: BackendType = BackendType.NATIVE,
    ):
        super().__init__(backend_impl=backend_impl, dtype=dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.bias_available = bias

        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl if self.backend is None else None,
        )

        self._forward_impl = self._forward_impl_fallback
        if self.backend is not None:
            self._forward_impl = self._forward_impl_backend

    def _forward_impl_fallback(
        self, x: torch.Tensor, mul: torch.Tensor
    ) -> torch.Tensor:
        out = self.linear(x)
        return out * mul

    def _forward_impl_backend(self, x: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            x,
            mul,
            self.linear.weight,
            self.linear.bias,
        )

    def forward(self, x: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, mul)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_available}, "
            f"dtype={self.dtype}, "
            f"backend_impl={self.backend if self.backend is not None else self.linear.backend}"
        )


def _get_fused_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    activation: Optional[str] = None,
    dtype: Optional[DataType] = None,
    backend_impl: Optional[BackendType] = BackendType.NATIVE,
) -> OperatorBase:
    """
    Create a fused linear operator based on the specified activation function.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        activation (str, optional): Activation function to use. Options are "relu", "gelu", "silu", or None.
        gate (bool, optional): Whether this specific operator is a gate. Defaults to False.
        dtype (Optional[DataType], optional): Data type for the operator. Defaults to None.
        backend_impl (BackendType, optional): Backend implementation to use. Defaults to BackendType.NATIVE.

    Returns:
        OperatorBase: The created fused linear operator.
    """
    # If activation is None, return a standard Linear operator
    if activation is None:
        return Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )

    # If activation is not None, return a fused linear operator
    # with the specified activation function
    if activation == "relu":
        return FusedLinearRelu(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )
    elif activation in ("gelu", "gelu_new", "gelu_pytorch_tanh"):
        return FusedLinearGelu(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )
    elif activation == "silu":
        return FusedLinearSiLU(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )
    elif activation == "mul":
        return FusedLinearMul(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            backend_impl=backend_impl,
        )
    else:
        PACE_ASSERT(
            False,
            f"Unsupported activation function: {activation}. Supported"
            " functions are: relu, gelu, gelu_new, gelu_pytorch_tanh, silu, mul.",
        )
