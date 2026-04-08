# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
from typing import Optional

import torch
from torch import nn

from pace.ops.base import OperatorBase
from pace.ops.enum import OperatorType, BackendType, DataType


class SoftMax(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.SOFTMAX

    def __init__(
        self,
        dim: Optional[int] = None,
        dtype: Optional[DataType] = None,
        backend_impl=BackendType.NATIVE,
    ):

        self.dim = dim
        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(
            input, dim=self.dim, dtype=self.dtype.to_torch_dtype()
        )

    def extra_repr(self):
        return f"dim={self.dim}, dtype={self.dtype}, backend={self.backend}"


class ReLU(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.RELU

    def __init__(
        self,
        inplace=False,
        dtype=None,
        backend_impl=BackendType.NATIVE,
    ):

        self.inplace = inplace
        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(input, inplace=self.inplace)

    def extra_repr(self):
        return f"inplace={self.inplace}, dtype={self.dtype}, " f"backend={self.backend}"


class GeLU(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.GELU

    def __init__(
        self,
        approximate: str = "none",
        dtype=None,
        backend_impl=BackendType.NATIVE,
    ):

        self.approximate = approximate
        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(input, approximate=self.approximate)

    def extra_repr(self):
        return (
            f"approximate={self.approximate}, dtype={self.dtype}, "
            f"backend={self.backend}"
        )


class SiLU(OperatorBase):

    @property
    def operator_type(self):
        return OperatorType.SILU

    def __init__(
        self,
        inplace=False,
        dtype=None,
        backend_impl=BackendType.NATIVE,
    ):

        self.inplace = inplace
        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(input, inplace=self.inplace)

    def extra_repr(self):
        return f"inplace={self.inplace}, dtype={self.dtype}, " f"backend={self.backend}"


class Tanh(OperatorBase):
    @property
    def operator_type(self):
        return OperatorType.TANH

    def __init__(
        self,
        dtype=None,
        backend_impl=BackendType.NATIVE,
    ):

        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(input)

    def extra_repr(self):
        return f"dtype={self.dtype}, backend={self.backend}"


class Sigmoid(OperatorBase):
    @property
    def operator_type(self):
        return OperatorType.SIGMOID

    def __init__(
        self,
        dtype=None,
        backend_impl=BackendType.NATIVE,
    ):

        self.dtype = dtype
        super().__init__(backend_impl=backend_impl, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.execute(input)

    def extra_repr(self):
        return f"dtype={self.dtype}, backend={self.backend}"


# Wrapper for all activation classes since HF
# uses a name based approach to get the activation function
class Activation(nn.Module):
    def __init__(self, act_type: str, **kwargs):
        super().__init__()
        self.act_type = act_type
        self.kwargs = kwargs

        if act_type == "relu":
            self.act = ReLU(**kwargs)
        elif act_type == "gelu":
            self.act = GeLU(**kwargs)
        elif act_type in ("gelu_new", "gelu_pytorch_tanh"):
            # gelu_pytorch_tanh is the same as gelu_new (GELU with tanh approximation)
            self.act = GeLU(approximate="tanh", **kwargs)
        elif act_type == "silu":
            self.act = SiLU(**kwargs)
        elif act_type == "tanh":
            self.act = Tanh(**kwargs)
        elif act_type == "sigmoid":
            self.act = Sigmoid(**kwargs)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
