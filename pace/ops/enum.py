# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import List

try:
    from enum import auto, StrEnum
except ImportError:
    # Backport for Python versions < 3.11
    # This is needed for compatibility with older Python versions
    # where StrEnum is not available in the enum module.
    from backports.strenum import StrEnum
    from enum import auto

import torch


class OperatorType(StrEnum):
    LINEAR = auto()
    REPEATEDKVLINEAR = auto()
    SOFTMAX = auto()
    RELU = auto()
    GELU = auto()
    SILU = auto()
    TANH = auto()
    SIGMOID = auto()
    LAYERNORM = auto()
    RMSNORM = auto()
    ROTARYEMBEDDING = auto()


class FusedOperatorType(StrEnum):
    FUSEDMLPLINEAR = auto()
    FUSEDLINEARRELU = auto()
    FUSEDLINEARGELU = auto()
    FUSEDLINEARSILU = auto()
    FUSEDLINEARMUL = auto()
    FUSED_RMSNORM_RESIDUAL = auto()
    FUSED_LAYERNORM_RESIDUAL = auto()


class BackendType(StrEnum):
    NATIVE = auto()
    JIT = auto()
    TPP = auto()
    IMBPS = auto()
    AOCLDLP = auto()


class DataType(StrEnum):
    FLOAT32 = auto()
    BFLOAT16 = auto()

    @classmethod
    def from_torch(self, dtype):
        if dtype == torch.float32:
            return DataType.FLOAT32
        elif dtype == torch.bfloat16:
            return DataType.BFLOAT16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def to_torch_dtype(self):
        if self == DataType.FLOAT32:
            return torch.float32
        elif self == DataType.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self}")


# If in case the backend is not supported, fallback to these backends
# The order of the backends in this list is important. The first backend that
# supports the operator will be used.
FALLBACK_BACKEND: List[BackendType] = [
    BackendType.NATIVE,
    BackendType.JIT,
    BackendType.TPP,
    BackendType.IMBPS,
    BackendType.AOCLDLP,
]
