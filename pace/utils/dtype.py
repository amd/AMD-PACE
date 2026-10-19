# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch


class TorchDtypeResolver:
    """
    Utility class to resolve string representations of data types to corresponding
    PyTorch dtype objects.

    This class provides a mapping from common string names (such as "bf16", "fp16", "float32")
    to their respective `torch.dtype` objects. It is useful for converting user or configuration
    input into the correct PyTorch dtype for model initialization or tensor operations.

    Usage:
        dtype = TorchDtypeResolver.resolve("bf16")  # returns torch.bfloat16

    Raises:
        ValueError: If the provided string does not correspond to a supported dtype.
    """

    _dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    @classmethod
    def resolve(cls, dtype_str: str):
        key = dtype_str.lower()
        if key not in cls._dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return cls._dtype_map[key]
