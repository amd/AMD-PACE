# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from pydantic import BaseModel, ConfigDict, Field
from typing import Union, Optional
from typing import Dict, Any

from uuid import UUID
import torch


class PrefillRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt text")
    request_id: Union[str, UUID, int] = Field(
        ..., description="Unique request identifier"
    )
    gen_config: Dict[str, Any] = Field(
        default_factory=dict, description="Generation configuration parameters"
    )
    mlperf_mode: bool = Field(
        default=False, description="Enable MLPerf mode (skip text decoding)"
    )

    class Config:
        # Allow extra fields if needed
        extra = "forbid"  # or "allow" if you want flexibility


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    modelId: str
    dataType: str = "bf16"
    kvCacheType: str = "BMC"
    norm_backend: str = "NATIVE"
    qkv_projection_backend: str = "TPP"
    attention_backend: str = "JIT"
    out_projection_backend: str = "TPP"
    mlp_backend: str = "TPP"
    lm_head_backend: str = "NATIVE"
    spec_config: Optional[Dict[str, Any]] = None
    kv_cache_memory_gb: Optional[float] = None


class ServerConfig(BaseModel):
    modelConfig: ModelConfig


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
