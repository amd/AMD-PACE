# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from pydantic import BaseModel, ConfigDict, Field, conlist, model_validator
from typing import List, Union, Optional
from typing import Dict, Any

from uuid import UUID
import torch


class PrefillItem(BaseModel):
    """A single prefill entry. Reused by ``/step`` and ``/step_auto_mixed``."""

    req_id: str
    prompt: List[int] = Field(default_factory=list)
    generation_config: Optional[Dict[str, Any]] = None
    is_prefill: bool = True

    class Config:
        extra = "ignore"


class PrefillRequest(BaseModel):
    """Internal engine prefill interface (used by ``ModelExecutor.prefill``)."""

    prompt: List[int] = Field(..., min_length=1)
    request_id: Union[str, UUID]
    gen_config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class StepRequest(BaseModel):
    """Schema for ``POST /step``."""

    is_decode: bool = False
    prefill_batch: Optional[List[PrefillItem]] = None

    class Config:
        extra = "ignore"


class MixedStepRequest(BaseModel):
    """Schema for ``POST /step_auto_mixed``."""

    prefill_batch: List[PrefillItem] = Field(default_factory=list)
    chunk_info_map: Dict[str, conlist(int, min_length=2, max_length=2)] = Field(
        default_factory=dict
    )

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def _validate_chunks(self) -> "MixedStepRequest":
        for rid, (start, size) in self.chunk_info_map.items():
            if start < 0:
                raise ValueError(
                    f"chunk_info_map[{rid}]: chunk_start must be >= 0 (got {start})"
                )
            if size <= 0:
                raise ValueError(
                    f"chunk_info_map[{rid}]: chunk_size must be > 0 (got {size})"
                )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    modelId: str
    dataType: str = "bf16"
    kvCacheType: str = "SLAB_POOL"
    norm_backend: str = "NATIVE"
    qkv_projection_backend: str = "TPP"
    attention_backend: str = "SLAB"
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
