# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conlist,
    field_validator,
    model_validator,
)
from typing import List, Union, Optional
from typing import Dict, Any

from uuid import UUID

from pace.llm.generation.config import (
    GenerationBackendConfig,
    resolve_generation_backend_config,
)


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
    kvCacheType: Optional[str] = None
    norm_backend: str = "NATIVE"
    qkv_projection_backend: str = "TPP"
    attention_backend: str = "SLAB"
    gated_deltanet_backend: str = "NATIVE"
    out_projection_backend: str = "TPP"
    mlp_backend: str = "TPP"
    lm_head_backend: str = "NATIVE"
    spec_config: Optional[Dict[str, Any]] = None
    kv_cache_memory_gb: Optional[float] = None
    generation_backend_config: Optional[GenerationBackendConfig] = None

    @field_validator("generation_backend_config", mode="before")
    @classmethod
    def _resolve_generation_backend_config(cls, value):
        if value is None:
            return None
        return resolve_generation_backend_config(value)


class ServerConfig(BaseModel):
    modelConfig: ModelConfig
