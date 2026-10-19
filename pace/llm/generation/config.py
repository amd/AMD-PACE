# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from typing import Dict, Literal, Type

import torch
from pydantic import BaseModel, ConfigDict

_CONFIG_TYPES: Dict[str, Type["GenerationBackendConfig"]] = {}


class GenerationBackendConfig(BaseModel, ABC):
    """Typed configuration that owns construction of its generation backend."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    @abstractmethod
    def initialize_backend(self, model_name_or_path, *, serving=False, **kwargs):
        """Construct this config's backend for offline or serving use."""


def register_generation_backend_config(
    config_type: Type[GenerationBackendConfig],
) -> None:
    """Register a concrete config using its ``type`` field as the key."""
    type_field = config_type.model_fields.get("type")
    if type_field is None:
        raise TypeError("GenerationBackendConfig must declare a type field")
    backend_type = type_field.default
    if not isinstance(backend_type, str) or not backend_type:
        raise TypeError(
            "GenerationBackendConfig type must have a non-empty string default"
        )
    existing = _CONFIG_TYPES.get(backend_type)
    if existing is not None and existing is not config_type:
        raise TypeError(
            f"generation backend type '{backend_type}' is already registered"
        )
    _CONFIG_TYPES[backend_type] = config_type


class NativeGenerationBackendConfig(GenerationBackendConfig):
    """Configuration for the built-in native generation backend."""

    type: Literal["native"] = "native"

    def initialize_backend(
        self,
        model_name_or_path,
        *,
        serving=False,
        tokenizer_name_or_path=None,
        dtype=torch.bfloat16,
        kv_cache_type=None,
        spec_config=None,
        opconfig=None,
        disable_tqdm=False,
        cache_kwargs=None,
        eviction_config=None,
    ):
        from pace.llm.generation.native import NativeGenerationBackend

        constructor = (
            NativeGenerationBackend.for_serving
            if serving
            else NativeGenerationBackend.for_offline
        )
        return constructor(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=dtype,
            kv_cache_type=kv_cache_type,
            spec_config=spec_config,
            opconfig=opconfig,
            disable_tqdm=disable_tqdm,
            cache_kwargs=cache_kwargs,
            eviction_config=eviction_config,
        )


register_generation_backend_config(NativeGenerationBackendConfig)


def resolve_generation_backend_config(value) -> GenerationBackendConfig:
    """Resolve JSON at the serving boundary into a concrete backend config."""
    if isinstance(value, GenerationBackendConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("generation_backend_config must be a JSON object")
    backend_type = value.get("type", "native")
    try:
        config_type = _CONFIG_TYPES[backend_type]
    except KeyError as error:
        supported = ", ".join(sorted(_CONFIG_TYPES))
        raise ValueError(
            f"unknown generation backend type '{backend_type}'; "
            f"supported types: {supported}"
        ) from error
    return config_type.model_validate(value)
