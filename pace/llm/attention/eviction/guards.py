# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace.llm.attention.base import KVCacheType
from pace.llm.attention.eviction.config import EvictionConfig


def model_rotary_embedding(model):
    return getattr(getattr(model, "model", None), "rotary_emb", None)


def _order_dependent_masks(model_config) -> str:
    if getattr(model_config, "sliding_window", None):
        return "sliding_window"
    if getattr(model_config, "attention_chunk_size", None):
        return "attention_chunk_size"
    for layer_type in getattr(model_config, "layer_types", None) or []:
        if layer_type in ("sliding_attention", "chunked_attention"):
            return layer_type
    return ""


def validate_eviction_setup(
    eviction_config: EvictionConfig,
    kv_cache_type: KVCacheType,
    model,
    spec_decoder=None,
) -> None:
    if kv_cache_type not in (KVCacheType.DYNAMIC, KVCacheType.BMC):
        raise NotImplementedError(
            f"KV eviction / EQUIP is only supported on contiguous caches "
            f"(DYNAMIC, BMC); got {kv_cache_type}."
        )

    if spec_decoder is not None:
        raise NotImplementedError(
            "KV eviction cannot be combined with speculative decoding because "
            "in-place eviction cannot roll back rejected draft tokens."
        )

    if model_rotary_embedding(model) is None:
        raise NotImplementedError(
            f"KV eviction is not supported for {type(model).__name__}: it does "
            f"not use rotary position embeddings."
        )

    mask = _order_dependent_masks(model.config)
    if mask:
        raise NotImplementedError(
            f"KV eviction is not supported for models using {mask}: that mask "
            f"depends on the physical KV order."
        )

    if not getattr(model, "supports_eviction_rope", False):
        raise NotImplementedError(
            f"{type(model).__name__} does not support KV eviction: only "
            f"Llama-style Neox RoPE is implemented."
        )

    text = getattr(model.config, "text_config", None) or model.config
    if getattr(text, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            f"KV eviction does not support models with KV-sharing layers "
            f"(num_kv_shared_layers); got {type(model).__name__}."
        )
