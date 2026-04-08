# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Union, Optional

import torch
from torch import nn

from .base import (
    AttentionBackend,
    AttentionBackendType,
    AttentionType,
    Cache,
    CacheContext,
    KVCacheType,
    KVCacheBase,
    KVCacheManager,
)
from .contiguous.backend import ContiguousAttentionBackend
from .contiguous.cache import BMCKVCache, DynamicKVCache, ContiguousCache
from .slab.backend import SlabAttentionBackend
from .slab.cache import (
    SlabCache,
    SlabPoolManager,
    SlabPoolContext,
    SlabPoolLayerView,
    create_slab_pool,
)
from .paged.backend import PagedAttentionBackend


def get_kv_cache_class(cache_type: Union[str, KVCacheType]) -> type:
    """Returns the KVCacheBase subclass for the given cache type."""
    cache_type = KVCacheType.get_kv_cache_type(cache_type)
    if cache_type == KVCacheType.DYNAMIC:
        return DynamicKVCache
    elif cache_type == KVCacheType.BMC:
        return BMCKVCache
    raise ValueError(f"Invalid cache type: {cache_type}")


def create_cache(cache_type: KVCacheType, model_config=None, **kwargs) -> Cache:
    """Create an engine-level cache backend for the given cache type.

    Args:
        cache_type: Which KV cache implementation to use.
        model_config: PretrainedConfig (required for SLAB_POOL/PAGED, ignored otherwise).
        **kwargs: Forwarded to the backend constructor (e.g. kv_cache_memory_gb).
    """
    if cache_type in (KVCacheType.DYNAMIC, KVCacheType.BMC):
        return ContiguousCache(cache_type)
    elif cache_type == KVCacheType.SLAB_POOL:
        if model_config is None:
            raise ValueError("SLAB_POOL requires a model config at creation time")
        return SlabCache(model_config, **kwargs)
    elif cache_type == KVCacheType.PAGED:
        from .paged.cache import PagedCache

        if model_config is None:
            raise ValueError("PAGED requires a model config at creation time")
        return PagedCache(model_config, **kwargs)
    raise ValueError(f"Unknown cache type: {cache_type}")


class Attention(nn.Module):
    """
    Unified attention module that routes to the appropriate backend
    based on the operator configuration.

    Models create this once in __init__ and call it in forward().
    All dispatch decisions are made at construction time.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        opconfig,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        attn_backend = opconfig.attention

        if attn_backend in (AttentionBackendType.JIT, AttentionBackendType.NATIVE):
            self._backend = ContiguousAttentionBackend(
                num_heads,
                num_kv_heads,
                head_dim,
                attn_backend,
                sliding_window=sliding_window,
                sinks=sinks,
            )
        elif attn_backend == AttentionBackendType.SLAB:
            self._backend = SlabAttentionBackend(
                num_heads,
                num_kv_heads,
                head_dim,
                attn_backend,
                sliding_window=sliding_window,
                sinks=sinks,
            )
        elif attn_backend == AttentionBackendType.PAGED:
            self._backend = PagedAttentionBackend(
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window=sliding_window,
                sinks=sinks,
                scale=scale,
            )
        else:
            raise ValueError(f"Unknown attention backend: {attn_backend}")

    def forward(self, Q, K, V, kv_cache, positions, **kwargs):
        return self._backend(Q, K, V, kv_cache, positions, **kwargs)


__all__ = [
    "Attention",
    "AttentionBackend",
    "AttentionBackendType",
    "AttentionType",
    "Cache",
    "CacheContext",
    "ContiguousCache",
    "KVCacheType",
    "KVCacheBase",
    "KVCacheManager",
    "create_cache",
    "get_kv_cache_class",
    "ContiguousAttentionBackend",
    "BMCKVCache",
    "DynamicKVCache",
    "SlabAttentionBackend",
    "SlabCache",
    "SlabPoolManager",
    "SlabPoolContext",
    "SlabPoolLayerView",
    "create_slab_pool",
    "PagedAttentionBackend",
]
