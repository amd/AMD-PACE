# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union, Tuple, List

import torch
from torch import nn
from transformers import PretrainedConfig

try:
    from enum import auto, StrEnum
except ImportError:
    from backports.strenum import StrEnum
    from enum import auto


class AttentionType(StrEnum):
    """Internal dispatch: MHA vs GQA. Used by backends, not by models."""

    MHA = auto()
    GQA = auto()


class AttentionBackendType(StrEnum):
    """Attention-specific backend selection."""

    JIT = auto()
    NATIVE = auto()
    SLAB = auto()
    PAGED = auto()


class KVCacheType(Enum):
    """Types of key-value caches."""

    DYNAMIC = "dynamic"
    BMC = "bmc"
    SLAB_POOL = "slab_pool"
    PAGED = "paged"

    @staticmethod
    def get_kv_cache_type(cache_type: Union[str, "KVCacheType"]) -> "KVCacheType":
        if isinstance(cache_type, str):
            return KVCacheType(cache_type)
        return cache_type


class KVCacheBase(ABC):
    """Abstract base class for key-value caches."""

    def __init__(self):
        pass

    @abstractmethod
    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def remove_cache(self, remove_len: int) -> None:
        pass


class CacheContext(ABC):
    """Per-sequence cache handle passed to model forward as kv_cache."""

    @abstractmethod
    def __getitem__(self, layer_idx: int) -> Any: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def remove_cache(self, remove_len: int) -> None: ...


class Cache(ABC):
    """Engine-level cache backend. Created once at startup."""

    supports_packing = False

    def reserve_pool(self) -> None:
        """Allocate serving capacity before the first request. Caches that allocate per request do nothing."""

    def prepare_batch(
        self,
        max_seq_length: int,
        batch_size: int,
        spec_headroom: int = 0,
    ) -> None:
        """Reserve capacity before request contexts are created."""

    def get_stats(self) -> dict:
        """Pool and prefix-cache statistics, empty for caches that keep none."""
        return {}

    @abstractmethod
    def create_context(
        self, config: PretrainedConfig, max_seq_length: int, **kwargs
    ) -> CacheContext: ...

    def remove_context(self, context: CacheContext) -> None:
        """Release resources held by a context.

        Called when a request is done. The Cache backend that created
        the context is responsible for freeing any pool blocks or
        other resources. Default is a no-op (contiguous caches rely
        on GC). Pooled backends (SLAB, paged) override to return
        blocks to the shared pool.
        """
        pass


class _ContiguousKVSharingView:
    """Read-only alias of a source layer's contiguous KV cache (KV sharing).

    A layer with a ``kv_sharing_target`` owns one of these instead of its own
    cache: it proxies the source's key/value and discards anything the backend
    tries to append.
    """

    def __init__(self, source: KVCacheBase, token: str = ""):
        self._src = source
        self.token = token

    @property
    def key(self):
        return self._src.key

    @property
    def value(self):
        return self._src.value

    def update_cache(self, key_states, value_states, concat_dim):
        return self._src.key, self._src.value

    @property
    def seq_len(self):
        return self._src.seq_len

    def remove_cache(self, remove_len: int) -> None:
        # The source cache is removed via its own entry; this view is a no-op.
        pass


class KVCacheManager(CacheContext):
    """Manages key-value caches for multiple layers (contiguous backend).

    KV sharing is resolved here (not in models): layers whose spec sets a
    ``kv_sharing_target`` get a read-only view over the target layer's cache
    instead of their own storage.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_seq_length: int,
        cache_type: KVCacheType,
        token: str = "",
    ):
        from pace.llm.attention import get_kv_cache_class
        from pace.llm.attention.specs import compute_layer_kv_specs
        from pace.llm.attention.recurrent_state import (
            RecurrentStateStore,
            num_recurrent_layers,
        )

        self.token = token
        self.cache_type = cache_type
        self.kv_cache_class = get_kv_cache_class(self.cache_type)

        specs = compute_layer_kv_specs(config)
        self.num_layers = len(specs)
        cache_objects: List[object] = [None] * self.num_layers
        for layer_idx, spec in enumerate(specs):
            if spec.kv_sharing_target is None:
                obj = self.kv_cache_class(max_seq_length)
                obj.token = token
                cache_objects[layer_idx] = obj
        for layer_idx, spec in enumerate(specs):
            if spec.kv_sharing_target is not None:
                cache_objects[layer_idx] = _ContiguousKVSharingView(
                    cache_objects[spec.kv_sharing_target], token=token
                )
        self.cache_objects: List[KVCacheBase] = cache_objects

        # Per-request Gated DeltaNet (linear-attention) recurrent/conv state.
        # Empty for full-attention-only models. Owned by the request context so
        # it batches and is released with the KV cache (online serving).
        self.recurrent_state = RecurrentStateStore(num_recurrent_layers(config))

    def __len__(self) -> int:
        return int(self.cache_objects[0].seq_len)

    def __getitem__(self, idx: int) -> KVCacheBase:
        return self.cache_objects[idx]

    def remove_cache(self, remove_len: int):
        for cache_object in self.cache_objects:
            cache_object.remove_cache(remove_len)


class AttentionBackend(ABC, nn.Module):
    """Abstract base class for attention backends."""

    @abstractmethod
    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor],
        V: Optional[torch.Tensor],
        kv_cache,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: [BS, q_len, num_heads/num_kv_heads, head_dim] (BSHD).
                Already projected and RoPE'd by the model. K and V are None for
                a layer that reuses another layer's KV (``kv_sharing_target``):
                there is nothing to store, and the layer attends against the
                target layer's cached KV.
            positions: [BS, q_len] absolute position indices.  Used to
                detect left-padding and build causal+padding masks internally.
        Returns:
            [BS, q_len, num_heads, head_dim] (BSHD).
        """
        ...
