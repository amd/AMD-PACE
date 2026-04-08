# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union, Tuple, List

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


class KVCacheManager(CacheContext):
    """Manages key-value caches for multiple layers (contiguous backend)."""

    def __init__(
        self,
        config: PretrainedConfig,
        max_seq_length: int,
        cache_type: KVCacheType,
        token: str = "",
    ):
        from pace.llm.attention import get_kv_cache_class

        self.token = token
        self.num_layers = config.num_hidden_layers
        self.cache_type = cache_type
        self.kv_cache_class = get_kv_cache_class(self.cache_type)
        self.cache_objects: List[KVCacheBase] = [
            self.kv_cache_class(max_seq_length) for _ in range(self.num_layers)
        ]
        for obj in self.cache_objects:
            obj.token = token

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
        K: torch.Tensor,
        V: torch.Tensor,
        kv_cache,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: [BS, q_len, num_heads/num_kv_heads, head_dim] (BSHD).
                Already projected and RoPE'd by the model.
            positions: [BS, q_len] absolute position indices.  Used to
                detect left-padding and build causal+padding masks internally.
        Returns:
            [BS, q_len, num_heads, head_dim] (BSHD).
        """
        ...
