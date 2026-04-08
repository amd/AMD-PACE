# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import math
from typing import Optional, Tuple

import torch

from pace.llm.attention.base import KVCacheBase, KVCacheType, KVCacheManager, Cache
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_DEBUG


class BMCKVCache(KVCacheBase):
    """Key-value cache using Balancing Memory and Compute strategy."""

    def __init__(self, max_seq_length: int):
        super().__init__()
        num_splits = int(
            os.getenv("PACE_BMC_NUM_SPLITS", int(math.sqrt(max_seq_length)))
        )
        self.tokens_per_split = max_seq_length // num_splits
        self.key = None
        self.value = None
        self.seq_len = 0
        self.concat_dim = None

        PACE_LLM_DEBUG(
            f"BMCKVCache initialized with {num_splits} splits, "
            f"tokens per split: {self.tokens_per_split}, "
            f"max sequence length: {max_seq_length}"
        )

    def remove_cache(self, remove_len: int) -> None:
        if self.key is not None and self.value is not None:
            if remove_len > self.seq_len:
                raise ValueError("Cannot remove more tokens than available in cache.")
            self.seq_len -= remove_len

    def _create_new_segment(self, shape: Tuple[int], dtype: torch.dtype):
        if self.key is not None and self.value is not None:
            new_key = torch.empty(shape, dtype=dtype)
            new_value = torch.empty(shape, dtype=dtype)
            old_len = self.key.size(self.concat_dim)
            new_key.narrow(self.concat_dim, 0, old_len).copy_(self.key)
            new_value.narrow(self.concat_dim, 0, old_len).copy_(self.value)
            new_len = new_key.size(self.concat_dim)
            if new_len > old_len:
                new_key.narrow(self.concat_dim, old_len, new_len - old_len).zero_()
                new_value.narrow(self.concat_dim, old_len, new_len - old_len).zero_()
        else:
            new_key = torch.zeros(shape, dtype=dtype)
            new_value = torch.zeros(shape, dtype=dtype)

        self.key = new_key
        self.value = new_value

    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.concat_dim is None:
            self.concat_dim = concat_dim
        else:
            PACE_LLM_ASSERT(
                self.concat_dim == concat_dim,
                "concat_dim should be the same for all updates",
            )

        token_count = key_states.size(concat_dim)
        updated_seq_len = self.seq_len + token_count

        need_segment = (self.key is None or self.value is None) or (
            updated_seq_len > self.key.size(concat_dim)
        )
        if need_segment:
            PACE_LLM_DEBUG(
                f"Creating new segment for key and value tensors. "
                f"Updated sequence length: {updated_seq_len}, "
                f"Current allocated sequence length: "
                f"{self.key.size(concat_dim) if self.key is not None else 0}, "
                f"Segment index: {(updated_seq_len - 1) // self.tokens_per_split}"
            )
            segment_idx = (updated_seq_len - 1) // self.tokens_per_split
            new_shape = list(key_states.shape)
            new_shape[concat_dim] = (segment_idx + 1) * self.tokens_per_split
            self._create_new_segment(tuple(new_shape), key_states.dtype)

        self.key.narrow(concat_dim, self.seq_len, token_count).copy_(key_states)
        self.value.narrow(concat_dim, self.seq_len, token_count).copy_(value_states)
        self.seq_len = updated_seq_len
        return self.key, self.value


class DynamicKVCache(KVCacheBase):
    """Key-value cache using dynamic (torch.cat) allocation."""

    def __init__(self, max_seq_length):
        super().__init__()
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        self.seq_len = 0
        self.concat_dim = None

        PACE_LLM_DEBUG(
            f"DynamicKVCache initialized with max sequence length: {max_seq_length}"
        )

    def remove_cache(self, remove_len: int) -> None:
        if self.key is not None and self.value is not None:
            if remove_len > self.seq_len:
                raise ValueError("Cannot remove more tokens than available in cache.")
            self.seq_len -= remove_len
            self.key = self.key.narrow(self.concat_dim, 0, self.seq_len)
            self.value = self.value.narrow(self.concat_dim, 0, self.seq_len)

    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.concat_dim is None:
            self.concat_dim = concat_dim
        else:
            PACE_LLM_ASSERT(
                self.concat_dim == concat_dim,
                "concat_dim should be the same for all updates",
            )
        if self.key is not None and self.value is not None:
            self.key = torch.cat([self.key, key_states], dim=concat_dim)
            self.value = torch.cat([self.value, value_states], dim=concat_dim)
        else:
            self.key = key_states
            self.value = value_states
        self.seq_len = int(self.key.size(concat_dim))
        return self.key, self.value


class ContiguousCache(Cache):
    """Cache backend for contiguous KV caches (Dynamic, BMC)."""

    def __init__(self, cache_type: KVCacheType):
        self.cache_type = cache_type

    def create_context(self, config, max_seq_length, **kwargs) -> KVCacheManager:
        return KVCacheManager(
            config, max_seq_length, self.cache_type, token=kwargs.get("token", "")
        )

    def merge_contexts(self, contexts, query_len=1):
        """For contiguous caches, return single or list as-is."""
        return contexts[0] if len(contexts) == 1 else contexts

    def build_prefill_metadata(self, context, seq_len, past_len=0):
        """Contiguous caches don't need prefill metadata."""
        return None

    def remove_context(self, context):
        """Contiguous caches rely on GC."""
        pass
