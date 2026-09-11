# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
SlabPool: Global KV Cache Pool for Serving

Provides SlabPoolManager (multi-layer pool manager), SlabPoolContext
(duck-type compatible with KVCacheManager), and SlabPoolLayerView
(per-layer cache view for attention layers).

The underlying C++ pool is a torch::CustomClassHolder accessed via
torch.classes.pace.SlabPool. No global state map -- state is directly
on the class instance, ref-counted and garbage collected.
"""

import threading
from typing import Optional, List
import torch
from transformers import PretrainedConfig

from pace.llm.attention.base import (
    Cache,
    CacheContext,
)
from pace.llm.attention.prefix_cache import (
    PrefixCacheContextMixin,
    PrefixCacheMixin,
    compute_block_hashes,
)
from pace.utils.logging import PACE_LLM_INFO

_BF16_BYTES = 2


def autotune_block_size(num_kv_heads: int, head_dim: int) -> int:
    return int(torch.ops.pace.slab_autotune_block_size(num_kv_heads, head_dim))


def kv_cache_bytes_per_token(num_kv_heads: int, head_dim: int) -> int:
    return 2 * num_kv_heads * head_dim * _BF16_BYTES


def create_slab_pool(
    total_blocks: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 64,
):
    """
    Create a SlabPool instance (C++ custom class).

    Args:
        total_blocks: Number of blocks in the pool
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        block_size: Tokens per block (default 64, must be > 0)

    Returns:
        torch.classes.pace.SlabPool instance
    """
    return torch.classes.pace.SlabPool(total_blocks, num_kv_heads, head_dim, block_size)


class SlabPoolLayerView:
    """
    Per-layer view into SlabPoolManager.

    Received by the SlabAttentionBackend as the kv_cache argument.
    Delegates to the pool manager for the specific layer.
    """

    __slots__ = ("pool_manager", "tokens", "layer_idx", "query_lens")

    def __init__(
        self,
        pool_manager: "SlabPoolManager",
        tokens: List[str],
        layer_idx: int,
        query_lens: Optional[List[int]] = None,
    ):
        self.pool_manager = pool_manager
        self.tokens = tokens
        self.layer_idx = layer_idx
        self.query_lens = query_lens

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
    ) -> None:
        self.pool_manager.update_cache_batched(
            self.layer_idx, self.tokens, key, value, seq_lens=seq_lens
        )

    def attend(
        self,
        query: torch.Tensor,
        scale: float,
        seq_lens: Optional[List[int]] = None,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Unified attention: dispatches to decode/prefill in C++ based on q_len."""
        return self.pool_manager.attention(
            self.layer_idx,
            self.tokens,
            query,
            seq_lens=seq_lens,
            scale=scale,
            sliding_window=sliding_window,
            sinks=sinks,
            chunk_size=chunk_size,
        )

    @property
    def seq_len(self) -> int:
        if self.tokens:
            return self.pool_manager.get_sequence_length(self.tokens[0])
        return 0


class _SlabPoolLayerList:
    """
    Lazy indexable that returns SlabPoolLayerView per layer.
    Mimics KVCacheManager.cache_objects so model forward() can access
    kv_cache.cache_objects[layer_idx] without changes.
    """

    __slots__ = ("_pool_manager", "_tokens", "_query_lens")

    def __init__(
        self,
        pool_manager: "SlabPoolManager",
        tokens: List[str],
        query_lens: Optional[List[int]] = None,
    ):
        self._pool_manager = pool_manager
        self._tokens = tokens
        self._query_lens = query_lens

    def __getitem__(self, layer_idx: int) -> SlabPoolLayerView:
        return SlabPoolLayerView(
            self._pool_manager, self._tokens, layer_idx, self._query_lens
        )


class SlabPoolContext(PrefixCacheContextMixin, CacheContext):
    """
    Per-sequence cache context for SlabPool. Implements CacheContext ABC.

    Supports the same interface models expect:
    - cache_objects[layer_idx] -> SlabPoolLayerView
    - remove_cache(n) -> truncate sequences
    - len() -> current sequence length
    """

    def __init__(
        self,
        pool_manager: "SlabPoolManager",
        tokens: List[str],
    ):
        self.pool_manager = pool_manager
        self.tokens = tokens
        self._cache_objects = _SlabPoolLayerList(pool_manager, tokens)
        self.slab_query_lens: Optional[List[int]] = None
        self._init_prefix_fields()

    @property
    def cache_objects(self):
        self._cache_objects._query_lens = self.slab_query_lens
        return self._cache_objects

    def __getitem__(self, layer_idx: int) -> SlabPoolLayerView:
        return self.cache_objects[layer_idx]

    def __len__(self) -> int:
        if self.tokens:
            return self.pool_manager.get_sequence_length(self.tokens[0])
        return 0

    def remove_cache(self, remove_len: int) -> None:
        for token in self.tokens:
            self.pool_manager.truncate_sequence(token, remove_len)


class SlabPoolManager:
    """
    Multi-layer global KV cache pool manager for serving.

    Creates one SlabPool (C++ custom class) per transformer layer.
    Manages sequences via token-based string identifiers mapped
    to internal int64 sequence IDs.

    Usage:
        mgr = SlabPoolManager(config, max_total_tokens=32768)
        token = mgr.create_sequence("req-123")
        ctx = SlabPoolContext(mgr, [token])
        model.forward(input_ids, positions, ctx, mask)
        mgr.remove_sequence(token)
    """

    @staticmethod
    def compute_max_tokens_from_memory(
        kv_cache_memory_gb: float,
        bytes_per_token_all_layers: int,
    ) -> int:
        """
        Compute max_total_tokens from a memory budget.

        ``bytes_per_token_all_layers`` is the summed per-token cost of the
        layers that own storage, each contributing
          2 (K+V) * num_kv_heads * head_dim * sizeof(BF16)
        bytes. Layers that reuse another layer's KV contribute nothing.
        """
        total_bytes = int(kv_cache_memory_gb * 1024 * 1024 * 1024)
        return total_bytes // bytes_per_token_all_layers

    def __init__(
        self,
        config,
        kv_cache_memory_gb: Optional[float] = None,
        max_total_tokens: Optional[int] = None,
        block_size: int = 0,
        default_max_seq_len: Optional[int] = None,
    ):
        from pace.llm.attention.specs import (
            compute_layer_kv_specs,
            resolve_text_config,
        )

        text_config = resolve_text_config(config)
        self.num_kv_heads = getattr(
            text_config, "num_key_value_heads", text_config.num_attention_heads
        )
        self.num_q_heads = text_config.num_attention_heads
        self.head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        if block_size <= 0:
            block_size = autotune_block_size(self.num_kv_heads, self.head_dim)
        self.block_size = block_size

        self._specs = compute_layer_kv_specs(config)
        self.num_layers = len(self._specs)
        self._bytes_per_token_all_layers = sum(
            kv_cache_bytes_per_token(s.num_kv_heads, s.head_dim)
            for s in self._specs
            if s.kv_sharing_target is None
        )

        if max_total_tokens is not None:
            self.max_total_tokens = max_total_tokens
        elif kv_cache_memory_gb is not None:
            self.max_total_tokens = self.compute_max_tokens_from_memory(
                kv_cache_memory_gb, self._bytes_per_token_all_layers
            )
        else:
            max_pos = getattr(config, "max_position_embeddings", 4096)
            self.max_total_tokens = min(max_pos, 8192)

        if default_max_seq_len is not None:
            self.default_max_seq_len = default_max_seq_len
        else:
            self.default_max_seq_len = getattr(config, "max_position_embeddings", 4096)

        total_blocks = (self.max_total_tokens + block_size - 1) // block_size
        self.total_blocks = total_blocks

        pool_memory_gb = (self.max_total_tokens * self._bytes_per_token_all_layers) / (
            1024**3
        )

        num_shared = sum(1 for s in self._specs if s.kv_sharing_target is not None)
        PACE_LLM_INFO(
            f"SlabPoolManager: {self.max_total_tokens} max tokens, "
            f"{total_blocks} blocks, {pool_memory_gb:.2f} GB KV cache memory "
            f"({self.num_layers} layers, {num_shared} KV-shared, "
            f"block_size {block_size})"
        )

        # One pool per storage-owning layer; shared layers alias their target's.
        self._pools: List = [None] * self.num_layers
        self._owned_pools: List = []
        self._layer_readonly: List[bool] = [False] * self.num_layers
        self._layer_head_dim: List[int] = [0] * self.num_layers
        self._layer_num_kv_heads: List[int] = [0] * self.num_layers
        for layer_idx, spec in enumerate(self._specs):
            self._layer_head_dim[layer_idx] = spec.head_dim
            self._layer_num_kv_heads[layer_idx] = spec.num_kv_heads
            if spec.kv_sharing_target is None:
                pool = create_slab_pool(
                    total_blocks, spec.num_kv_heads, spec.head_dim, block_size
                )
                self._pools[layer_idx] = pool
                self._owned_pools.append(pool)
        for layer_idx, spec in enumerate(self._specs):
            if spec.kv_sharing_target is not None:
                self._pools[layer_idx] = self._pools[spec.kv_sharing_target]
                self._layer_readonly[layer_idx] = True

        self._lock = threading.Lock()
        self._token_to_seq_id: dict = {}
        self._seq_id_to_token: dict = {}
        self._next_seq_id = 0

    def create_sequence(
        self, max_seq_len: Optional[int] = None, token: Optional[str] = None
    ) -> str:
        with self._lock:
            if max_seq_len is None:
                max_seq_len = self.default_max_seq_len

            if token is None:
                import uuid

                token = str(uuid.uuid4())

            if token in self._token_to_seq_id:
                raise ValueError(f"Sequence token already exists: {token}")

            seq_id = self._next_seq_id
            self._next_seq_id += 1

            created_in = []
            try:
                for pool in self._owned_pools:
                    pool.create_sequence(seq_id, max_seq_len)
                    created_in.append(pool)
            except Exception:
                for pool in created_in:
                    pool.remove_sequence(seq_id)
                self._next_seq_id -= 1
                raise

            self._token_to_seq_id[token] = seq_id
            self._seq_id_to_token[seq_id] = token

            return token

    def remove_sequence(self, token: str) -> None:
        with self._lock:
            if token not in self._token_to_seq_id:
                return

            seq_id = self._token_to_seq_id[token]

            for pool in self._owned_pools:
                pool.remove_sequence(seq_id)

            del self._token_to_seq_id[token]
            del self._seq_id_to_token[seq_id]

    def get_internal_ids(self, tokens: List[str]) -> List[int]:
        with self._lock:
            return [self._token_to_seq_id[t] for t in tokens]

    def truncate_sequence(self, token: str, remove_len: int) -> None:
        with self._lock:
            seq_id = self._token_to_seq_id.get(token)
            if seq_id is None:
                return
            for pool in self._owned_pools:
                pool.truncate_sequence(seq_id, remove_len)

    def get_sequence_length(self, token: str) -> int:
        with self._lock:
            seq_id = self._token_to_seq_id.get(token)
            if seq_id is None or not self._pools:
                return 0
            return self._pools[0].get_sequence_length(seq_id)

    def get_active_tokens(self) -> List[str]:
        with self._lock:
            return list(self._token_to_seq_id.keys())

    def get_free_blocks(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._pools):
            return self._pools[layer_idx].get_free_block_count()
        return 0

    def get_stats(self) -> dict:
        """Pool stats mirroring the paged pool so downstream metrics
        (``pace_kv_cache_usage_perc``) work identically for slab."""
        free = self.get_free_blocks()
        allocated = self.total_blocks - free
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": free,
            "allocated_blocks": allocated,
            "utilization_pct": (
                (allocated / self.total_blocks * 100) if self.total_blocks > 0 else 0
            ),
        }

    def get_block_indices(self, token: str) -> List[int]:
        """Return the C++ physical block indices for a sequence."""
        with self._lock:
            seq_id = self._token_to_seq_id.get(token)
            if seq_id is None or not self._pools:
                return []
            return list(self._pools[0].get_block_indices(seq_id))

    def reserve_blocks(self, token: str, num_blocks: int) -> List[int]:
        """Eagerly allocate ``num_blocks`` blocks per layer, return their indices."""
        with self._lock:
            seq_id = self._token_to_seq_id.get(token)
            if seq_id is None or not self._owned_pools:
                return []
            for pool in self._owned_pools:
                pool.reserve_blocks(seq_id, num_blocks)
            return list(self._owned_pools[0].get_block_indices(seq_id))

    def assign_prefix_blocks(
        self, dst_token: str, pool_block_indices: List[int], num_tokens: int
    ) -> None:
        """Zero-copy: assign existing pool blocks to a sequence (all layers)."""
        with self._lock:
            dst_seq_id = self._token_to_seq_id.get(dst_token)
            if dst_seq_id is None:
                raise ValueError(f"Destination token {dst_token} not found")
            for pool in self._owned_pools:
                pool.assign_prefix_blocks(dst_seq_id, pool_block_indices, num_tokens)

    def reassign_block(self, token: str, block_pos: int, new_pool_block: int) -> None:
        """Repoint one logical block of a sequence at another pool block (all layers)."""
        with self._lock:
            seq_id = self._token_to_seq_id.get(token)
            if seq_id is None:
                return
            for pool in self._owned_pools:
                pool.reassign_block(seq_id, block_pos, new_pool_block)

    def retain_blocks(self, pool_block_indices: List[int]) -> int:
        """Mark blocks as retained across sequence removals (all layers)."""
        if not self._owned_pools:
            return 0
        count = 0
        for pool in self._owned_pools:
            count = pool.retain_blocks(pool_block_indices)
        return count

    def release_retained_blocks(self, pool_block_indices: List[int]) -> None:
        """Release previously retained blocks back to the free list (all layers)."""
        for pool in self._owned_pools:
            pool.release_retained_blocks(pool_block_indices)

    def update_cache_batched(
        self,
        layer_idx: int,
        tokens: List[str],
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
    ) -> None:
        if self._layer_readonly[layer_idx]:
            return
        seq_ids = self.get_internal_ids(tokens)
        self._pools[layer_idx].cache_update(seq_ids, keys, values, seq_lens or [])

    def attention(
        self,
        layer_idx: int,
        tokens: List[str],
        query: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
        scale: Optional[float] = None,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Unified attention: C++ dispatches to decode/prefill based on q_len."""
        if scale is None:
            scale = 1.0 / (self._layer_head_dim[layer_idx] ** 0.5)
        seq_ids = self.get_internal_ids(tokens)
        sinks_tensor = sinks if sinks is not None else torch.tensor([])
        return self._pools[layer_idx].attention(
            seq_ids,
            query,
            seq_lens or [],
            [],
            scale,
            sliding_window,
            sinks_tensor,
            chunk_size,
        )


class SlabCache(PrefixCacheMixin, Cache):
    """Engine-level cache backend for SlabPool.

    Implements the Cache ABC and mixes in PrefixCacheMixin for prefix
    caching (enabled by default; disable via ``PACE_PREFIX_CACHE=0`` or kwarg
    ``enable_prefix_cache=False``).

    SLAB reuses cached prefix blocks zero-copy: matched blocks are referenced
    by the new sequence via the C++ ``SlabPool.assign_prefix_blocks`` and kept
    alive across sequence removals via ``retain_blocks``, so the engine can
    skip recomputing the matched prefix.
    The prefix-hash block size is derived from SLAB's autotuned block size so
    hashing granularity matches the pool's physical blocks.

    When kv_cache_memory_gb is provided (server path), the SlabPoolManager
    is created eagerly so the pool is allocated before any requests.
    Otherwise (offline / spec-decode) the manager is created lazily in
    create_context() once max_seq_length and batch_size are known.
    """

    # kwargs accepted by SlabPoolManager; others are silently ignored.
    _SUPPORTED_KWARGS = {
        "kv_cache_memory_gb",
        "max_total_tokens",
        "block_size",
        "default_max_seq_len",
    }

    def __init__(self, config: PretrainedConfig, **kwargs):
        self._config = config
        self._kwargs = {k: v for k, v in kwargs.items() if k in self._SUPPORTED_KWARGS}
        self._manager: Optional[SlabPoolManager] = None
        if "kv_cache_memory_gb" in self._kwargs:
            self._manager = SlabPoolManager(config, **self._kwargs)

        self._resolved_block_size = self._init_prefix_cache_from_config(
            config, "SlabCache", autotune_fn=autotune_block_size, **kwargs
        )

    def _ensure_manager(
        self, max_seq_length: int, batch_size: int, spec_headroom: int
    ) -> None:
        pool_seq_len = max_seq_length + spec_headroom
        # Account for per-sequence block alignment: each sequence
        # rounds up to a whole number of blocks independently.
        from pace.llm.attention.specs import resolve_text_config

        text_config = resolve_text_config(self._config)
        num_kv_heads = getattr(
            text_config,
            "num_key_value_heads",
            text_config.num_attention_heads,
        )
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        block_size = self._kwargs.get("block_size", 0)
        if block_size <= 0:
            block_size = autotune_block_size(num_kv_heads, head_dim)
        blocks_per_seq = (pool_seq_len + block_size - 1) // block_size
        needed_tokens = blocks_per_seq * block_size * batch_size * 2

        if self._manager is not None:
            if "kv_cache_memory_gb" in self._kwargs:
                return
            if self._manager.max_total_tokens >= needed_tokens:
                return
            self._manager = None

        kw = dict(self._kwargs)
        if "max_total_tokens" not in kw and "kv_cache_memory_gb" not in kw:
            kw["max_total_tokens"] = needed_tokens
        self._manager = SlabPoolManager(self._config, **kw)

    def create_context(
        self, config: PretrainedConfig, max_seq_length: int, **kwargs
    ) -> SlabPoolContext:
        token = kwargs.get("token", None)
        batch_size = kwargs.get("batch_size", 1)
        spec_headroom = kwargs.get("spec_headroom", 0)
        effective_len = max_seq_length + spec_headroom

        self._ensure_manager(max_seq_length, batch_size, spec_headroom)

        if token is not None:
            pool_token = self._manager.create_sequence(
                max_seq_len=effective_len, token=token
            )
            return SlabPoolContext(self._manager, [pool_token])

        tokens = []
        for _ in range(batch_size):
            pool_token = self._manager.create_sequence(max_seq_len=effective_len)
            tokens.append(pool_token)
        return SlabPoolContext(self._manager, tokens)

    def merge_contexts(self, contexts, **kwargs) -> SlabPoolContext:
        """Combine per-sequence contexts into a single batched context."""
        all_tokens = []
        for ctx in contexts:
            all_tokens.extend(ctx.tokens)
        return SlabPoolContext(self._manager, all_tokens)

    def merge_contexts_mixed(self, contexts_with_query_lens: list) -> SlabPoolContext:
        """Merge contexts with per-sequence query lengths for mixed batches.

        Args:
            contexts_with_query_lens: List of ``(kv_cache_context, query_len)``
                tuples.  The query_len per sequence is stored on the returned
                context as ``slab_query_lens`` so the attention kernel can
                dispatch the packed path correctly.
        """
        all_tokens = []
        query_lens = []
        for ctx, ql in contexts_with_query_lens:
            mgr = ctx.kv_cache_manager if hasattr(ctx, "kv_cache_manager") else ctx
            if isinstance(mgr, SlabPoolContext):
                all_tokens.extend(mgr.tokens)
            else:
                all_tokens.extend(ctx.tokens)
            query_lens.append(ql)
        merged = SlabPoolContext(self._manager, all_tokens)
        merged.slab_query_lens = query_lens
        return merged

    def build_prefill_metadata(self, *args, **kwargs):
        """No-op for slab. Paged attention uses this for prefill metadata."""
        return None

    # ---- PrefixCacheMixin hooks ----

    def _get_block_indices_for_context(self, context):
        """Return the physical C++ pool block indices for the first sequence."""
        if not context.tokens or self._manager is None:
            return []
        return self._manager.get_block_indices(context.tokens[0])

    def _claim_prefix_blocks_impl(
        self, context, token_ids, matched_blocks, num_matched_tokens
    ):
        """Zero-copy: point the new sequence at the cached physical blocks."""
        if not context.tokens or self._manager is None:
            return
        self._manager.assign_prefix_blocks(
            context.tokens[0], matched_blocks, num_matched_tokens
        )

    def _free_duplicate_block(self, block_idx: int) -> None:
        """Release a dedup'd block; safe because the owner was repointed off it first."""
        if self._manager is not None:
            self._manager.release_retained_blocks([block_idx])

    def _repoint_duplicate_block(
        self, context, block_pos: int, old_block: int, canonical_block: int
    ) -> None:
        """Zero-copy repoint of this sequence's logical block onto the canonical block."""
        if self._manager is None or not context.tokens:
            return
        self._manager.reassign_block(context.tokens[0], block_pos, canonical_block)

    def _reserve_blocks_for_context(self, context, num_full_blocks):
        if self._manager is None or not context.tokens:
            return []
        deficit = num_full_blocks - self._manager.get_free_blocks()
        if deficit > 0:
            self.evict_prefix_blocks(deficit)
        blocks = self._manager.reserve_blocks(context.tokens[0], num_full_blocks)
        self._manager.retain_blocks(blocks[:num_full_blocks])
        return blocks

    def insert_prefix(self, context, token_ids: list) -> int:
        """Register newly-computed suffix blocks and retain them in the pool.

        The reused prefix (``prefix_matched_tokens``) is already shared zero-copy
        and skipped. Retained blocks survive ``remove_sequence`` and are freed
        only on LRU eviction.
        """
        if self._prefix_map is None:
            return 0
        blocks = self._get_block_indices_for_context(context)
        if not blocks:
            return 0
        skip = getattr(context, "prefix_matched_tokens", 0) // self._prefix_block_size
        hashes = compute_block_hashes(token_ids, self._prefix_block_size)
        num_full = min(len(hashes), len(blocks))
        if num_full > skip:
            self._manager.retain_blocks(blocks[skip:num_full])
        return self._register_blocks(context, hashes, blocks, start=skip)

    def evict_prefix_blocks(self, num_blocks: int) -> list:
        """Evict LRU cached blocks and release their retained pool blocks."""
        if self._prefix_map is None:
            return []
        evicted = self._prefix_map.evict(num_blocks)
        if evicted and self._manager is not None:
            self._manager.release_retained_blocks(evicted)
        return evicted

    def remove_context(self, context) -> None:
        """Release pool blocks and hashmap entries for all sequences.

        Blocks inserted into the prefix cache are retained in the pool, so
        their KV data remains available for future prefix matches; they are
        freed only on LRU eviction.
        """
        self._release_prefix_refs(context)
        for token in context.tokens:
            self._manager.remove_sequence(token)
        context.tokens.clear()

    @property
    def manager(self) -> Optional[SlabPoolManager]:
        return self._manager
