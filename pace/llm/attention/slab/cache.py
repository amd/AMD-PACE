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

from pace.llm.attention.base import Cache, CacheContext
from pace.utils.logging import PACE_LLM_INFO


def autotune_block_size(num_kv_heads: int, head_dim: int) -> int:
    """Pick the largest block_size that fits in L2/4.

    Reads L2 cache size from sysfs, falls back to 64 if unavailable.
    Respects SLAB_BLOCK_SIZE env var override.
    """
    import os

    env_bs = os.environ.get("SLAB_BLOCK_SIZE")
    if env_bs:
        return int(env_bs)

    l2_size = 0
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index2/size") as f:
            s = f.read().strip()
            val = int(s.rstrip("KkMm"))
            if s[-1] in "Kk":
                l2_size = val * 1024
            elif s[-1] in "Mm":
                l2_size = val * 1024 * 1024
            else:
                l2_size = val
    except (OSError, ValueError):
        return 64

    bytes_per_token = 2 * num_kv_heads * head_dim * 2  # K+V, BF16
    target = l2_size // 4
    for bs in (256, 128, 64, 32):
        if bs * bytes_per_token <= target:
            return bs
    return 32


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

    __slots__ = ("pool_manager", "tokens", "layer_idx")

    def __init__(
        self,
        pool_manager: "SlabPoolManager",
        tokens: List[str],
        layer_idx: int,
    ):
        self.pool_manager = pool_manager
        self.tokens = tokens
        self.layer_idx = layer_idx

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

    __slots__ = ("_pool_manager", "_tokens")

    def __init__(self, pool_manager: "SlabPoolManager", tokens: List[str]):
        self._pool_manager = pool_manager
        self._tokens = tokens

    def __getitem__(self, layer_idx: int) -> SlabPoolLayerView:
        return SlabPoolLayerView(self._pool_manager, self._tokens, layer_idx)


class SlabPoolContext(CacheContext):
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
        self.cache_objects = _SlabPoolLayerList(pool_manager, tokens)

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
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        """
        Compute max_total_tokens from a memory budget.

        Each token in the pool uses:
          num_layers * 2 (K+V) * num_kv_heads * head_dim * sizeof(BF16)
        bytes across all layers.
        """
        bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * 2
        bytes_per_token = num_layers * bytes_per_token_per_layer
        total_bytes = int(kv_cache_memory_gb * 1024 * 1024 * 1024)
        return total_bytes // bytes_per_token

    def __init__(
        self,
        config,
        kv_cache_memory_gb: Optional[float] = None,
        max_total_tokens: Optional[int] = None,
        block_size: int = 0,
        default_max_seq_len: Optional[int] = None,
    ):
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.num_q_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if block_size <= 0:
            block_size = autotune_block_size(self.num_kv_heads, self.head_dim)
        self.block_size = block_size

        if max_total_tokens is not None:
            self.max_total_tokens = max_total_tokens
        elif kv_cache_memory_gb is not None:
            self.max_total_tokens = self.compute_max_tokens_from_memory(
                kv_cache_memory_gb,
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            max_pos = getattr(config, "max_position_embeddings", 4096)
            self.max_total_tokens = min(max_pos, 8192)

        if default_max_seq_len is not None:
            self.default_max_seq_len = default_max_seq_len
        else:
            self.default_max_seq_len = getattr(config, "max_position_embeddings", 4096)

        total_blocks = (self.max_total_tokens + block_size - 1) // block_size

        bytes_per_token = self.num_layers * 2 * self.num_kv_heads * self.head_dim * 2
        pool_memory_gb = (self.max_total_tokens * bytes_per_token) / (1024**3)

        PACE_LLM_INFO(
            f"SlabPoolManager: {self.max_total_tokens} max tokens, "
            f"{total_blocks} blocks, {pool_memory_gb:.2f} GB KV cache memory "
            f"({self.num_layers} layers, {self.num_kv_heads} KV heads, "
            f"{self.head_dim} head_dim)"
        )

        self._pools = []
        for _ in range(self.num_layers):
            pool = create_slab_pool(
                total_blocks, self.num_kv_heads, self.head_dim, block_size
            )
            self._pools.append(pool)

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
                for pool in self._pools:
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

            for pool in self._pools:
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
            for pool in self._pools:
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

    def update_cache_batched(
        self,
        layer_idx: int,
        tokens: List[str],
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
    ) -> None:
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
    ) -> torch.Tensor:
        """Unified attention: C++ dispatches to decode/prefill based on q_len."""
        if scale is None:
            scale = 1.0 / (self.head_dim**0.5)
        seq_ids = self.get_internal_ids(tokens)
        sinks_tensor = sinks if sinks is not None else torch.tensor([])
        return self._pools[layer_idx].attention(
            seq_ids, query, seq_lens or [], [], scale, sliding_window, sinks_tensor
        )


class SlabCache(Cache):
    """Engine-level cache backend for SlabPool. Implements Cache ABC.

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

    def _ensure_manager(
        self, max_seq_length: int, batch_size: int, spec_headroom: int
    ) -> None:
        pool_seq_len = max_seq_length + spec_headroom
        # Account for per-sequence block alignment: each sequence
        # rounds up to a whole number of blocks independently.
        num_kv_heads = getattr(
            self._config,
            "num_key_value_heads",
            self._config.num_attention_heads,
        )
        head_dim = getattr(
            self._config,
            "head_dim",
            self._config.hidden_size // self._config.num_attention_heads,
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

    def build_prefill_metadata(self, *args, **kwargs):
        """No-op for slab. Paged attention uses this for prefill metadata."""
        return None

    def remove_context(self, context) -> None:
        """Release pool blocks and hashmap entries for all sequences."""
        for token in context.tokens:
            self._manager.remove_sequence(token)
        context.tokens.clear()

    @property
    def manager(self) -> Optional[SlabPoolManager]:
        return self._manager
