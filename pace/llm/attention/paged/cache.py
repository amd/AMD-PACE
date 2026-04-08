# ******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
#
# Inspired by vLLM's paged attention KV cache design
# (https://github.com/vllm-project/vllm)
# ******************************************************************************

"""
Paged KV cache implementations for efficient memory management.

Contains:
- PagedKVCache: Standalone paged cache for offline generation.
- PagedKVCachePool: Shared block pool for server-mode batched attention.
- SharedPagedKVCache: Per-request view into the shared pool.
"""

import os
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import torch
from transformers import PretrainedConfig

from pace.llm.attention.base import Cache, KVCacheBase, KVCacheType, KVCacheManager
from pace.utils.logging import (
    PACE_INFO,
    PACE_WARNING,
    PACE_LLM_ASSERT,
    PACE_LLM_DEBUG,
)


def compute_slot_mapping(
    layer_cache: "PagedKVCache",
    batch_size: int,
    seq_len: int,
    past_key_values_length: int,
    blocks_per_seq: int = 0,
) -> torch.Tensor:
    """Compute slot mapping for new tokens using vectorized operations.

    Maps each new token position to its physical slot in the paged cache
    (block_idx * block_size + block_offset). When batch_size > 1, each
    sequence is offset so it maps to its own set of physical blocks.

    Args:
        layer_cache: A PagedKVCache instance (typically from layer 0).
        batch_size: Number of sequences in the batch.
        seq_len: Number of new (query) tokens per sequence.
        past_key_values_length: Number of already-cached tokens.
        blocks_per_seq: Number of blocks allocated per sequence. When > 0,
            sequence i is offset by i * blocks_per_seq blocks.

    Returns:
        Int64 tensor of shape [batch_size * seq_len] with physical slot indices.
    """
    block_size = layer_cache.block_size
    num_tokens = batch_size * seq_len

    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64)

    total_needed = past_key_values_length + seq_len
    if batch_size > 1 and blocks_per_seq > 0:
        layer_cache._ensure_blocks_allocated(blocks_per_seq * batch_size * block_size)
    else:
        layer_cache._ensure_blocks_allocated(total_needed)

    num_allocated = len(layer_cache.allocated_blocks)
    if num_allocated == 0:
        return torch.full((num_tokens,), -1, dtype=torch.int64)

    block_table_tensor = torch.tensor(layer_cache.allocated_blocks, dtype=torch.int64)

    per_seq_positions = torch.arange(
        past_key_values_length, past_key_values_length + seq_len, dtype=torch.int64
    )

    all_slots = []
    for seq_idx in range(batch_size):
        block_offset_for_seq = seq_idx * blocks_per_seq if blocks_per_seq > 0 else 0
        block_indices = per_seq_positions // block_size + block_offset_for_seq
        block_offsets = per_seq_positions % block_size

        valid_mask = block_indices < num_allocated
        clamped = torch.clamp(block_indices, 0, num_allocated - 1)
        physical = block_table_tensor[clamped]
        slots = physical * block_size + block_offsets
        slots = torch.where(valid_mask, slots, torch.tensor(-1, dtype=torch.int64))
        all_slots.append(slots)

    return torch.cat(all_slots)


def compute_single_slot(layer_cache: "PagedKVCache", pos: int) -> int:
    """Compute the physical slot for a single token position.

    Fast path for decode where only one new token is written per request.
    Callers must ensure blocks are already allocated for ``pos``.

    Returns:
        Physical slot index, or -1 if ``pos`` exceeds allocated blocks.
    """
    block_idx = pos // layer_cache.block_size
    block_offset = pos % layer_cache.block_size
    if block_idx < len(layer_cache.allocated_blocks):
        physical_block = layer_cache.allocated_blocks[block_idx]
        return physical_block * layer_cache.block_size + block_offset
    return -1


class PagedKVCache(KVCacheBase):
    """
    Paged key-value cache implementation for efficient memory management.

    This cache stores KV pairs in fixed-size blocks that can be allocated
    and deallocated dynamically, similar to vLLM's paged attention.

    Attributes:
        block_size: Number of tokens per block
        num_blocks: Total number of allocated blocks
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each attention head
        key_cache: Paged key cache tensor [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache tensor [num_blocks, num_kv_heads, block_size, head_dim]
        block_table: Mapping from sequence positions to block indices
        slot_mapping: Mapping from token positions to cache slots
    """

    def __init__(
        self,
        max_seq_length: int,
        num_kv_heads: int = 32,
        head_dim: int = 128,
        block_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the PagedKVCache.

        Args:
            max_seq_length: Maximum sequence length
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each attention head
            block_size: Number of tokens per block
            dtype: Data type for the cache tensors
            device: Device to allocate the cache on
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Calculate number of blocks needed
        self.num_blocks = (max_seq_length + block_size - 1) // block_size

        # Allocate paged cache.  We use zeros rather than empty so that
        # stale slots (e.g. after spec-decode rollback) contain deterministic
        # values, making any slot-mapping bugs reproducible instead of silent.
        # This is a one-time allocation per model load, so the zeroing cost
        # is negligible relative to model loading time.
        cache_shape = (self.num_blocks, num_kv_heads, block_size, head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # Block allocation tracking
        self.allocated_blocks: List[int] = []
        self.free_blocks: deque = deque(range(self.num_blocks))

        # Sequence tracking
        self.seq_len = 0
        self.block_table: Optional[torch.Tensor] = None
        self.slot_mapping: Optional[torch.Tensor] = None

        PACE_LLM_DEBUG(
            f"PagedKVCache initialized with {self.num_blocks} blocks, "
            f"block_size={block_size}, max_seq_length={max_seq_length}"
        )

    def _allocate_block(self) -> int:
        """Allocate a new block from the free pool."""
        if not self.free_blocks:
            raise RuntimeError("No free blocks available in PagedKVCache")
        block_idx = self.free_blocks.popleft()
        self.allocated_blocks.append(block_idx)
        return block_idx

    def _free_block(self, block_idx: int) -> None:
        """Return a block to the free pool."""
        if block_idx in self.allocated_blocks:
            self.allocated_blocks.remove(block_idx)
            self.free_blocks.append(block_idx)

    def _ensure_blocks_allocated(self, required_len: int) -> None:
        """Ensure enough blocks are allocated for the given sequence length."""
        required_blocks = (required_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.allocated_blocks)

        if current_blocks >= required_blocks:
            if (
                self.block_table is not None
                and self.block_table.size(0) >= required_blocks
            ):
                return

        while current_blocks < required_blocks:
            self._allocate_block()
            current_blocks += 1

        self.block_table = torch.tensor(
            self.allocated_blocks[:required_blocks],
            dtype=torch.int32,
            device=self.device,
        )

    def remove_cache(self, remove_len: int) -> None:
        """
        Remove the last `remove_len` tokens from the cache.

        Args:
            remove_len: Number of tokens to remove
        """
        if remove_len > self.seq_len:
            raise ValueError("Cannot remove more tokens than available in cache.")

        new_seq_len = self.seq_len - remove_len

        new_blocks_needed = (new_seq_len + self.block_size - 1) // self.block_size

        while len(self.allocated_blocks) > new_blocks_needed:
            block_to_free = self.allocated_blocks.pop()
            self.free_blocks.append(block_to_free)

        self.seq_len = new_seq_len

        if new_seq_len > 0:
            self.block_table = torch.tensor(
                self.allocated_blocks, dtype=torch.int32, device=self.device
            )
        else:
            self.block_table = None

    def update_cache(
        self, key_states: torch.Tensor, value_states: torch.Tensor, concat_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the paged cache with new key/value states.

        This method only supports batch_size=1. For batched paged attention,
        use reshape_and_cache via build_paged_attention_metadata which handles
        per-sequence block offsets correctly.

        Args:
            key_states: New key states [1, num_kv_heads, seq_len, head_dim] (BNSH)
            value_states: New value states (same shape as key_states)
            concat_dim: Must be 2 (sequence dim in BNSH layout)

        Returns:
            Tuple of (key_cache, value_cache) tensor references
        """
        batch_size = key_states.size(0)
        PACE_LLM_ASSERT(
            batch_size == 1,
            f"PagedKVCache.update_cache only supports batch_size=1, got {batch_size}. "
            "For batched paged attention, use reshape_and_cache with "
            "per-sequence slot mappings from build_paged_attention_metadata.",
        )
        PACE_LLM_ASSERT(
            concat_dim == 2,
            f"Paged cache expects BNSH layout (concat_dim=2), got concat_dim={concat_dim}",
        )

        num_new_tokens = key_states.size(2)
        key_for_cache = key_states.transpose(1, 2)
        value_for_cache = value_states.transpose(1, 2)

        key_flat = key_for_cache.reshape(-1, self.num_kv_heads, self.head_dim)
        value_flat = value_for_cache.reshape(-1, self.num_kv_heads, self.head_dim)

        new_total_len = self.seq_len + num_new_tokens
        self._ensure_blocks_allocated(new_total_len)

        from pace.llm.attention.paged.ops import paged_attention_reshape_and_cache

        slot_mapping = compute_slot_mapping(
            self,
            batch_size=1,
            seq_len=num_new_tokens,
            past_key_values_length=self.seq_len,
        )

        paged_attention_reshape_and_cache(
            key_flat, value_flat, self.key_cache, self.value_cache, slot_mapping, "auto"
        )

        self.seq_len = new_total_len
        self.slot_mapping = slot_mapping

        return self.key_cache, self.value_cache

    def get_cache_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the paged cache tensors."""
        return self.key_cache, self.value_cache


class PagedKVCachePool:
    """
    Unified KV cache pool for efficient batched paged attention.

    This pool manages a single large KV cache tensor that is shared across
    all requests. Each request allocates blocks from this shared pool,
    enabling truly batched paged attention operations.

    Thread safety: The pool is NOT thread-safe. The current server architecture
    guarantees single-threaded access per engine process (prefill and decode
    run sequentially in the scheduler loop, multi-instance uses separate OS
    processes). If concurrency is introduced (async prefill, threaded request
    handling), allocate_blocks / free_blocks_for_request must be guarded with
    a lock.

    Attributes:
        total_blocks: Total number of blocks in the pool
        block_size: Number of tokens per block
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension of each attention head
        num_layers: Number of transformer layers
        key_cache: Unified key cache [num_layers, total_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Unified value cache [same shape as key_cache]
    """

    _instance = None  # Singleton instance

    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the unified KV cache pool.

        Args:
            total_blocks: Total number of blocks across all requests
            block_size: Number of tokens per block
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            num_layers: Number of transformer layers
            dtype: Data type for cache tensors
            device: Device to allocate on
        """
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # zeros (not empty) so stale slots after rollback are deterministic.
        # One-time allocation per model load; zeroing cost is negligible.
        cache_shape = (num_layers, total_blocks, num_kv_heads, block_size, head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        self.free_blocks: deque = deque(range(total_blocks))
        self.allocated_blocks: Dict[int, List[int]] = {}

        PACE_LLM_DEBUG(
            f"PagedKVCachePool initialized with {total_blocks} blocks, "
            f"block_size={block_size}, num_layers={num_layers}"
        )

    @classmethod
    def get_instance(cls) -> Optional["PagedKVCachePool"]:
        """Get the singleton instance of the pool."""
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing or reconfiguration)."""
        cls._instance = None

    def get_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        total_allocated = sum(len(blocks) for blocks in self.allocated_blocks.values())
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": total_allocated,
            "active_requests": len(self.allocated_blocks),
            "utilization_pct": (
                (total_allocated / self.total_blocks * 100)
                if self.total_blocks > 0
                else 0
            ),
        }

    def reset_allocations(self) -> None:
        """Reset all block allocations (free all blocks). Use with caution."""
        self.free_blocks = deque(range(self.total_blocks))
        self.allocated_blocks = {}
        PACE_LLM_DEBUG(
            f"PagedKVCachePool: Reset all allocations, "
            f"{self.total_blocks} blocks now free"
        )

    @classmethod
    def initialize(
        cls,
        total_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ) -> "PagedKVCachePool":
        """Initialize the singleton pool instance.

        If the pool already exists with different parameters, it will be reset.
        """
        if cls._instance is not None:
            if (
                cls._instance.total_blocks != total_blocks
                or cls._instance.block_size != block_size
                or cls._instance.num_kv_heads != num_kv_heads
                or cls._instance.head_dim != head_dim
                or cls._instance.num_layers != num_layers
                or cls._instance.dtype != dtype
            ):
                PACE_LLM_DEBUG("PagedKVCachePool: Parameters changed, recreating pool")
                cls._instance = None

        if cls._instance is None:
            cls._instance = cls(
                total_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                num_layers,
                dtype,
                device,
            )
        return cls._instance

    def allocate_blocks(self, request_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a request from the free pool."""
        if len(self.free_blocks) < num_blocks:
            total_allocated = sum(
                len(blocks) for blocks in self.allocated_blocks.values()
            )
            num_requests = len(self.allocated_blocks)
            raise RuntimeError(
                f"Not enough free blocks. Requested {num_blocks}, "
                f"available {len(self.free_blocks)}. "
                f"Total allocated: {total_allocated} blocks across "
                f"{num_requests} requests. "
                f"Total blocks: {self.total_blocks}"
            )

        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.popleft()
            allocated.append(block_idx)

        if request_id not in self.allocated_blocks:
            self.allocated_blocks[request_id] = []
        self.allocated_blocks[request_id].extend(allocated)

        total_for_req = len(self.allocated_blocks[request_id])
        PACE_LLM_DEBUG(
            f"[POOL] Allocated {num_blocks} blocks for request_id={request_id}, "
            f"total for req: {total_for_req}, free: {len(self.free_blocks)}"
        )

        return allocated

    def free_blocks_for_request(self, request_id: int) -> None:
        """Free all blocks allocated to a request."""
        if request_id in self.allocated_blocks:
            blocks = self.allocated_blocks.pop(request_id)
            self.free_blocks.extend(blocks)
            PACE_INFO(
                f"[POOL] Freed {len(blocks)} blocks for request_id={request_id}, "
                f"now {len(self.free_blocks)} free, "
                f"{len(self.allocated_blocks)} active requests"
            )
        else:
            PACE_WARNING(
                f"[POOL] Warning: No blocks found for request_id={request_id} "
                f"to free. Known request_ids: "
                f"{list(self.allocated_blocks.keys())[:10]}..."
            )

    def get_cache_tensors(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the cache tensors for a specific layer."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_blocks_for_request(self, request_id: int) -> List[int]:
        """Get the list of blocks allocated to a request."""
        return self.allocated_blocks.get(request_id, [])

    def ensure_blocks_for_request(
        self, request_id: int, required_blocks: int
    ) -> List[int]:
        """Ensure a request has the required number of blocks allocated."""
        current_blocks = self.allocated_blocks.get(request_id, [])
        if len(current_blocks) >= required_blocks:
            return current_blocks

        additional_needed = required_blocks - len(current_blocks)
        self.allocate_blocks(request_id, additional_needed)
        return self.allocated_blocks[request_id]


class SharedPagedKVCache(PagedKVCache):
    """
    PagedKVCache that uses a shared pool for truly batched operations.

    This cache does not allocate its own tensors but instead references
    blocks from a shared PagedKVCachePool. Inherits from PagedKVCache
    so that isinstance checks work correctly.
    """

    def __init__(
        self,
        pool: PagedKVCachePool,
        request_id: int,
        layer_idx: int,
        max_seq_length: int,
    ):
        """
        Initialize a shared paged cache for a specific request and layer.

        Args:
            pool: The shared cache pool
            request_id: Unique identifier for this request
            layer_idx: Which layer this cache is for
            max_seq_length: Maximum sequence length
        """
        KVCacheBase.__init__(self)
        self.pool = pool
        self.request_id = request_id
        self.layer_idx = layer_idx
        self.max_seq_length = max_seq_length
        self.block_size = pool.block_size
        self.seq_len = 0
        self.num_kv_heads = pool.num_kv_heads
        self.head_dim = pool.head_dim
        self.dtype = pool.dtype
        self.device = pool.device
        self.num_blocks = pool.total_blocks
        self.key_cache = None
        self.value_cache = None
        self.free_blocks: deque = deque()
        self.block_table = None

    def get_cache_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the unified cache tensors for this layer."""
        return self.pool.get_cache_tensors(self.layer_idx)

    def get_allocated_blocks(self) -> List[int]:
        """Get the blocks allocated to this request."""
        return self.pool.get_blocks_for_request(self.request_id)

    def _ensure_blocks_allocated(self, required_len: int) -> None:
        """Ensure enough blocks are allocated for the sequence length."""
        required_blocks = (required_len + self.block_size - 1) // self.block_size
        self.pool.ensure_blocks_for_request(self.request_id, required_blocks)

    @property
    def allocated_blocks(self) -> List[int]:
        """Property to get allocated blocks (for compatibility)."""
        return self.pool.get_blocks_for_request(self.request_id)

    def remove_cache(self, remove_len: int) -> None:
        """Remove tokens from the cache."""
        if remove_len > self.seq_len:
            raise ValueError("Cannot remove more tokens than available")
        self.seq_len -= remove_len

    def update_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        concat_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "SharedPagedKVCache.update_cache should not be called directly. "
            "Use paged_attention_reshape_and_cache via the model's paged "
            "attention path."
        )


class _PagedCacheContext:
    """Wraps a KVCacheManager with optional paged_attn_metadata for serving."""

    def __init__(self, kv_cache_manager, pool_request_id=None):
        self.kv_cache_manager = kv_cache_manager
        self.cache_objects = kv_cache_manager.cache_objects
        self.pool_request_id = pool_request_id
        self.paged_attn_metadata = None

    def __len__(self):
        return len(self.kv_cache_manager)

    def remove_cache(self, remove_len):
        self.kv_cache_manager.remove_cache(remove_len)


class PagedCache(Cache):
    """Engine-level cache backend for paged attention. Implements Cache ABC.

    Mirrors SlabCache: pool is created eagerly in __init__, contexts are
    created per-request via create_context, batched decode metadata is
    built in merge_contexts. Server code stays cache-type agnostic.
    """

    _query_start_loc_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    _isa_cache: Optional[str] = None

    def __init__(self, config: PretrainedConfig, **kwargs):
        self._config = config
        self._kwargs = kwargs

        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
        num_layers = config.num_hidden_layers
        self._block_size = kwargs.get("block_size", 16)
        self._dtype = kwargs.get("dtype", torch.bfloat16)
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_heads = num_heads

        max_total_tokens = kwargs.get(
            "max_total_tokens",
            int(os.getenv("PACE_MAX_CACHE_TOKENS", "262144")),
        )
        total_blocks = (max_total_tokens + self._block_size - 1) // self._block_size

        PagedKVCachePool.reset()
        self._pool = PagedKVCachePool(
            total_blocks=total_blocks,
            block_size=self._block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            dtype=self._dtype,
        )
        PACE_INFO(
            f"PagedCache: pool initialized with {max_total_tokens} tokens, "
            f"{total_blocks} blocks, {num_layers} layers"
        )

        self._next_request_id = 0

        # Pre-allocated decode tensors
        max_batch = 64
        max_seq = 4096
        max_blocks = (max_seq + self._block_size - 1) // self._block_size
        self._decode_seq_lens = torch.zeros(max_batch, dtype=torch.int32)
        self._decode_slot_mapping = torch.zeros(max_batch, dtype=torch.int64)
        self._decode_block_table = torch.zeros(max_batch, max_blocks, dtype=torch.int32)
        self._max_batch_size = max_batch

        # ISA cache
        if PagedCache._isa_cache is None:
            from pace.llm.attention.paged.utils import get_cached_isa

            PagedCache._isa_cache = get_cached_isa(
                self._dtype, self._block_size, head_dim
            )

    def create_context(
        self, config: PretrainedConfig, max_seq_length: int, **kwargs
    ) -> _PagedCacheContext:
        pool_request_id = self._next_request_id
        self._next_request_id += 1

        num_layers = config.num_hidden_layers
        manager = KVCacheManager.__new__(KVCacheManager)
        manager.num_layers = num_layers
        manager.cache_type = KVCacheType.PAGED
        manager.kv_cache_class = SharedPagedKVCache
        manager.cache_objects = [
            SharedPagedKVCache(
                pool=self._pool,
                request_id=pool_request_id,
                layer_idx=layer_idx,
                max_seq_length=max_seq_length,
            )
            for layer_idx in range(num_layers)
        ]

        return _PagedCacheContext(manager, pool_request_id=pool_request_id)

    def merge_contexts(self, contexts, query_len=1) -> _PagedCacheContext:
        """Merge per-request contexts and build batched paged attention metadata.

        Args:
            contexts: List of per-request cache contexts (KVCacheManager or
                _PagedCacheContext wrappers).
            query_len: Number of new tokens per request.  Default 1 for
                standard decode; set > 1 for speculative-decode target
                forwards where each request contributes multiple query tokens.
        """
        from pace.llm.attention.paged.ops import (
            PagedAttentionMetadata,
            get_paged_attention_scheduler_metadata,
        )

        num_reqs = len(contexts)
        isa = PagedCache._isa_cache
        is_decode = query_len == 1

        if is_decode and num_reqs <= self._max_batch_size:
            seq_lens = self._decode_seq_lens[:num_reqs]
            slot_mapping = self._decode_slot_mapping[:num_reqs]
        else:
            seq_lens = torch.zeros(num_reqs, dtype=torch.int32)
            slot_mapping = torch.zeros(num_reqs * query_len, dtype=torch.int64)

        max_blocks_per_seq = 0
        max_seq_len = 0
        kv_managers = []

        for i, ctx in enumerate(contexts):
            mgr = ctx.kv_cache_manager if hasattr(ctx, "kv_cache_manager") else ctx
            kv_managers.append(mgr)
            layer_cache = mgr.cache_objects[0]
            past_len = layer_cache.seq_len
            total_seq_len = past_len + query_len

            seq_lens[i] = total_seq_len
            if total_seq_len > max_seq_len:
                max_seq_len = total_seq_len

            layer_cache._ensure_blocks_allocated(total_seq_len)

            # compute_slot_mapping cannot be used here because each request
            # has its own SharedPagedKVCache with independent block allocations
            # and different past lengths — unlike the offline batch path where
            # all sequences share a single PagedKVCache.
            if is_decode:
                slot_mapping[i] = compute_single_slot(layer_cache, past_len)
            else:
                for t in range(query_len):
                    pos = past_len + t
                    slot_mapping[i * query_len + t] = compute_single_slot(
                        layer_cache, pos
                    )

            num_blocks = len(layer_cache.allocated_blocks)
            if num_blocks > max_blocks_per_seq:
                max_blocks_per_seq = num_blocks

        if (
            is_decode
            and num_reqs <= self._max_batch_size
            and max_blocks_per_seq <= self._decode_block_table.shape[1]
        ):
            block_table = self._decode_block_table[:num_reqs, :max_blocks_per_seq]
            block_table.zero_()
        else:
            block_table = torch.zeros(num_reqs, max_blocks_per_seq, dtype=torch.int32)

        # TODO: allocated_blocks returns a fresh Python list on every access.
        # Consider maintaining blocks as a tensor at the pool level to avoid
        # list creation and the subsequent torch.tensor conversion per request.
        for i, mgr in enumerate(kv_managers):
            layer_cache = mgr.cache_objects[0]
            blocks = layer_cache.allocated_blocks
            nb = min(len(blocks), max_blocks_per_seq)
            block_table[i, :nb] = torch.tensor(blocks[:nb], dtype=torch.int32)

        cache_key = (num_reqs, query_len)
        if cache_key not in PagedCache._query_start_loc_cache:
            PagedCache._query_start_loc_cache[cache_key] = (
                torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
            )
        query_start_loc = PagedCache._query_start_loc_cache[cache_key]

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            seq_lens=seq_lens,
            dtype=self._dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa=isa,
            enable_kv_split=True,
        )

        meta = PagedAttentionMetadata(
            isa=isa,
            num_actual_tokens=num_reqs * query_len,
            max_query_len=query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table.contiguous(),
            slot_mapping=slot_mapping,
            scheduler_metadata=scheduler_metadata,
            causal=True,
        )

        for i, mgr in enumerate(kv_managers):
            new_seq_len = int(seq_lens[i].item())
            for co in mgr.cache_objects:
                co.seq_len = new_seq_len

        merged = _PagedCacheContext(kv_managers[0])
        merged.paged_attn_metadata = meta
        merged._all_managers = kv_managers
        return merged

    def build_prefill_metadata(self, context, seq_len, past_len=0):
        """Build single-sequence metadata for prefill or draft decode."""
        from pace.llm.attention.paged.utils import build_paged_attention_metadata

        mgr = (
            context.kv_cache_manager
            if hasattr(context, "kv_cache_manager")
            else context
        )
        query_lengths = torch.tensor([seq_len], dtype=torch.long)
        past = torch.tensor([past_len], dtype=torch.long)
        meta = build_paged_attention_metadata(
            mgr,
            self._config,
            query_lengths,
            self._dtype,
            self._block_size,
            past_lengths=past,
        )
        new_seq_len = past_len + seq_len
        for co in mgr.cache_objects:
            co.seq_len = new_seq_len
        return meta

    def remove_context(self, context) -> None:
        """Release pool blocks for the request."""
        if hasattr(context, "pool_request_id") and context.pool_request_id is not None:
            blocks_before = len(self._pool.free_blocks)
            self._pool.free_blocks_for_request(context.pool_request_id)
            blocks_after = len(self._pool.free_blocks)
            freed = blocks_after - blocks_before
            if freed > 0:
                PACE_INFO(
                    f"[POOL] Freed {freed} blocks for "
                    f"pool_request_id={context.pool_request_id}"
                )
