# ******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
#
# Inspired by vLLM's paged attention design
# (https://github.com/vllm-project/vllm)
# ******************************************************************************

"""
Shared utility functions for paged attention.

These helpers are used by both the online server path
(pace.server.engine.server) and offline usage (examples, benchmarks).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pace.llm.attention.base import KVCacheType, KVCacheManager
from pace.llm.attention.paged.cache import PagedKVCache, compute_slot_mapping
from pace.llm.attention.paged.ops import (
    PagedAttentionMetadata,
    get_optimal_attention_isa,
)
from pace.llm.attention.specs import compute_layer_kv_specs, resolve_text_config


def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    metadata: PagedAttentionMetadata,
    block_size: int,
) -> PagedAttentionMetadata:
    """Rewrite paged metadata so a standard causal kernel produces chunked
    (local) attention (Llama4 iRoPE local layers).

    Chunked attention restricts every query at absolute position ``q`` to keys
    within its own fixed-size chunk ``[floor(q/C)*C, q]``. Rather than teach the
    C++ kernel a per-query window, we split each request into one *virtual*
    request per local chunk it spans — each with a chunk-local ``seq_len`` and a
    ``block_table`` sliced to that chunk's pages — and run the ordinary causal
    kernel over the virtual batch. This is vLLM's ``make_local_attention_
    virtual_batches`` approach; the query token order is preserved, so the
    flattened output maps back 1:1.

    Requires ``attn_chunk_size % block_size == 0``. Returns a new
    ``PagedAttentionMetadata``; ``slot_mapping`` is shared unchanged (KV is
    written with the real per-token slots before this transform is applied).
    """
    if attn_chunk_size % block_size != 0:
        raise ValueError(
            f"attn_chunk_size ({attn_chunk_size}) must be divisible by "
            f"block_size ({block_size}) for chunked paged attention."
        )

    def cdiv(a, b):
        return -(-a // b)

    query_start_loc_np = metadata.query_start_loc.cpu().numpy()
    seq_lens_np = metadata.seq_lens.cpu().numpy()
    block_table = metadata.block_table

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Tokens of each request that fall in its first (possibly partial) chunk.
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Per-virtual-batch query lengths (first/last chunk may be partial).
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # Per-virtual-batch key lengths: a full chunk except the final one.
    seqlens_k_local = np.full(virtual_batches, attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    block_starts = k_seqstarts_absolute // block_size
    pages_per_local_batch = attn_chunk_size // block_size

    # Slice each virtual batch's block_table rows to its chunk's pages.
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[
        torch.from_numpy(batch_indices).long(),
        torch.from_numpy(block_indices).long(),
    ].view(virtual_batches, -1)

    query_start_loc_local = torch.from_numpy(cu_seqlens_q_local).to(
        metadata.query_start_loc.device
    )
    seq_lens_local = torch.from_numpy(seqlens_k_local).to(metadata.seq_lens.device)

    return PagedAttentionMetadata(
        isa=metadata.isa,
        num_actual_tokens=metadata.num_actual_tokens,
        max_query_len=int(seqlens_q_local.max()),
        query_start_loc=query_start_loc_local,
        max_seq_len=int(seqlens_k_local.max()),
        seq_lens=seq_lens_local,
        block_table=block_table_local,
        slot_mapping=metadata.slot_mapping,
        causal=True,
    )


def create_paged_kv_cache_manager(
    model_config,
    max_seq_length: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
) -> KVCacheManager:
    """Create a KVCacheManager backed by per-layer PagedKVCache objects.

    Args:
        model_config: HuggingFace-style model config with num_attention_heads,
            num_key_value_heads, hidden_size, num_hidden_layers.
        max_seq_length: Maximum sequence length per sequence to allocate for.
        block_size: Number of tokens per paged cache block.
        dtype: Data type for cache tensors.
        batch_size: Number of sequences. Each sequence gets its own set of
            blocks so that batched offline generation works correctly.

    Returns:
        A KVCacheManager configured with PagedKVCache for each layer.
    """
    from pace.llm.attention.paged.cache import _PagedKVSharingView

    specs = compute_layer_kv_specs(model_config)
    num_layers = len(specs)

    blocks_per_seq = (max_seq_length + block_size - 1) // block_size
    total_blocks = blocks_per_seq * batch_size
    total_cache_length = total_blocks * block_size

    cache_objects = [None] * num_layers
    for layer_idx, spec in enumerate(specs):
        if spec.is_shared:
            continue
        cache_objects[layer_idx] = PagedKVCache(
            max_seq_length=total_cache_length,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            block_size=block_size,
            dtype=dtype,
        )
    for layer_idx, spec in enumerate(specs):
        if spec.is_shared:
            cache_objects[layer_idx] = _PagedKVSharingView(
                cache_objects[spec.kv_sharing_target]
            )

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.num_layers = num_layers
    manager.cache_type = KVCacheType.PAGED
    manager.kv_cache_class = PagedKVCache
    manager.cache_objects = cache_objects
    manager._blocks_per_seq = blocks_per_seq
    manager._batch_size = batch_size
    return manager


def build_block_table(
    layer_cache: PagedKVCache,
    batch_size: int,
    total_seq_len: int,
    blocks_per_seq: int = 0,
) -> torch.Tensor:
    """Build the block table mapping logical blocks to physical blocks.

    When blocks_per_seq > 0, each sequence gets its own range of blocks
    from the allocated pool (sequence i uses blocks
    [i*blocks_per_seq, (i+1)*blocks_per_seq)).

    Args:
        layer_cache: A PagedKVCache instance (typically from layer 0).
        batch_size: Number of sequences in the batch.
        total_seq_len: Total sequence length including cached tokens.
        blocks_per_seq: Number of blocks allocated per sequence. When > 0,
            each batch row gets a distinct range of physical blocks.

    Returns:
        Int32 tensor of shape [batch_size, max_blocks_per_seq].
    """
    block_size = layer_cache.block_size
    max_blocks_per_seq = (total_seq_len + block_size - 1) // block_size

    if batch_size > 1 and blocks_per_seq > 0:
        layer_cache._ensure_blocks_allocated(blocks_per_seq * batch_size * block_size)
    else:
        layer_cache._ensure_blocks_allocated(total_seq_len)

    num_allocated = len(layer_cache.allocated_blocks)
    if num_allocated == 0:
        return torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32)

    all_blocks = torch.tensor(layer_cache.allocated_blocks, dtype=torch.int32)

    if blocks_per_seq > 0 and batch_size > 1:
        rows = []
        for seq_idx in range(batch_size):
            start = seq_idx * blocks_per_seq
            end = start + min(max_blocks_per_seq, blocks_per_seq)
            row = all_blocks[start:end]
            if row.size(0) < max_blocks_per_seq:
                pad = torch.zeros(max_blocks_per_seq - row.size(0), dtype=torch.int32)
                row = torch.cat([row, pad])
            rows.append(row)
        return torch.stack(rows)

    num_blocks_to_use = min(max_blocks_per_seq, num_allocated)
    block_indices = all_blocks[:num_blocks_to_use]
    if num_blocks_to_use < max_blocks_per_seq:
        padding = torch.zeros(max_blocks_per_seq - num_blocks_to_use, dtype=torch.int32)
        block_indices = torch.cat([block_indices, padding])
    return block_indices.unsqueeze(0).expand(batch_size, -1).contiguous()


def build_paged_attention_metadata(
    kv_cache_manager: KVCacheManager,
    model_config,
    actual_lengths: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    past_lengths: Optional[torch.Tensor] = None,
) -> PagedAttentionMetadata:
    """Build metadata for packed (no-padding) batched paged attention.

    Sequences of different lengths are concatenated into a single token
    stream.  ``query_start_loc`` marks per-sequence boundaries so the
    kernel processes each sequence independently within one call.

    Args:
        kv_cache_manager: KVCacheManager with blocks_per_seq partitioning.
        model_config: HuggingFace-style model config.
        actual_lengths: Int tensor [batch_size] — number of new tokens
            per sequence (query lengths).
        dtype: Model data type.
        block_size: Paged cache block size.
        past_lengths: Optional int tensor [batch_size] — tokens already
            cached per sequence (0 for prefill, >0 for decode).
    """
    layer_cache = kv_cache_manager.cache_objects[0]
    blocks_per_seq = getattr(kv_cache_manager, "_blocks_per_seq", 0)
    batch_size = len(actual_lengths)

    text_config = resolve_text_config(model_config)
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_heads)

    if past_lengths is None:
        past_lengths = torch.zeros(batch_size, dtype=torch.long)

    seq_lens = (past_lengths + actual_lengths).to(torch.int32)
    total_tokens = int(actual_lengths.sum().item())
    max_seq_len = int(seq_lens.max().item())

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32)
    torch.cumsum(actual_lengths.to(torch.int32), dim=0, out=query_start_loc[1:])
    max_query_len = int(actual_lengths.max().item())

    max_blocks_needed = (max_seq_len + block_size - 1) // block_size
    if batch_size > 1 and blocks_per_seq > 0:
        layer_cache._ensure_blocks_allocated(blocks_per_seq * batch_size * block_size)
    else:
        layer_cache._ensure_blocks_allocated(max_seq_len)

    num_allocated = len(layer_cache.allocated_blocks)
    block_table_tensor = torch.tensor(layer_cache.allocated_blocks, dtype=torch.int64)

    all_slots = []
    all_block_rows = []
    for seq_idx in range(batch_size):
        q_len = int(actual_lengths[seq_idx].item())
        past = int(past_lengths[seq_idx].item())
        block_offset_for_seq = seq_idx * blocks_per_seq if blocks_per_seq > 0 else 0

        positions = torch.arange(past, past + q_len, dtype=torch.int64)
        block_indices = positions // block_size + block_offset_for_seq
        block_offsets = positions % block_size

        valid_mask = block_indices < num_allocated
        clamped = torch.clamp(block_indices, 0, max(num_allocated - 1, 0))
        physical = block_table_tensor[clamped]
        slots = physical * block_size + block_offsets
        slots = torch.where(valid_mask, slots, torch.tensor(-1, dtype=torch.int64))
        all_slots.append(slots)

        total_for_seq = past + q_len
        blocks_needed = (total_for_seq + block_size - 1) // block_size
        seq_blocks = layer_cache.allocated_blocks[
            block_offset_for_seq : block_offset_for_seq + blocks_needed
        ]
        row = torch.tensor(seq_blocks, dtype=torch.int32)
        if row.size(0) < max_blocks_needed:
            pad = torch.zeros(max_blocks_needed - row.size(0), dtype=torch.int32)
            row = torch.cat([row, pad])
        all_block_rows.append(row[:max_blocks_needed])

    slot_mapping = torch.cat(all_slots)
    block_table = torch.stack(all_block_rows)

    isa = get_cached_isa(dtype, block_size, head_dim)

    # Paged attention writes KV via reshape_and_cache (C++ kernel) which
    # does not update PagedKVCache.seq_len.  Sync it here so that
    # remove_cache() (used by speculative decoding rollback) sees the
    # correct token count.  For batch_size > 1 with a shared cache,
    # use max_seq_len since all sequences share the same cache objects.
    target_seq_len = max_seq_len
    for co in kv_cache_manager.cache_objects:
        co.seq_len = target_seq_len

    # scheduler_metadata is intentionally left as None here.
    # PagedAttentionBackend.forward() builds it per-layer with the
    # correct sliding_window_size, matching the vLLM approach.
    return PagedAttentionMetadata(
        isa=isa,
        num_actual_tokens=total_tokens,
        max_query_len=max_query_len,
        query_start_loc=query_start_loc,
        max_seq_len=max_seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )


# ---------------------------------------------------------------------------
# ISA caching
# ---------------------------------------------------------------------------

_isa_cache: Dict[Tuple, str] = {}


def get_cached_isa(dtype: torch.dtype, block_size: int, head_dim: int = 0) -> str:
    """Return the optimal attention ISA, caching the result.

    Avoids calling into C++ on every metadata build.
    """
    key = (str(dtype), block_size, head_dim)
    if key not in _isa_cache:
        _isa_cache[key] = get_optimal_attention_isa(dtype, block_size, head_dim)
    return _isa_cache[key]


# ---------------------------------------------------------------------------
# Batched metadata builder (shared between server and potential future users)
# ---------------------------------------------------------------------------


def build_batched_paged_attention_metadata(
    layer_caches: List[PagedKVCache],
    seq_lens_list: List[int],
    model_config,
    query_len: int,
    dtype: torch.dtype,
    block_size: int,
    isa: Optional[str] = None,
) -> PagedAttentionMetadata:
    """Build PagedAttentionMetadata for a batch of heterogeneous sequences.

    Unlike :func:`build_paged_attention_metadata` which assumes all sequences
    share a single ``PagedKVCache`` and the same ``total_seq_len``, this
    function accepts a *list* of per-sequence layer caches and seq lengths.
    This is the common case in the serving path where each request has its
    own ``SharedPagedKVCache`` backed by a shared pool.

    Args:
        layer_caches: List of per-sequence layer-0 cache objects.
        seq_lens_list: List of total sequence lengths (one per request).
        model_config: HuggingFace-style model config.
        query_len: Number of new query tokens per sequence (1 for decode).
        dtype: Model dtype (used for ISA selection).
        block_size: Paged cache block size.
        isa: Pre-computed ISA string. If None, computed via
            :func:`get_cached_isa`.

    Returns:
        PagedAttentionMetadata for the batched forward pass.
    """
    num_reqs = len(layer_caches)

    text_config = resolve_text_config(model_config)
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // num_heads)

    if isa is None:
        isa = get_cached_isa(dtype, block_size, head_dim)

    all_slot_mappings = []
    all_block_tables = []
    max_seq_len = 0
    max_blocks_per_seq = 0

    # TODO: Consider vectorizing this loop by maintaining a batched block
    # table tensor at the pool level, avoiding per-cache Python iteration.
    for i, layer_cache in enumerate(layer_caches):
        total_seq_len = seq_lens_list[i]
        past_len = total_seq_len - query_len
        if total_seq_len > max_seq_len:
            max_seq_len = total_seq_len

        layer_cache._ensure_blocks_allocated(total_seq_len)
        slot_mapping = compute_slot_mapping(
            layer_cache,
            batch_size=1,
            seq_len=query_len,
            past_key_values_length=past_len,
        )
        all_slot_mappings.append(slot_mapping)

        num_blocks = len(layer_cache.allocated_blocks)
        blocks_needed = (total_seq_len + block_size - 1) // block_size
        num_blocks_to_use = min(blocks_needed, num_blocks)
        if num_blocks_to_use > 0:
            bt = torch.tensor(
                layer_cache.allocated_blocks[:num_blocks_to_use],
                dtype=torch.int32,
            )
            if num_blocks_to_use < blocks_needed:
                bt = torch.cat(
                    [
                        bt,
                        torch.zeros(
                            blocks_needed - num_blocks_to_use, dtype=torch.int32
                        ),
                    ]
                )
            block_table = bt.unsqueeze(0)
        else:
            block_table = torch.zeros(1, blocks_needed, dtype=torch.int32)
        all_block_tables.append(block_table)
        if block_table.shape[1] > max_blocks_per_seq:
            max_blocks_per_seq = block_table.shape[1]

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
    slot_mapping = torch.cat(all_slot_mappings, dim=0)

    if max_blocks_per_seq > 0:
        block_table = torch.zeros(num_reqs, max_blocks_per_seq, dtype=torch.int32)
        for i, bt in enumerate(all_block_tables):
            block_table[i, : bt.shape[1]] = bt[0]
    else:
        block_table = torch.zeros(num_reqs, 1, dtype=torch.int32)

    # scheduler_metadata is intentionally left as None here.
    # PagedAttentionBackend.forward() builds it per-layer with the
    # correct sliding_window_size, matching the vLLM approach.
    return PagedAttentionMetadata(
        isa=isa,
        num_actual_tokens=num_reqs * query_len,
        max_query_len=query_len,
        query_start_loc=query_start_loc,
        max_seq_len=max_seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )
