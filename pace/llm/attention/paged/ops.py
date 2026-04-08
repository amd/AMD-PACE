# ******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
#
# Inspired by vLLM's paged attention design
# (https://github.com/vllm-project/vllm)
# ******************************************************************************

"""
CPU Paged Attention APIs for PACE

This module provides vLLM-style paged attention operations optimized for CPU.
Paged attention enables efficient memory management by storing KV cache in
fixed-size blocks that can be allocated and deallocated dynamically.

Main APIs:
- paged_attention_with_kv_cache: Main attention operation with paged KV cache
- paged_attention_reshape_and_cache: Update paged KV cache with new KV tensors
- get_paged_attention_scheduler_metadata: Pre-compute scheduling metadata
"""

from typing import Optional, Tuple
from dataclasses import dataclass

import torch

# Import pace C++ ops (required for paged attention kernels)
import pace._C  # noqa: F401


@dataclass
class PagedAttentionMetadata:
    """Metadata for paged attention computation.

    Attributes:
        isa: ISA hint for optimized implementation ("vec", "vec16")
        num_actual_tokens: Number of tokens excluding padding
        max_query_len: Maximum query length in the batch
        query_start_loc: Cumulative query positions [num_reqs + 1]
        max_seq_len: Maximum sequence length
        seq_lens: Sequence lengths [num_reqs]
        block_table: Block table mapping [num_reqs, max_blocks_per_seq]
        slot_mapping: Mapping from token positions to cache slots [num_tokens]
        scheduler_metadata: Pre-computed scheduler metadata tensor
        causal: Whether to use causal masking
    """

    isa: str
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: Optional[torch.Tensor] = None
    causal: bool = True


def get_optimal_attention_isa(
    dtype: torch.dtype, block_size: int, head_dim: int = 0
) -> str:
    """Determine the optimal ISA for the current CPU.

    Uses vLLM ISA conventions: "vec" (AVX2/AVX512), "vec16".
    Considers head_dim: if head_dim % 32 != 0 but head_dim % 16 == 0, uses "vec16".
    """
    try:
        return torch.ops.pace.get_optimal_attention_isa(dtype, block_size, head_dim)
    except (RuntimeError, AttributeError) as e:
        raise RuntimeError(
            "Paged attention C++ ops not available. "
            "Ensure the pace package is built with C++ extensions."
        ) from e


def get_paged_attention_scheduler_metadata(
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool = True,
    sliding_window_size: int = -1,
    isa: str = "auto",
    enable_kv_split: bool = True,
) -> torch.Tensor:
    """Get scheduler metadata for paged attention.

    Pre-computes scheduling metadata for efficient parallel attention computation.
    Callers must ensure seq_lens and query_start_loc are int32 and contiguous.
    """
    return torch.ops.pace.get_paged_attention_scheduler_metadata(
        num_reqs,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        dtype,
        query_start_loc,
        causal,
        sliding_window_size,
        isa,
        enable_kv_split,
    )


def paged_attention_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str = "auto",
) -> None:
    """Reshape and cache key/value tensors into paged KV cache.

    Callers must ensure key/value are contiguous and slot_mapping is int64 contiguous.
    """
    torch.ops.pace.paged_attention_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping, isa
    )


def paged_attention_with_kv_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: Optional[torch.Tensor] = None,
) -> None:
    """CPU paged attention with KV cache.

    Callers must ensure query is contiguous, and query_start_loc/seq_lens
    are int32 contiguous tensors. block_table will be made contiguous if needed
    (it is often a non-contiguous 2D slice of a pre-allocated tensor).

    Note: alibi_slopes is accepted for vLLM kernel compatibility but is not
    used by any model currently supported in PACE. Pass None.
    """
    if alibi_slopes is not None:
        raise NotImplementedError(
            "ALiBi attention is not supported. All PACE models use RoPE or "
            "learned positional embeddings. Pass alibi_slopes=None."
        )
    if not block_table.is_contiguous():
        block_table = block_table.contiguous()
    torch.ops.pace.paged_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        alibi_slopes,
        sliding_window[0],
        sliding_window[1],
        block_table,
        softcap,
        scheduler_metadata,
        s_aux,
    )


__all__ = [
    "PagedAttentionMetadata",
    "paged_attention_with_kv_cache",
    "paged_attention_reshape_and_cache",
    "get_paged_attention_scheduler_metadata",
    "get_optimal_attention_isa",
]
