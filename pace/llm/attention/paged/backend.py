# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
PagedAttentionBackend: receives pre-built PagedAttentionMetadata via kwargs,
flattens QKV from BSNH to [T,N,H], and calls the vLLM C++ free functions
(reshape_and_cache + paged_attention_with_kv_cache) directly.

Shared metadata (slot_mapping, block_table, seq_lens, query_start_loc) is
built once per step in the generator/server.  The scheduler_metadata (work
item partitioning) is built per-layer here, so each layer's sliding_window
is correctly accounted for — matching the vLLM approach and avoiding empty
KV splits that would corrupt output for models with attention sinks.
"""

import math
from typing import Optional

import torch

from pace.llm.attention.base import AttentionBackend
from pace.llm.attention.paged.ops import get_paged_attention_scheduler_metadata
from pace.utils.logging import PACE_LLM_ASSERT


class PagedAttentionBackend(AttentionBackend):

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.sinks = sinks
        # ISA depends on head_dim, so it must be resolved per-layer rather than
        # taken from the pool-wide value on the shared metadata.
        self._isa_by_dtype = {}

    def _local_metadata(self, meta, block_size: int):
        """Return chunked (local-attention) metadata for ``meta``, computed once
        per step and shared across all chunked layers of the same chunk_size.

        The transform depends only on the step's shared metadata and chunk_size,
        so we memoize it on the metadata object to avoid recomputing it for every
        local layer.
        """
        from pace.llm.attention.paged.utils import (
            make_local_attention_virtual_batches,
        )

        cache = getattr(meta, "_local_by_chunk", None)
        if cache is None:
            cache = {}
            # dataclass is mutable; stash a small per-step memo on it.
            object.__setattr__(meta, "_local_by_chunk", cache)
        key = (self.chunk_size, block_size)
        local = cache.get(key)
        if local is None:
            local = make_local_attention_virtual_batches(
                self.chunk_size, meta, block_size
            )
            cache[key] = local
        return local

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor],
        V: Optional[torch.Tensor],
        kv_cache,
        positions,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            Q: [B, S, num_heads, head_dim] — already RoPE'd, BSNH
            K: [B, S, num_kv_heads, head_dim], or None on a layer that reuses
                another layer's KV (no write, read the target's blocks)
            V: [B, S, num_kv_heads, head_dim], or None (see K)
            kv_cache: PagedKVCache (single layer cache with get_cache_tensors)
            positions: ignored (paged attention uses causal masking internally)
            **kwargs: must contain paged_attn_metadata
        Returns:
            [B, S, num_heads, head_dim]
        """
        paged_attn_metadata = kwargs["paged_attn_metadata"]

        bsz, q_len = Q.shape[0], Q.shape[1]
        num_tokens = bsz * q_len

        # Flatten BSNH [B, S, N, H] -> [T, N, H]
        Q_flat = Q.reshape(num_tokens, self.num_heads, self.head_dim)

        # Get cache tensors from the per-layer PagedKVCache
        key_cache, value_cache = kv_cache.get_cache_tensors()

        isa = self._isa_by_dtype.get(Q.dtype)
        if isa is None:
            from pace.llm.attention.paged.ops import get_optimal_attention_isa

            isa = get_optimal_attention_isa(Q.dtype, key_cache.shape[2], self.head_dim)
            self._isa_by_dtype[Q.dtype] = isa

        PACE_LLM_ASSERT(
            (K is None) == (V is None),
            "PagedAttentionBackend: K and V must both be provided or both be "
            "None (None only for a layer that reuses another layer's KV).",
        )
        # Uses the untransformed slot mapping, not the chunked-attention
        # metadata built below.
        if (
            K is not None
            and V is not None
            and not getattr(kv_cache, "kv_sharing_readonly", False)
        ):
            torch.ops.pace.paged_attention_reshape_and_cache(
                K.reshape(num_tokens, self.num_kv_heads, self.head_dim).contiguous(),
                V.reshape(num_tokens, self.num_kv_heads, self.head_dim).contiguous(),
                key_cache,
                value_cache,
                paged_attn_metadata.slot_mapping,
                isa,
            )

        # Chunked (local) attention: rewrite the metadata into per-chunk virtual
        # batches so the ordinary causal kernel yields chunked attention. The KV
        # write above already used the real slots, so only the read side changes.
        attn_meta = paged_attn_metadata
        if self.chunk_size > 0:
            attn_meta = self._local_metadata(
                paged_attn_metadata, block_size=key_cache.shape[2]
            )

        # Chunked attention is expressed via virtual batches, not a window, so
        # no sliding window is applied when chunking.
        sw = 0 if self.chunk_size > 0 else self.sliding_window
        sw_left = sw - 1 if sw > 0 else -1
        sw_right = 0

        # Build scheduler metadata per-layer with the correct sliding window.
        # This matches the vLLM approach: the scheduler partitions KV work
        # items using the layer's actual sliding_window_size, so it never
        # creates splits outside the window — avoiding empty KV splits that
        # would corrupt output for models with attention sinks (e.g. GPT-OSS).
        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=attn_meta.seq_lens.shape[0],
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            seq_lens=attn_meta.seq_lens,
            dtype=Q.dtype,
            query_start_loc=attn_meta.query_start_loc,
            causal=attn_meta.causal,
            sliding_window_size=sw if sw > 0 else -1,
            isa=isa,
            enable_kv_split=True,
        )

        s_aux = self.sinks
        if s_aux is not None:
            if s_aux.dtype != torch.bfloat16:
                s_aux = s_aux.to(torch.bfloat16)
            remainder = s_aux.numel() % 16
            if remainder != 0:
                s_aux = torch.nn.functional.pad(s_aux, (0, 16 - remainder))
            s_aux = s_aux.contiguous()

        output = torch.empty_like(Q_flat)
        torch.ops.pace.paged_attention_with_kv_cache(
            Q_flat.contiguous(),
            key_cache,
            value_cache,
            output,
            attn_meta.query_start_loc,
            attn_meta.seq_lens,
            self.scaling,
            attn_meta.causal,
            None,
            sw_left,
            sw_right,
            attn_meta.block_table,
            0.0,
            scheduler_metadata,
            s_aux,
        )

        # Packed (ragged) inputs arrive as [1, T, N, H] — the kernel uses
        # query_start_loc to dispatch per-sequence, so this view round-trips correctly.
        return output.view(bsz, q_len, self.num_heads, self.head_dim)
