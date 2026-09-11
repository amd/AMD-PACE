# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
SlabAttentionBackend: Attention backend for SlabPool KV cache.

Routes Q/K/V through the SlabPoolLayerView for cache update + attention.
Input: BSHD [BS, q_len, num_heads, head_dim] (already projected and RoPE'd).
"""

from typing import Optional

import torch

from pace.llm.attention.base import AttentionBackend
from pace.llm.attention.slab.cache import SlabPoolLayerView
from pace.utils.logging import PACE_LLM_ASSERT


class SlabAttentionBackend(AttentionBackend):
    """
    Attention backend for slab-allocated block KV caches.

    All dispatch decisions are made at __init__ time:
    - sliding_window and sinks are stored as attributes
    - forward() delegates cache update + attention to SlabPoolLayerView
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_backend,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.sinks = sinks
        self.scale = scale if scale is not None else 1.0 / (head_dim**0.5)

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
            Q: [BS, q_len, num_q_heads, head_dim]
            K: [BS, q_len, num_kv_heads, head_dim], or None on a layer that
                reuses another layer's KV (no store, read the target's pool)
            V: [BS, q_len, num_kv_heads, head_dim], or None (see K)
            kv_cache: SlabPoolLayerView instance
            positions: [BS, q_len] absolute position indices.
                Used to derive per-sequence actual lengths so that
                left-padding tokens are excluded from the KV cache.
        """
        layer_view: SlabPoolLayerView = kv_cache
        seq_len = Q.shape[1]
        packed = layer_view.query_lens is not None

        if packed:
            actual_lens = layer_view.query_lens
            Q = Q.squeeze(0).contiguous()
        else:
            actual_lens = (positions[:, -1] + 1).clamp(max=seq_len).tolist()
            Q = Q.contiguous()

        PACE_LLM_ASSERT(
            (K is None) == (V is None),
            "SlabAttentionBackend: K and V must both be provided or both be "
            "None (None only for a layer that reuses another layer's KV).",
        )
        if K is not None and V is not None:
            if packed:
                K = K.squeeze(0).contiguous()
                V = V.squeeze(0).contiguous()
            else:
                K = K.contiguous()
                V = V.contiguous()
            layer_view.update(K, V, seq_lens=actual_lens)
        attn_out = layer_view.attend(
            Q,
            self.scale,
            seq_lens=actual_lens,
            sliding_window=self.sliding_window,
            sinks=self.sinks,
            chunk_size=self.chunk_size,
        )
        return attn_out.unsqueeze(0) if packed else attn_out
