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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.scale = 1.0 / (head_dim**0.5)

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
            Q: [BS, q_len, num_q_heads, head_dim]
            K: [BS, q_len, num_kv_heads, head_dim]
            V: [BS, q_len, num_kv_heads, head_dim]
            kv_cache: SlabPoolLayerView instance
            positions: [BS, q_len] absolute position indices.
                Used to derive per-sequence actual lengths so that
                left-padding tokens are excluded from the KV cache.
        """
        layer_view: SlabPoolLayerView = kv_cache
        seq_len = Q.shape[1]

        # Derive per-sequence actual lengths from positions.
        # C++ uses these to skip left-padding via source offsets.
        actual_lens = (positions[:, -1] + 1).clamp(max=seq_len).tolist()

        layer_view.update(K.contiguous(), V.contiguous(), seq_lens=actual_lens)
        return layer_view.attend(
            Q.contiguous(),
            self.scale,
            seq_lens=actual_lens,
            sliding_window=self.sliding_window,
            sinks=self.sinks,
        )
