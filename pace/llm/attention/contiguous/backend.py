# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from pace.llm.attention.base import (
    AttentionBackend,
    AttentionBackendType,
)


class MaskCache:
    """Shared causal-mask buffer pool that reuses allocations across calls.

    Storage is class-level: every ``MaskCache()`` instance is a lightweight
    handle into the same pool.  Buffers are keyed by ``(shape, dtype, owner)``
    so identically-shaped masks share one allocation per owner (request),
    avoiding cross-request buffer aliasing.

    Mask contents are tracked alongside the buffer.  When the content
    parameters (seq_len, sliding_window, leading_pad) match the
    previous write, the buffer is returned immediately — giving all
    decoder layers in a single forward pass a zero-cost cache hit.
    When parameters change (next decode step or new generation), the
    buffer is rewritten in-place.

    Entries are LRU-evicted when the pool exceeds ``PACE_MASK_CACHE_MAX_ENTRIES``
    (default 1000) to cap memory from many distinct shapes.
    """

    # pool_key -> (buf, content_key); order is LRU (oldest first).
    _pool: OrderedDict[tuple, tuple[torch.Tensor, tuple]] = OrderedDict()
    _MAX_ENTRIES: int = max(1, int(os.getenv("PACE_MASK_CACHE_MAX_ENTRIES", "1000")))

    @classmethod
    def _evict_lru(cls) -> None:
        while len(cls._pool) > cls._MAX_ENTRIES:
            cls._pool.popitem(last=False)

    def get(
        self,
        bs: int,
        q_len: int,
        kv_len: int,
        seq_len: int,
        dtype: torch.dtype,
        sliding_window: int = 0,
        leading_pad: Optional[torch.Tensor] = None,
        owner: str = "",
    ) -> torch.Tensor:
        """Return a ``[bs, 1, q_len, kv_len]`` additive causal mask.

        Reuses cached content when parameters match (cross-layer sharing),
        rewrites in-place when they change (next decode step / generation).

        Args:
            owner: A unique identifier for the request or sequence that owns
                this mask, typically the UUID or request-ID assigned when the
                inference request is created.  Defaults to ``""`` for the
                non-serving (single-request) path.  Using a per-request owner
                prevents concurrent requests that happen to share the same
                ``(shape, dtype)`` from aliasing each other's mask buffer.
        """
        shape = (bs, 1, q_len, kv_len)
        pool_key = (shape, dtype, owner)
        pad_tuple = tuple(leading_pad.tolist()) if leading_pad is not None else ()
        content_key = (seq_len, sliding_window, pad_tuple)
        min_val = torch.finfo(dtype).min

        entry = MaskCache._pool.get(pool_key)
        if entry is not None:
            buf, prev_content = entry
            if prev_content == content_key:
                MaskCache._pool.move_to_end(pool_key, last=True)
                return buf
        else:
            buf = torch.empty(shape, dtype=dtype)

        # Write causal mask
        effective = seq_len if seq_len > 0 else kv_len
        buf.fill_(min_val)
        buf.triu_(diagonal=effective - q_len + 1)

        # Sliding window
        if sliding_window > 0:
            q_pos = torch.arange(kv_len - q_len, kv_len)
            k_pos = torch.arange(kv_len)
            dist = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
            outside = (dist < 0) | (dist >= sliding_window)
            buf.masked_fill_(outside.unsqueeze(0).unsqueeze(0), min_val)

        # Leading pad
        if leading_pad is not None and (leading_pad > 0).any():
            col = torch.arange(kv_len, device=leading_pad.device)
            flags = col.unsqueeze(0) < leading_pad.unsqueeze(1)  # [B, kv]
            buf.masked_fill_(flags[:, None, None, :], min_val)

        MaskCache._pool[pool_key] = (buf, content_key)
        MaskCache._pool.move_to_end(pool_key, last=True)
        MaskCache._evict_lru()
        return buf

    @classmethod
    def reset(cls) -> None:
        """Drop all cached buffers."""
        cls._pool.clear()


def _apply_sink_to_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    sinks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply attention sink bias to K/V/mask tensors.

    When the mask has a trailing -inf column (BMC trailing padding), the
    last padded KV position is repurposed as the sink — only the mask
    column is overwritten, avoiding K/V concatenation entirely.
    """
    batch_size, num_kv_heads, seq_len, head_dim = key.shape
    num_q_heads = query.shape[1]
    query_len = query.shape[2]
    sink_col = (
        sinks.to(dtype=mask.dtype)
        .reshape(1, -1, 1, 1)
        .expand(batch_size, num_q_heads, query_len, 1)
    )
    mask = mask.expand(batch_size, num_q_heads, -1, -1)

    min_val = torch.finfo(mask.dtype).min
    if bool(mask[0, 0, -1, -1] <= min_val / 2):
        mask = mask.clone()
        mask[..., -1:] = sink_col
    else:
        key = torch.cat(
            [
                key,
                torch.zeros(
                    batch_size,
                    num_kv_heads,
                    1,
                    head_dim,
                    dtype=key.dtype,
                    device=key.device,
                ),
            ],
            dim=2,
        )
        value = torch.cat(
            [
                value,
                torch.zeros(
                    batch_size,
                    num_kv_heads,
                    1,
                    head_dim,
                    dtype=value.dtype,
                    device=value.device,
                ),
            ],
            dim=2,
        )
        mask = torch.cat([mask.contiguous(), sink_col], dim=-1)

    return key, value, mask


def _compute_pad_lens(positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Leading left-pad count per sequence: ``seq_len - (last_pos + 1)``."""
    return seq_len - (positions[:, -1] + 1)


def _can_use_brgemm_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    sinks: Optional[torch.Tensor],
    sliding_window: int = 0,
) -> bool:
    """Check if BRGeMM tiled prefill is applicable (dtype/shape checks).

    Padding is handled inside the C++ op via per-sequence query_lens,
    so no padding check is needed here.
    """
    # Take batched SDPA path for smaller inputs, non-bf16, sinks, sliding window
    if (
        query.size(2) < 512
        or query.dtype != torch.bfloat16
        or sinks is not None
        or sliding_window > 0
        or query.size(1) % key.size(1) != 0
    ):
        return False
    return True


def _extract_seq_ranges(
    positions: torch.Tensor,
    q_padded: int,
) -> tuple[list[int], list[int]]:
    """Extract per-sequence real token start and count from positions.

    Assumes left-padding: real tokens are contiguous at the end of
    each sequence, so positions[:, -1] is the last real token's
    position index. This matches the BMC contiguous cache convention.
    Returns (query_offsets, query_lens).
    Returns ([], []) if no padding detected (all tokens real).
    """
    B = positions.size(0)
    last_pos = positions[:, -1]
    expected_last = q_padded - 1
    if (last_pos == expected_last).all():
        return [], []

    offsets = []
    lengths = []
    for b in range(B):
        ql = int(last_pos[b].item()) + 1
        qo = q_padded - ql
        offsets.append(qo)
        lengths.append(ql)
    return offsets, lengths


class ContiguousAttentionBackend(AttentionBackend):
    """Attention backend for contiguous KV caches (Dynamic, BMC).

    Masks are fully managed here.  A single ``_get_mask`` method handles
    causal, sliding-window, leading-padding, and BMC-trailing-padding in
    one call.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_backend: AttentionBackendType,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        use_gqa = num_heads != num_kv_heads
        self._use_jit = attn_backend == AttentionBackendType.JIT
        self._has_sliding = sliding_window is not None and sliding_window > 0
        self._has_sinks = sinks is not None
        self.sinks = sinks
        self.sliding_window = sliding_window if self._has_sliding else 0
        self._mask_cache = MaskCache()

        if self._use_jit:
            self._attn_fn = self._jit_gqa if use_gqa else self._jit_mha
            self._attn_list_fn = self._jit_gqa_list if use_gqa else self._jit_mha_list
        else:
            self._enable_gqa = use_gqa
            self._attn_fn = self._native_sdpa
            self._attn_list_fn = self._native_sdpa_list

    # Unified mask
    def _get_mask(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        leading_pad: torch.Tensor,
        trailing_pad: int,
        owner: str = "",
    ) -> Optional[torch.Tensor]:
        """Return a [B,1,q,kv] additive mask, or None when not needed.

        Uses the shared :class:`MaskCache` buffer pool.  Buffers are
        reused across calls, with contents preserved on a cache hit
        (when content parameters match) and rewritten in-place when
        those parameters change.

        Args:
            owner: UUID or request-ID of the owning sequence (see
                :meth:`MaskCache.get`).  Pass ``getattr(kv_cache, "token", "")``
                so each concurrent request gets an isolated buffer.
        """
        bs, q_len, kv_len = Q.shape[0], Q.shape[2], K.shape[2]
        seq_len = kv_len - trailing_pad
        has_leading_pad = (leading_pad > 0).any()
        has_pad = trailing_pad > 0 or has_leading_pad

        needs_mask = (
            has_pad
            or self._has_sliding
            or self._has_sinks
            or (q_len > 1 and self._use_jit)
        )
        if not needs_mask:
            return None

        lp = leading_pad if has_leading_pad else None
        return self._mask_cache.get(
            bs, q_len, kv_len, seq_len, Q.dtype, self.sliding_window, lp, owner
        )

    # Forward
    def forward(self, Q, K, V, kv_cache, positions, **kwargs):
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if isinstance(kv_cache, list):
            out = self._forward_list(Q, K, V, kv_cache, positions)
        else:
            K, V = kv_cache.update_cache(K, V, concat_dim=2)
            if _can_use_brgemm_prefill(Q, K, self.sinks, self.sliding_window):
                real_len = kv_cache.seq_len
                q_offsets, q_lens = _extract_seq_ranges(positions, Q.size(2))
                out = torch.ops.pace.prefill_attention(
                    Q,
                    K.narrow(2, 0, real_len),
                    V.narrow(2, 0, real_len),
                    q_offsets,
                    q_lens,
                )
            else:
                leading_pad = _compute_pad_lens(positions, kv_cache.seq_len)
                trailing_pad = K.shape[2] - kv_cache.seq_len
                mask = self._get_mask(
                    Q,
                    K,
                    leading_pad,
                    trailing_pad,
                    owner=getattr(kv_cache, "token", ""),
                )
                if self._has_sinks:
                    K, V, mask = _apply_sink_to_kv(Q, K, V, mask, self.sinks)
                out = self._attn_fn(Q, K, V, mask)

        return out.transpose(1, 2)

    def _forward_list(self, Q, K, V, kv_cache_list, positions):
        bsz = Q.shape[0]
        Q_list, K_list, V_list, mask_list = [], [], [], []
        for i in range(bsz):
            seq_k = K[i : i + 1]
            seq_v = V[i : i + 1]
            seq_k, seq_v = kv_cache_list[i].update_cache(seq_k, seq_v, concat_dim=2)

            lp = _compute_pad_lens(positions[i : i + 1], kv_cache_list[i].seq_len)
            tp = seq_k.shape[2] - kv_cache_list[i].seq_len

            Q_i = Q[i : i + 1]
            m = self._get_mask(
                Q_i, seq_k, lp, tp, owner=getattr(kv_cache_list[i], "token", "")
            )
            if self._has_sinks:
                seq_k, seq_v, m = _apply_sink_to_kv(Q_i, seq_k, seq_v, m, self.sinks)

            Q_list.append(Q_i)
            K_list.append(seq_k)
            V_list.append(seq_v)
            mask_list.append(m)

        return self._attn_list_fn(Q_list, K_list, V_list, mask_list)

    # Kernel dispatches
    def _jit_mha(self, Q, K, V, mask):
        return torch.ops.pace.multi_head_attention(
            Q.contiguous(), K.contiguous(), V.contiguous(), mask, False
        )

    def _jit_gqa(self, Q, K, V, mask):
        return torch.ops.pace.grouped_query_attention(
            Q.contiguous(), K.contiguous(), V.contiguous(), mask
        )

    def _native_sdpa(self, Q, K, V, mask):
        if mask is not None:
            return F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, enable_gqa=self._enable_gqa
            )
        return F.scaled_dot_product_attention(
            Q, K, V, is_causal=(Q.shape[2] > 1), enable_gqa=self._enable_gqa
        )

    def _jit_mha_list(self, Q_list, K_list, V_list, mask_list):
        resolved = self._resolve_list_masks(Q_list, mask_list)
        return torch.ops.pace.multi_head_attention_list(
            [q.contiguous() for q in Q_list],
            [k.contiguous() for k in K_list],
            [v.contiguous() for v in V_list],
            resolved,
            False,
        )

    def _jit_gqa_list(self, Q_list, K_list, V_list, mask_list):
        resolved = self._resolve_list_masks(Q_list, mask_list)
        return torch.ops.pace.grouped_query_attention_list(
            [q.contiguous() for q in Q_list],
            [k.contiguous() for k in K_list],
            [v.contiguous() for v in V_list],
            resolved,
        )

    def _resolve_list_masks(self, Q_list, mask_list):
        """Replace None with empty tensor (C++ list ops cannot accept None)."""
        return [
            m if m is not None else torch.empty(0, dtype=Q_list[i].dtype)
            for i, m in enumerate(mask_list)
        ]

    def _native_sdpa_list(self, Q_list, K_list, V_list, mask_list):
        results = [
            F.scaled_dot_product_attention(
                Q_list[i],
                K_list[i],
                V_list[i],
                attn_mask=mask_list[i],
                is_causal=(mask_list[i] is None and Q_list[i].shape[2] > 1),
                enable_gqa=self._enable_gqa,
            )
            for i in range(len(Q_list))
        ]
        return torch.cat(results, dim=0)
