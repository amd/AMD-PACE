# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Unified batch builder for mixed prefill + decode batches.

Builds packed input tensors and attention metadata for batches that contain
both prefill (multi-token) and decode (single-token) requests in a single
forward pass.  Supports both PAGED and SLAB_POOL cache backends.

Both backends use **packed** ``[1, T]`` input tensors (zero padding waste).

- **Paged:** delegates to ``PagedCache.merge_contexts_mixed()`` which produces
  ``PagedAttentionMetadata`` (block tables, slot mappings, scheduler metadata).
- **SLAB:** delegates to ``SlabCache.merge_contexts_mixed()`` which produces a
  ``SlabPoolContext`` carrying ``slab_query_lens``.  The attention backend
  squeezes the packed 4D tensors to 3D so the C++ kernel uses its native
  packed path with ``token_counts``/``query_lens``.

Used by ModelExecutor.prefill_decode_mixed_batch().
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class UnifiedBatchMetadata:
    """Packed tensors and metadata for a mixed prefill+decode batch.

    Both Paged and SLAB use packed ``[1, T]`` tensors.
    """

    packed_input_ids: torch.Tensor
    packed_positions: torch.Tensor
    paged_attn_metadata: Any
    merged_kv_cache: Any
    batch_kv_caches: Any
    query_lens: List[int]
    seq_lens: List[int]
    slab_query_lens: Optional[List[int]] = None

    @property
    def num_requests(self) -> int:
        return len(self.query_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def has_prefill(self) -> bool:
        return any(q > 1 for q in self.query_lens)

    @property
    def has_decode(self) -> bool:
        return any(q == 1 for q in self.query_lens)

    def extract_logits_per_request(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """Extract last-token logits for each request from packed output.

        Args:
            logits: ``[1, total_tokens, vocab_size]``

        Returns:
            List of ``[1, vocab_size]`` tensors, one per request.
        """
        result = []
        offset = 0
        for ql in self.query_lens:
            last_idx = offset + ql - 1
            result.append(logits[0, last_idx : last_idx + 1, :])
            offset += ql
        return result

    def extract_full_logits_per_request(
        self, logits: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract all logits for each request from packed output.

        Returns the full ``[1, query_len, vocab_size]`` slice per request.
        Needed for speculative-decode verification where all K+1 logit
        positions must be compared against draft predictions.

        Args:
            logits: ``[1, total_tokens, vocab_size]``

        Returns:
            List of ``[1, query_len_i, vocab_size]`` tensors, one per request.
        """
        result = []
        offset = 0
        for ql in self.query_lens:
            result.append(logits[0, offset : offset + ql, :].unsqueeze(0))
            offset += ql
        return result


class UnifiedBatchBuilder:
    """Builds packed batches for mixed prefill+decode forward passes.

    Coordinates between Sequence objects and the cache backend to produce
    tensors needed by ``generator.step()``.  Works with PAGED and SLAB_POOL.

    Ordering convention: decode-first.  Decode requests occupy indices
    ``0..n_decode-1``, prefill requests ``n_decode..n_decode+n_prefill-1``.
    """

    def __init__(self, cache_backend, model_config):
        self._cache_backend = cache_backend
        self._model_config = model_config

        self._fast_max_batch = 0
        self._fast_input_ids: Optional[torch.Tensor] = None
        self._fast_positions: Optional[torch.Tensor] = None

    def _ensure_fast_buffers(self, max_batch: int) -> None:
        if self._fast_max_batch >= max_batch:
            return
        n = max(max_batch, 128)
        self._fast_max_batch = n
        self._fast_input_ids = torch.empty(1, n, dtype=torch.long)
        self._fast_positions = torch.empty(1, n, dtype=torch.long)

    def build_decode_only_fast(
        self,
        decode_sequences: List[Any],
    ) -> UnifiedBatchMetadata:
        """Optimised path for decode-only steps (all query_len=1).

        Uses merge_contexts (uniform query_len=1) with pre-allocated buffers.
        Output is always ``[1, n]`` (packed).
        """
        n = len(decode_sequences)
        self._ensure_fast_buffers(n)

        input_ids = self._fast_input_ids
        positions = self._fast_positions
        kv_caches = []

        for i, req in enumerate(decode_sequences):
            input_ids[0, i] = req.input_encoded.input_ids[0, -1]
            positions[0, i] = req.num_computed_tokens
            kv_caches.append(req.kv_cache_manager)

        merged = self._cache_backend.merge_contexts(kv_caches, query_len=1)

        paged_meta = getattr(merged, "paged_attn_metadata", None)

        query_lens = [1] * n
        if (
            n > 1
            and hasattr(merged, "slab_query_lens")
            and getattr(merged, "slab_query_lens", None) is None
        ):
            merged.slab_query_lens = query_lens
        slab_qlens = getattr(merged, "slab_query_lens", None)
        if paged_meta is not None:
            seq_lens = [int(paged_meta.seq_lens[i].item()) for i in range(n)]
        else:
            seq_lens = [
                int(req.input_encoded.input_ids.shape[1]) for req in decode_sequences
            ]

        return UnifiedBatchMetadata(
            packed_input_ids=input_ids[:, :n],
            packed_positions=positions[:, :n],
            paged_attn_metadata=paged_meta,
            merged_kv_cache=merged,
            batch_kv_caches=kv_caches,
            query_lens=query_lens,
            seq_lens=seq_lens,
            slab_query_lens=slab_qlens,
        )

    def build_mixed_batch(
        self,
        prefill_sequences: List[Any],
        decode_sequences: List[Any],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]] = None,
        decode_token_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> UnifiedBatchMetadata:
        """Build packed metadata for a mixed prefill+decode batch.

        Decode-first ordering: decode requests at indices 0..n_dec-1,
        prefill requests at n_dec..n_dec+n_pre-1.

        Args:
            prefill_sequences: Sequences needing prefill (full or chunked).
            decode_sequences: Sequences in decode phase.
            chunk_info_map: Optional ``{req_id_str: (chunk_start, chunk_size)}``
                for chunked prefill.
            decode_token_map: Optional ``{req_id_str: tensor([tok0, ..., tokK])}``
                with extended input tokens for speculative-decode sequences.
                When present, the decode sequence contributes ``K+1`` tokens
                (last accepted + K speculated) instead of the default 1.

        Returns:
            ``UnifiedBatchMetadata`` ready for ``model.step()``.
        """
        all_input_ids: List[torch.Tensor] = []
        all_positions: List[torch.Tensor] = []
        contexts_with_query_lens: List[Tuple[Any, int]] = []
        kv_caches: List[Any] = []
        query_lens: List[int] = []
        seq_lens: List[int] = []

        for req in decode_sequences:
            req_id_str = str(req.id)
            spec_tokens = (
                decode_token_map.get(req_id_str)
                if decode_token_map is not None
                else None
            )
            if spec_tokens is not None:
                query_len = spec_tokens.shape[-1]
                ids = spec_tokens.squeeze(0)
                if ids.dim() == 0:
                    ids = ids.unsqueeze(0)
                past_len = req.num_computed_tokens
                total_seq_len = past_len + query_len
                all_input_ids.append(ids)
                all_positions.append(
                    torch.arange(past_len, past_len + query_len, dtype=torch.long)
                )
            else:
                total_seq_len = req.input_encoded.input_ids.shape[1]
                past_len = total_seq_len - 1
                query_len = 1
                tok = req.input_encoded.input_ids[:, -1].squeeze(0)
                if tok.dim() == 0:
                    tok = tok.unsqueeze(0)
                all_input_ids.append(tok)
                all_positions.append(torch.tensor([past_len], dtype=torch.long))

            contexts_with_query_lens.append((req.kv_cache_manager, query_len))
            kv_caches.append(req.kv_cache_manager)
            query_lens.append(query_len)
            seq_lens.append(total_seq_len)

        for req in prefill_sequences:
            req_id_str = str(req.id)
            total_prompt_len = req.input_encoded.input_ids.shape[1]

            ci = chunk_info_map.get(req_id_str) if chunk_info_map else None
            if ci is not None:
                chunk_start, chunk_size = ci
                past_len = chunk_start
                query_len = min(chunk_size, total_prompt_len - past_len)
            else:
                query_len = total_prompt_len
                past_len = 0

            seq_len_after = past_len + query_len
            ids = req.input_encoded.input_ids.squeeze(0)[
                past_len : past_len + query_len
            ]
            all_input_ids.append(ids)
            all_positions.append(
                torch.arange(past_len, past_len + query_len, dtype=torch.long)
            )

            contexts_with_query_lens.append((req.kv_cache_manager, query_len))
            kv_caches.append(req.kv_cache_manager)
            query_lens.append(query_len)
            seq_lens.append(seq_len_after)

        packed_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)
        packed_positions = torch.cat(all_positions, dim=0).unsqueeze(0)

        merged = self._cache_backend.merge_contexts_mixed(contexts_with_query_lens)

        paged_meta = getattr(merged, "paged_attn_metadata", None)
        slab_qlens = getattr(merged, "slab_query_lens", None)

        return UnifiedBatchMetadata(
            packed_input_ids=packed_input_ids,
            packed_positions=packed_positions,
            paged_attn_metadata=paged_meta,
            merged_kv_cache=merged,
            batch_kv_caches=kv_caches,
            query_lens=query_lens,
            seq_lens=seq_lens,
            slab_query_lens=slab_qlens,
        )
