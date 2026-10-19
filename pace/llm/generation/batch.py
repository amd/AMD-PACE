# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from pace.llm.generation.types import GenerationState


@dataclass
class UnifiedBatchMetadata:
    """
    New tokens from several requests packed into one sequence for a single forward.

    query_lens holds each request's token count in packing order, which is how each request's logits are cut back out.
    """

    packed_input_ids: torch.Tensor
    packed_positions: torch.Tensor
    paged_attn_metadata: Any
    merged_kv_cache: Any
    query_lens: List[int]

    @property
    def total_tokens(self) -> int:
        """Tokens across every request in the batch."""
        return sum(self.query_lens)

    def extract_logits_per_request(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """Each request's logits at its last packed position, shaped [1, vocab]."""
        result = []
        offset = 0
        for query_len in self.query_lens:
            last_index = offset + query_len - 1
            result.append(logits[0, last_index : last_index + 1, :])
            offset += query_len
        return result

    def extract_full_logits_per_request(
        self, logits: torch.Tensor
    ) -> List[torch.Tensor]:
        """Each request's logits at all of its packed positions, shaped [1, query_len, vocab]."""
        result = []
        offset = 0
        for query_len in self.query_lens:
            result.append(logits[0, offset : offset + query_len, :].unsqueeze(0))
            offset += query_len
        return result


class UnifiedBatchBuilder:
    """Packs decode tokens and prefill chunks from many requests into one UnifiedBatchMetadata."""

    def __init__(self, cache_backend):
        """
        Args:
            cache_backend: Cache whose merge_contexts and merge_contexts_mixed build the batch's KV cache.
        """
        self._cache_backend = cache_backend
        self._fast_max_batch = 0
        self._fast_input_ids: Optional[torch.Tensor] = None
        self._fast_positions: Optional[torch.Tensor] = None

    @staticmethod
    def _kv_cache(state: GenerationState):
        backend_state = state.backend_state
        if backend_state is None or backend_state.kv_cache_manager is None:
            raise RuntimeError(f"request {state.request_id} has no allocated KV cache")
        return backend_state.kv_cache_manager

    def _ensure_fast_buffers(self, max_batch: int) -> None:
        if self._fast_max_batch >= max_batch:
            return
        size = max(max_batch, 128)
        self._fast_max_batch = size
        self._fast_input_ids = torch.empty(1, size, dtype=torch.long)
        self._fast_positions = torch.empty(1, size, dtype=torch.long)

    def build_decode_only_fast(
        self,
        decode_states: Sequence[GenerationState],
    ) -> UnifiedBatchMetadata:
        count = len(decode_states)
        self._ensure_fast_buffers(count)

        input_ids = self._fast_input_ids
        positions = self._fast_positions
        kv_caches = []

        for index, state in enumerate(decode_states):
            input_ids[0, index] = state.input_ids[0, -1]
            positions[0, index] = state.num_computed_tokens
            kv_caches.append(self._kv_cache(state))

        merged = self._cache_backend.merge_contexts(kv_caches, query_len=1)
        paged_metadata = getattr(merged, "paged_attn_metadata", None)
        query_lens = [1] * count
        if (
            count > 1
            and hasattr(merged, "slab_query_lens")
            and getattr(merged, "slab_query_lens", None) is None
        ):
            merged.slab_query_lens = query_lens
        return UnifiedBatchMetadata(
            packed_input_ids=input_ids[:, :count],
            packed_positions=positions[:, :count],
            paged_attn_metadata=paged_metadata,
            merged_kv_cache=merged,
            query_lens=query_lens,
        )

    def build_mixed_batch(
        self,
        prefill_states: Sequence[GenerationState],
        decode_states: Sequence[GenerationState],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]] = None,
        decode_token_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> UnifiedBatchMetadata:
        all_input_ids: List[torch.Tensor] = []
        all_positions: List[torch.Tensor] = []
        contexts_with_query_lens: List[Tuple[Any, int]] = []
        query_lens: List[int] = []

        for state in decode_states:
            spec_tokens = (
                decode_token_map.get(state.request_id)
                if decode_token_map is not None
                else None
            )
            past_len = state.num_computed_tokens
            if spec_tokens is not None:
                query_len = spec_tokens.shape[-1]
                ids = spec_tokens.squeeze(0)
                if ids.dim() == 0:
                    ids = ids.unsqueeze(0)
            else:
                query_len = 1
                ids = state.input_ids[:, -1].squeeze(0)
                if ids.dim() == 0:
                    ids = ids.unsqueeze(0)

            all_input_ids.append(ids)
            all_positions.append(
                torch.arange(past_len, past_len + query_len, dtype=torch.long)
            )
            kv_cache = self._kv_cache(state)
            contexts_with_query_lens.append((kv_cache, query_len))
            query_lens.append(query_len)

        for state in prefill_states:
            prompt_start = state.backend_state.prompt_start
            total_prompt_len = state.backend_state.actual_prompt_length
            chunk_info = (
                chunk_info_map.get(state.request_id) if chunk_info_map else None
            )
            if chunk_info is not None:
                chunk_start, chunk_size = chunk_info
                past_len = chunk_start
                query_len = min(chunk_size, total_prompt_len - past_len)
            else:
                query_len = total_prompt_len
                past_len = 0

            physical_start = prompt_start + past_len
            ids = state.prompt_token_ids.squeeze(0)[
                physical_start : physical_start + query_len
            ]
            all_input_ids.append(ids)
            all_positions.append(
                torch.arange(past_len, past_len + query_len, dtype=torch.long)
            )
            kv_cache = self._kv_cache(state)
            contexts_with_query_lens.append((kv_cache, query_len))
            query_lens.append(query_len)

        packed_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)
        packed_positions = torch.cat(all_positions, dim=0).unsqueeze(0)
        merged = self._cache_backend.merge_contexts_mixed(contexts_with_query_lens)

        return UnifiedBatchMetadata(
            packed_input_ids=packed_input_ids,
            packed_positions=packed_positions,
            paged_attn_metadata=getattr(merged, "paged_attn_metadata", None),
            merged_kv_cache=merged,
            query_lens=query_lens,
        )
