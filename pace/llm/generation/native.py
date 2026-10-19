# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers.utils import GENERATION_CONFIG_NAME

from pace.llm.attention import KVCacheType, create_cache
from pace.llm.attention.eviction.cache import EvictingCache
from pace.llm.attention.eviction.config import EvictionConfig, resolve_eviction_config
from pace.llm.attention.eviction.guards import (
    model_rotary_embedding,
    validate_eviction_setup,
)
from pace.llm.attention.prefix_cache import compute_block_hashes
from pace.llm.attention.recurrent_state import num_recurrent_layers
from pace.llm.configs import (
    OperatorConfig,
    SamplingConfig,
    SamplingMode,
    SpecDecodeConfig,
)
from pace.llm.generation.base import GenerationBackend
from pace.llm.generation.batch import UnifiedBatchBuilder
from pace.llm.generation.types import (
    GenerationProposal,
    GenerationState,
    GenerationStatus,
    TerminationReason,
)
from pace.llm.generation.validation import validate_model_inputs
from pace.llm.outputs import GeneratorOutput, SpeculativeStats
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import get_tokenizer, init_model
from pace.llm.sampler import Sampler
from pace.llm.speculative import create_speculative_decoder
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_ERROR


@dataclass
class ContiguousBatchGroup:
    kv_cache_manager: Any
    remaining_requests: int

    def release(self) -> bool:
        self.remaining_requests -= 1
        return self.remaining_requests == 0


@dataclass
class NativeRequestState:
    kv_cache_manager: Any
    cache_position: int = 0
    prompt_start: int = 0
    actual_prompt_length: int = 0
    prefix_length: int = 0
    draft_kv_cache_manager: Any = None
    draft_num_computed_tokens: Optional[torch.Tensor] = None
    accepted_speculative_tokens: List[int] = field(default_factory=list)
    prompt_logits: List[torch.Tensor] = field(default_factory=list)
    block_hashes: List[bytes] = field(default_factory=list)
    contiguous_batch_group: Optional[ContiguousBatchGroup] = None


class NativeGenerationBackend(GenerationBackend):
    @classmethod
    def for_offline(cls, model_name_or_path, **kwargs):
        cache_kwargs = dict(kwargs.pop("cache_kwargs", None) or {})
        # TODO: Re-enable offline prefix caching after transactional same-step
        # dedup and lazy pool growth have dedicated offline coverage.
        cache_kwargs.setdefault("enable_prefix_cache", False)
        kwargs["cache_kwargs"] = cache_kwargs
        return cls._from_pretrained(model_name_or_path, **kwargs)

    @classmethod
    def for_serving(
        cls,
        model_name_or_path,
        kv_cache_type: Optional[KVCacheType] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        backend = cls._from_pretrained(
            model_name_or_path,
            kv_cache_type=kv_cache_type,
            cache_kwargs=cache_kwargs,
            **kwargs,
        )
        backend.cache_backend.reserve_pool()
        return backend

    @classmethod
    def _from_pretrained(
        cls,
        model_name_or_path,
        tokenizer_name_or_path=None,
        dtype=torch.bfloat16,
        kv_cache_type: Optional[KVCacheType] = None,
        spec_config: Optional[SpecDecodeConfig] = None,
        opconfig: Optional[OperatorConfig] = None,
        disable_tqdm: bool = False,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        eviction_config: Optional[EvictionConfig] = None,
    ):
        kv_cache_type = kv_cache_type or (
            KVCacheType.PAGED if dtype == torch.float32 else KVCacheType.SLAB_POOL
        )
        model_path = resolve_model_path(model_name_or_path)
        tokenizer_path = (
            resolve_model_path(tokenizer_name_or_path)
            if tokenizer_name_or_path is not None
            else None
        )
        validate_model_inputs(model_path, tokenizer_path, dtype)

        finalized_opconfig = (
            opconfig.finalize(cache_type=kv_cache_type)
            if opconfig is not None
            else OperatorConfig().finalize(cache_type=kv_cache_type)
        )
        model = init_model(
            model_path,
            dtype=dtype,
            opconfig=finalized_opconfig,
            disable_tqdm=disable_tqdm,
        )
        tokenizer = get_tokenizer(model_path, tokenizer_path)
        resolved_cache_kwargs = dict(cache_kwargs or {})
        resolved_cache_kwargs.setdefault("dtype", dtype)

        resolved_eviction = resolve_eviction_config(eviction_config)
        if resolved_eviction is not None:
            validate_eviction_setup(
                resolved_eviction, kv_cache_type, model, spec_config
            )
            cache_backend = EvictingCache(
                resolved_eviction,
                rotary=model_rotary_embedding(model),
                kv_cache_type=kv_cache_type,
            )
        else:
            cache_backend = create_cache(
                kv_cache_type,
                model_config=model.config,
                **resolved_cache_kwargs,
            )
        resolved_spec_config = replace(spec_config) if spec_config is not None else None
        spec_decoder = (
            cls._create_speculative_decoder(
                resolved_spec_config,
                model.config,
                dtype,
                finalized_opconfig,
                kv_cache_type,
            )
            if resolved_spec_config is not None
            else None
        )
        if (
            spec_decoder is not None
            and getattr(resolved_spec_config, "draft_kv_cache_memory_gb", None) is None
            and resolved_cache_kwargs.get("kv_cache_memory_gb") is not None
        ):
            target_size = cls._kv_footprint(model.config)
            draft_size = cls._kv_footprint(spec_decoder.model.config)
            if target_size and draft_size:
                resolved_spec_config.draft_kv_cache_memory_gb = (
                    resolved_cache_kwargs["kv_cache_memory_gb"]
                    * draft_size
                    / target_size
                )
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache_backend=cache_backend,
            kv_cache_type=kv_cache_type,
            spec_decoder=spec_decoder,
            model_config=model.config,
            model_path=model_path,
        )

    @staticmethod
    def _kv_footprint(config) -> float:
        heads = getattr(config, "num_attention_heads", 1) or 1
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        return (
            getattr(config, "num_hidden_layers", 0)
            * kv_heads
            * getattr(config, "hidden_size", 0)
            / heads
        )

    @staticmethod
    def _create_speculative_decoder(
        spec_config,
        model_config,
        dtype,
        opconfig,
        kv_cache_type,
    ):
        PACE_LLM_ASSERT(
            num_recurrent_layers(model_config) == 0,
            "Speculative decoding is not supported for Gated DeltaNet "
            "(linear-attention) models such as Qwen3.5: the recurrent state "
            "cannot be rolled back on token rejection (autoregressive "
            "decoding only). Disable speculative decoding (spec_config=None) "
            "for this model.",
        )
        return create_speculative_decoder(
            spec_config,
            dtype=dtype,
            opconfig=opconfig,
            cache_type=kv_cache_type,
        )

    def __init__(
        self,
        model,
        tokenizer,
        cache_backend,
        kv_cache_type: KVCacheType = KVCacheType.SLAB_POOL,
        spec_decoder=None,
        *,
        model_config=None,
        model_path: Optional[os.PathLike] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._cache_backend = cache_backend
        self.kv_cache_type = KVCacheType.get_kv_cache_type(kv_cache_type)
        self.spec_decoder = spec_decoder
        self._model_config = model_config or self._resolve_model_config(model)
        self._model_path = self._resolve_model_path(model, model_path)
        self._base_sampling_config = SamplingConfig()
        if self._model_path is not None:
            generation_config = os.path.join(self._model_path, GENERATION_CONFIG_NAME)
            if os.path.exists(generation_config):
                self._base_sampling_config = SamplingConfig.from_pretrained(
                    generation_config
                )
        self._batch_builder = (
            UnifiedBatchBuilder(cache_backend)
            if cache_backend.supports_packing
            else None
        )

    @staticmethod
    def _resolve_model_config(model):
        if hasattr(model, "get_config"):
            return model.get_config()
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("model_config is required when model has no config")
        return config

    @staticmethod
    def _resolve_model_path(model, model_path):
        if model_path is not None:
            return os.fspath(model_path)
        path = getattr(model, "model_path", None)
        return os.fspath(path) if path is not None else None

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_config(self):
        return self._model_config

    @property
    def model_path(self):
        return self._model_path

    @property
    def cache_backend(self):
        return self._cache_backend

    @property
    def supports_packing(self) -> bool:
        return self._cache_backend.supports_packing

    @property
    def supports_mixed_batch(self) -> bool:
        return self._batch_builder is not None and (
            self.spec_decoder is None or self.spec_decoder.supports_mixed_batch
        )

    def _prepare_sampling_config(
        self,
        user_config: Optional[SamplingConfig],
        prompt_length: int,
    ) -> SamplingConfig:
        config = copy.deepcopy(self._base_sampling_config)
        config.merge_from(user_config, self.tokenizer)
        config.verify_max_new_tokens(
            prompt_length, self.model_config.max_position_embeddings
        )
        config.finalize()
        return config

    def _prepare_batch(self, states: Sequence[GenerationState]) -> None:
        if (
            not self.supports_packing
            and self.spec_decoder is not None
            and len(states) > 1
        ):
            raise ValueError(
                "batched speculative decoding requires a cache with packing support"
            )
        if states:
            max_seq_length = max(
                state.prompt_length + state.sampling_config.max_new_tokens
                for state in states
            )
            spec_headroom = (
                self.spec_decoder.num_speculative_tokens + 1
                if self.spec_decoder is not None
                else 0
            )
            if self.supports_packing:
                self._cache_backend.prepare_batch(
                    max_seq_length,
                    len(states),
                    spec_headroom,
                )
            if self.spec_decoder is not None:
                self.spec_decoder.prepare_draft_batch(
                    max_seq_length,
                    self.kv_cache_type,
                    len(states),
                )
        if not self.supports_packing and len(states) > 1:
            max_seq_length = max(
                state.prompt_length + state.sampling_config.max_new_tokens
                for state in states
            )
            shared_cache = self._cache_backend.create_context(
                self.model_config,
                max_seq_length,
                batch_size=len(states),
            )
            batch_group = ContiguousBatchGroup(
                kv_cache_manager=shared_cache,
                remaining_requests=len(states),
            )
            for state in states:
                prompt_start, actual_length = self._prompt_bounds(state)
                state.backend_state = NativeRequestState(
                    kv_cache_manager=shared_cache,
                    cache_position=actual_length,
                    prompt_start=prompt_start,
                    actual_prompt_length=actual_length,
                    block_hashes=self._prompt_block_hashes(state, prompt_start),
                    contiguous_batch_group=batch_group,
                )

    def _allocate(self, state: GenerationState) -> None:
        if self.spec_decoder is not None:
            PACE_LLM_ASSERT(
                state.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH,
                "Speculative decoding requires greedy search sampling mode "
                f"but got {state.sampling_config.sampling_mode}",
            )
        if state.backend_state is not None:
            return
        prompt_start, actual_length = self._prompt_bounds(state)
        max_seq_length = state.prompt_length + state.sampling_config.max_new_tokens
        spec_headroom = (
            self.spec_decoder.num_speculative_tokens + 1
            if self.spec_decoder is not None
            else 0
        )
        kv_cache = self._cache_backend.create_context(
            self.model_config,
            max_seq_length,
            batch_size=state.allocation_batch_size,
            spec_headroom=spec_headroom,
            token=state.request_id,
        )
        native_state = NativeRequestState(
            kv_cache_manager=kv_cache,
            cache_position=actual_length,
            prompt_start=prompt_start,
            actual_prompt_length=actual_length,
            block_hashes=self._prompt_block_hashes(state, prompt_start),
        )
        state.backend_state = native_state
        try:
            if self.spec_decoder is not None:
                draft_length = max_seq_length
                if self.kv_cache_type == KVCacheType.PAGED:
                    draft_length += self.spec_decoder.num_speculative_tokens
                native_state.draft_kv_cache_manager = (
                    self.spec_decoder.create_draft_kv_cache(
                        draft_length,
                        self.kv_cache_type,
                    )
                )
                native_state.draft_num_computed_tokens = torch.zeros(
                    1, dtype=torch.long, device=state.input_ids.device
                )
        except Exception:
            if native_state.draft_kv_cache_manager is not None:
                self.spec_decoder.remove_draft_context(
                    native_state.draft_kv_cache_manager
                )
            self._cache_backend.remove_context(kv_cache)
            state.backend_state = None
            raise

    def _prompt_block_hashes(
        self, state: GenerationState, prompt_start: int
    ) -> List[bytes]:
        if prompt_start != 0 or not getattr(
            self._cache_backend, "prefix_cache_enabled", False
        ):
            return []
        return compute_block_hashes(
            state.prompt_token_ids[0].tolist(),
            self._cache_backend._prefix_block_size,
        )

    def _prompt_bounds(self, state: GenerationState) -> Tuple[int, int]:
        prompt_length = state.prompt_length
        if state.attention_mask is not None:
            mask = state.attention_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != state.prompt_token_ids.shape:
                raise ValueError(
                    "attention_mask shape must match input_ids shape, got "
                    f"{tuple(mask.shape)} and {tuple(state.prompt_token_ids.shape)}"
                )
            if not bool(mask[0, -1]):
                raise ValueError("right-padded inputs are not supported")
            actual_length = int(mask.sum().item())
            return prompt_length - actual_length, actual_length

        pad_token_id = state.sampling_config.pad_token_id
        if (
            state.infer_padding
            and pad_token_id is not None
            and bool(state.prompt_token_ids[0, 0].eq(pad_token_id))
        ):
            leading_pads = int(
                state.prompt_token_ids[0]
                .eq(pad_token_id)
                .long()
                .cumprod(dim=-1)
                .sum()
                .item()
            )
            return leading_pads, prompt_length - leading_pads
        return 0, prompt_length

    def _claim_prefix(self, state: GenerationState) -> None:
        backend_state = state.backend_state
        backend = self._cache_backend
        if (
            backend_state.prompt_start != 0
            or state.sampling_config.return_input_logprobs
            or not getattr(backend, "prefix_cache_enabled", False)
        ):
            return
        matched_blocks, matched_tokens = backend.try_prefix_match(
            block_hashes=backend_state.block_hashes,
            query_token_count=state.prompt_length,
        )
        block_size = backend._prefix_block_size
        if matched_tokens >= state.prompt_length:
            matched_tokens -= block_size
        if matched_tokens <= 0:
            return
        matched_blocks = matched_blocks[: matched_tokens // block_size]
        backend.claim_prefix_blocks(
            backend_state.kv_cache_manager,
            backend_state.block_hashes,
            matched_blocks,
            matched_tokens,
        )
        backend_state.prefix_length = matched_tokens
        state.num_computed_tokens = matched_tokens

    def _prefill(
        self,
        states: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        if not states:
            return {}
        if self._batch_builder is not None and self.spec_decoder is None:
            return self._run_packed(states, [], chunk_info)
        if (
            self._batch_builder is None
            and self.spec_decoder is None
            and chunk_info is None
            and all(state.num_computed_tokens == 0 for state in states)
        ):
            return self._prefill_contiguous(states)
        proposals = {}
        for state in states:
            try:
                proposals[state.request_id] = self._prefill_one(
                    state,
                    chunk_info.get(state.request_id) if chunk_info else None,
                )
            except Exception as error:
                proposals[state.request_id] = self._error_proposal(error)
        return proposals

    def _prefill_contiguous(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        actual_lengths = [state.backend_state.actual_prompt_length for state in states]
        max_length = max(actual_lengths)
        input_rows = []
        position_rows = []
        for state, actual_length in zip(states, actual_lengths):
            prompt_start = state.backend_state.prompt_start
            tokens = state.prompt_token_ids[
                :, prompt_start : prompt_start + actual_length
            ]
            pad_length = max_length - actual_length
            if pad_length:
                padding = torch.full(
                    (1, pad_length),
                    int(state.sampling_config.pad_token_id),
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = torch.cat([padding, tokens], dim=1)
            input_rows.append(tokens)
            position_rows.append(
                torch.clamp(
                    torch.arange(max_length, dtype=torch.long) - pad_length,
                    min=0,
                ).unsqueeze(0)
            )
        input_ids = torch.cat(input_rows, dim=0)
        positions = torch.cat(position_rows, dim=0)
        caches = [state.backend_state.kv_cache_manager for state in states]
        merged = self._merge_contiguous_contexts(caches)
        model_output = self._model_step(input_ids, positions, merged, None)
        return self._sample_batch(
            states,
            model_output.logits[:, -1, :],
            [GenerationStatus.PREFILL_COMPLETED] * len(states),
            actual_lengths,
            [
                model_output.logits[
                    index : index + 1,
                    -actual_lengths[index] :,
                    :,
                ]
                for index in range(len(states))
            ],
        )

    def _decode(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        if not states:
            return {}
        if self._batch_builder is not None:
            return self._run_packed([], states, None)
        if self.spec_decoder is not None:
            return self._decode_speculative_contiguous(states)
        return self._decode_contiguous(states)

    def _mixed(
        self,
        prefills: Sequence[GenerationState],
        decodes: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        if not decodes:
            return self._prefill(prefills, chunk_info)
        if not prefills:
            return self._decode(decodes)
        if self.supports_mixed_batch:
            return self._run_packed(prefills, decodes, chunk_info)
        raise NotImplementedError(
            "mixed prefill/decode requires SLAB_POOL or PAGED cache"
        )

    def _effective_chunk(
        self,
        state: GenerationState,
        chunk_info: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        prefix_length = state.backend_state.prefix_length
        prompt_length = state.backend_state.actual_prompt_length
        if chunk_info is None:
            start = max(prefix_length, state.num_computed_tokens)
            return start, prompt_length - start
        start, size = chunk_info
        if start >= prompt_length:
            raise ValueError(
                f"chunk_start {start} must be below prompt length {prompt_length}"
            )
        if prefix_length > start:
            start = prefix_length
        if start != state.num_computed_tokens:
            raise ValueError(
                f"chunk_start {start} does not match computed tokens "
                f"{state.num_computed_tokens}"
            )
        if size <= 0:
            raise ValueError(f"chunk_size must be positive, got {size}")
        return start, min(size, prompt_length - start)

    def _prefill_one(
        self,
        state: GenerationState,
        chunk_info: Optional[Tuple[int, int]],
    ) -> GenerationProposal:
        backend_state = state.backend_state
        if state.num_computed_tokens == 0 and backend_state.prefix_length == 0:
            self._claim_prefix(state)
        if backend_state.prompt_start:
            start = backend_state.prompt_start
            size = backend_state.actual_prompt_length
            logical_end = state.prompt_length
        else:
            start, size = self._effective_chunk(state, chunk_info)
            logical_end = start + size
        is_final = logical_end >= state.prompt_length

        if (
            self.spec_decoder is not None
            and chunk_info is None
            and backend_state.prefix_length == 0
        ):
            return self._prefill_speculative(state, start, size)

        input_ids = state.prompt_token_ids[:, start : start + size]
        positions = torch.arange(size, dtype=torch.long).unsqueeze(0)
        if backend_state.prompt_start == 0:
            positions += start
        metadata = self._cache_backend.build_prefill_metadata(
            backend_state.kv_cache_manager, size, past_len=state.num_computed_tokens
        )
        self._ensure_token_headroom(size, prefill_count=1, decode_count=0)
        model_output = self._model_step(
            input_ids, positions, backend_state.kv_cache_manager, metadata
        )
        full_logits = None
        if state.sampling_config.return_input_logprobs:
            backend_state.prompt_logits.append(model_output.logits)
            if is_final:
                full_logits = torch.cat(backend_state.prompt_logits, dim=1)
                backend_state.prompt_logits.clear()
        computed = (
            backend_state.actual_prompt_length
            if backend_state.prompt_start
            else logical_end
        )
        if not is_final:
            return GenerationProposal(
                token_ids=torch.empty((1, 0), dtype=torch.long),
                status=GenerationStatus.PREFILL_IN_PROGRESS,
                computed_prompt_tokens=computed,
            )

        if self.spec_decoder is not None:
            self._warm_draft_cache(state)
        return self._sample_proposal(
            state,
            model_output.logits[:, -1, :].float(),
            GenerationStatus.PREFILL_COMPLETED,
            computed,
            prompt_logits=full_logits,
        )

    def _prefill_speculative(
        self, state: GenerationState, start: int, size: int
    ) -> GenerationProposal:
        backend_state = state.backend_state
        prompt = state.prompt_token_ids[:, start : start + size]
        positions = torch.arange(size, dtype=torch.long).unsqueeze(0)
        speculation = self.spec_decoder.speculate(
            prompt,
            draft_kv_cache=backend_state.draft_kv_cache_manager,
            draft_num_computed=backend_state.draft_num_computed_tokens,
            initial_positions=positions,
        )
        extended = speculation.extended_input
        speculative_count = extended.shape[-1] - prompt.shape[-1]
        target_positions = torch.arange(extended.shape[-1], dtype=torch.long).unsqueeze(
            0
        )
        metadata = self._cache_backend.build_prefill_metadata(
            backend_state.kv_cache_manager, extended.shape[-1]
        )
        self._ensure_token_headroom(extended.shape[-1], prefill_count=1, decode_count=0)
        model_output = self._model_step(
            extended,
            target_positions,
            backend_state.kv_cache_manager,
            metadata,
        )
        verification_logits = model_output.logits[
            :, -speculative_count - 1 :, :
        ].float()
        prompt_logits = model_output.logits[:, : prompt.shape[-1], :]
        return self._verify_speculation(
            state,
            verification_logits,
            extended,
            computed_before=0,
            target_query_count=extended.shape[-1],
            status=GenerationStatus.PREFILL_COMPLETED,
            prompt_logits=prompt_logits,
        )

    def _run_packed(
        self,
        prefills: Sequence[GenerationState],
        decodes: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        effective_chunks = dict(chunk_info or {})
        dedup = len(prefills) > 1
        if dedup:
            for state in prefills:
                chunk = effective_chunks.get(state.request_id)
                if chunk is not None and chunk[0] != 0:
                    self._preinsert_batch_prefix(state, *chunk)

        for state in prefills:
            requested_chunk = effective_chunks.get(state.request_id)
            if (
                state.num_computed_tokens == 0
                and state.backend_state.prefix_length == 0
            ):
                self._claim_prefix(state)
            chunk = self._effective_chunk(state, requested_chunk)
            effective_chunks[state.request_id] = chunk
            if dedup and (requested_chunk is None or requested_chunk[0] == 0):
                self._preinsert_batch_prefix(state, *chunk)

        decode_token_map = None
        speculated_inputs = None
        if self.spec_decoder is not None and decodes:
            decode_token_map, speculated_inputs = self._draft_decode(decodes)

        if not prefills and self.spec_decoder is None:
            batch = self._batch_builder.build_decode_only_fast(decodes)
        else:
            batch = self._batch_builder.build_mixed_batch(
                prefills,
                decodes,
                effective_chunks or None,
                decode_token_map,
            )
        self._ensure_pool_headroom(batch, len(prefills), len(decodes))
        model_output = self._model_step(
            batch.packed_input_ids,
            batch.packed_positions,
            batch.merged_kv_cache,
            batch.paged_attn_metadata,
        )
        last_logits = batch.extract_logits_per_request(model_output.logits)
        full_logits = batch.extract_full_logits_per_request(model_output.logits)
        proposals: Dict[str, GenerationProposal] = {}

        if self.spec_decoder is not None:
            for index, state in enumerate(decodes):
                try:
                    proposals[state.request_id] = self._verify_speculation(
                        state,
                        full_logits[index].float(),
                        speculated_inputs[state.request_id],
                        computed_before=state.num_computed_tokens,
                        target_query_count=batch.query_lens[index],
                        status=GenerationStatus.DECODING_IN_PROGRESS,
                    )
                except Exception as error:
                    proposals[state.request_id] = self._error_proposal(error)
        elif decodes:
            proposals.update(
                self._sample_batch(
                    decodes,
                    [last_logits[index] for index in range(len(decodes))],
                    [GenerationStatus.DECODING_IN_PROGRESS] * len(decodes),
                    [state.num_computed_tokens + 1 for state in decodes],
                )
            )

        decode_count = len(decodes)
        completed_prefills = []
        completed_logits = []
        completed_counts = []
        completed_prompt_logits = []
        for index, state in enumerate(prefills):
            logits_index = decode_count + index
            start, size = effective_chunks[state.request_id]
            computed = start + size
            if state.sampling_config.return_input_logprobs:
                state.backend_state.prompt_logits.append(
                    full_logits[logits_index].clone()
                )
            if computed < state.backend_state.actual_prompt_length:
                proposals[state.request_id] = GenerationProposal(
                    token_ids=torch.empty((1, 0), dtype=torch.long),
                    status=GenerationStatus.PREFILL_IN_PROGRESS,
                    computed_prompt_tokens=computed,
                )
                continue
            prompt_logits = None
            if state.sampling_config.return_input_logprobs:
                prompt_logits = torch.cat(
                    state.backend_state.prompt_logits,
                    dim=1,
                )
                state.backend_state.prompt_logits.clear()
            try:
                if self.spec_decoder is not None:
                    self._warm_draft_cache(state)
            except Exception as error:
                proposals[state.request_id] = self._error_proposal(error)
                continue
            completed_prefills.append(state)
            completed_logits.append(last_logits[logits_index])
            completed_counts.append(computed)
            completed_prompt_logits.append(prompt_logits)
        if completed_prefills:
            proposals.update(
                self._sample_batch(
                    completed_prefills,
                    completed_logits,
                    [GenerationStatus.PREFILL_COMPLETED] * len(completed_prefills),
                    completed_counts,
                    completed_prompt_logits,
                )
            )
        return proposals

    def _preinsert_batch_prefix(
        self,
        state: GenerationState,
        start: int,
        size: int,
    ) -> None:
        # TODO: Roll back preinserted hashes when the packed forward fails.
        # Also make pending hashes batch-local if engine execution becomes
        # concurrent.
        backend = self._cache_backend
        if state.backend_state.prompt_start != 0 or not getattr(
            backend, "prefix_cache_enabled", False
        ):
            return
        num_full_blocks = (start + size) // backend._prefix_block_size
        if num_full_blocks <= 0:
            return
        backend.preinsert_computed_blocks(
            state.backend_state.kv_cache_manager,
            state.backend_state.block_hashes,
            num_full_blocks,
        )

    def _ensure_pool_headroom(
        self,
        batch,
        prefill_count: int,
        decode_count: int,
    ) -> None:
        self._ensure_token_headroom(
            batch.total_tokens,
            prefill_count=prefill_count,
            decode_count=decode_count,
        )

    def _ensure_token_headroom(
        self,
        token_count: int,
        prefill_count: int,
        decode_count: int,
    ) -> None:
        manager = getattr(self._cache_backend, "_manager", None)
        if manager is None or not getattr(
            self._cache_backend, "prefix_cache_enabled", False
        ):
            return
        needed = -(-token_count // manager.block_size) + prefill_count + decode_count
        deficit = needed - manager.get_free_blocks()
        if deficit > 0:
            self._cache_backend.evict_prefix_blocks(deficit)

    def _decode_contiguous(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        input_ids = torch.cat([state.input_ids[:, -1:] for state in states], dim=0)
        positions = torch.tensor(
            [[state.num_computed_tokens] for state in states], dtype=torch.long
        )
        caches = [state.backend_state.kv_cache_manager for state in states]
        merged = self._merge_contiguous_contexts(caches)
        model_output = self._model_step(input_ids, positions, merged, None)
        return self._sample_batch(
            states,
            model_output.logits[:, -1, :],
            [GenerationStatus.DECODING_IN_PROGRESS] * len(states),
            [state.num_computed_tokens + 1 for state in states],
        )

    def _decode_fixed_batch(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        active_indices = [
            index for index, state in enumerate(states) if not state.finished
        ]
        active = [states[index] for index in active_indices]
        input_rows = []
        for state in states:
            if state.finished:
                input_rows.append(
                    torch.full(
                        (1, 1),
                        int(state.sampling_config.pad_token_id),
                        dtype=state.input_ids.dtype,
                        device=state.input_ids.device,
                    )
                )
            else:
                input_rows.append(state.input_ids[:, -1:])
        input_ids = torch.cat(input_rows, dim=0)
        positions = torch.tensor(
            [[state.backend_state.cache_position] for state in states],
            dtype=torch.long,
        )
        batch_group = states[0].backend_state.contiguous_batch_group
        shared_cache = (
            batch_group.kv_cache_manager
            if batch_group is not None
            else states[0].backend_state.kv_cache_manager
        )
        model_output = self._model_step(input_ids, positions, shared_cache, None)
        for state in states:
            state.backend_state.cache_position += 1
        logits = model_output.logits[active_indices, -1, :]
        return self._sample_batch(
            active,
            logits,
            [GenerationStatus.DECODING_IN_PROGRESS] * len(active),
            [state.num_computed_tokens + 1 for state in active],
        )

    def _merge_contiguous_contexts(self, contexts: Sequence[Any]):
        shared = contexts[0]
        if all(context is shared for context in contexts):
            return shared
        return self._cache_backend.merge_contexts(contexts)

    def _decode_speculative_contiguous(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        token_map, speculated_inputs = self._draft_decode(states)
        target_inputs = [token_map[state.request_id] for state in states]
        query_len = target_inputs[0].shape[-1]
        PACE_LLM_ASSERT(
            all(tokens.shape[-1] == query_len for tokens in target_inputs),
            "contiguous speculative decode requires a uniform query length",
        )
        input_ids = torch.cat(target_inputs, dim=0)
        positions = torch.cat(
            [
                torch.arange(
                    state.num_computed_tokens,
                    state.num_computed_tokens + query_len,
                    dtype=torch.long,
                ).unsqueeze(0)
                for state in states
            ],
            dim=0,
        )
        caches = [state.backend_state.kv_cache_manager for state in states]
        merged = self._cache_backend.merge_contexts(caches, query_len=query_len)
        model_output = self._model_step(input_ids, positions, merged, None)
        proposals = {}
        for index, state in enumerate(states):
            try:
                proposals[state.request_id] = self._verify_speculation(
                    state,
                    model_output.logits[index : index + 1].float(),
                    speculated_inputs[state.request_id],
                    computed_before=state.num_computed_tokens,
                    target_query_count=query_len,
                    status=GenerationStatus.DECODING_IN_PROGRESS,
                )
            except Exception as error:
                proposals[state.request_id] = self._error_proposal(error)
        return proposals

    def _draft_decode(
        self, states: Sequence[GenerationState]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        token_map = {}
        speculated_inputs = {}
        for state in states:
            backend_state = state.backend_state
            draft_computed = int(backend_state.draft_num_computed_tokens[0].item())
            logical_tokens = state.input_ids[:, backend_state.prompt_start :]
            missing = max(1, logical_tokens.shape[-1] - draft_computed)
            draft_input = logical_tokens[:, -missing:]
            speculation = self.spec_decoder.speculate(
                draft_input,
                draft_kv_cache=backend_state.draft_kv_cache_manager,
                draft_num_computed=backend_state.draft_num_computed_tokens,
            )
            target_count = self.spec_decoder.num_speculative_tokens + 1
            target_input = speculation.extended_input[:, -target_count:]
            token_map[state.request_id] = target_input
            speculated_inputs[state.request_id] = speculation.extended_input
        return token_map, speculated_inputs

    def _warm_draft_cache(self, state: GenerationState) -> None:
        backend_state = state.backend_state
        if int(backend_state.draft_num_computed_tokens[0].item()) != 0:
            return
        prompt = state.prompt_token_ids[
            :, backend_state.prompt_start : state.prompt_length
        ]
        positions = torch.arange(prompt.shape[-1], dtype=torch.long).unsqueeze(0)
        self.spec_decoder.draft_prefill(
            prompt,
            positions,
            backend_state.draft_kv_cache_manager,
        )
        backend_state.draft_num_computed_tokens += prompt.shape[-1]

    def _sample_proposal(
        self,
        state: GenerationState,
        logits: torch.Tensor,
        status: GenerationStatus,
        computed_tokens: int,
        *,
        prompt_logits: Optional[torch.Tensor] = None,
    ) -> GenerationProposal:
        sampler_output = state.sampler.sample(state.input_ids, logits)
        token_ids = sampler_output.next_tokens.reshape(1, -1)
        prompt_logprobs = self._prompt_logprobs(state, prompt_logits)
        return GenerationProposal(
            token_ids=token_ids,
            status=status,
            computed_prompt_tokens=computed_tokens,
            probs=(
                sampler_output.probs if state.sampling_config.return_probs else None
            ),
            logprobs=(
                sampler_output.logprobs
                if state.sampling_config.return_logprobs
                else None
            ),
            prompt_logprobs=prompt_logprobs,
        )

    @staticmethod
    def _error_proposal(error: Exception) -> GenerationProposal:
        PACE_LLM_ERROR(str(error))
        return GenerationProposal(
            token_ids=torch.empty((1, 0), dtype=torch.long),
            status=GenerationStatus.ERROR,
            finish_reason="error",
            error=str(error),
        )

    def _sample_batch(
        self,
        states: Sequence[GenerationState],
        logits: Union[torch.Tensor, Sequence[torch.Tensor]],
        statuses: Sequence[GenerationStatus],
        computed_tokens: Sequence[int],
        prompt_logits: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Dict[str, GenerationProposal]:
        prompt_logits = prompt_logits or [None] * len(states)
        batched_input = isinstance(logits, torch.Tensor)

        def logits_for(index: int) -> torch.Tensor:
            if batched_input:
                return logits[index : index + 1]
            return logits[index]

        batch_sample = Sampler.sample_batch(
            [state.sampler for state in states],
            [state.input_ids for state in states],
            logits,
        )
        if batch_sample is None:
            # TODO: Batch compatible non-greedy requests without sharing RNG,
            # penalty history, or failure state across requests.
            proposals = {}
            for index, state in enumerate(states):
                try:
                    proposals[state.request_id] = self._sample_proposal(
                        state,
                        logits_for(index).float(),
                        statuses[index],
                        computed_tokens[index],
                        prompt_logits=prompt_logits[index],
                    )
                except Exception as error:
                    proposals[state.request_id] = self._error_proposal(error)
            return proposals

        token_ids, valid_rows = batch_sample
        proposals = {}
        for index, state in enumerate(states):
            if not bool(valid_rows[index]):
                proposals[state.request_id] = self._error_proposal(
                    ValueError("invalid logits contain NaN")
                )
                continue
            try:
                proposals[state.request_id] = GenerationProposal(
                    token_ids=token_ids[index : index + 1].reshape(1, 1),
                    status=statuses[index],
                    computed_prompt_tokens=computed_tokens[index],
                    prompt_logprobs=self._prompt_logprobs(state, prompt_logits[index]),
                )
            except Exception as error:
                proposals[state.request_id] = self._error_proposal(error)
        return proposals

    def _verify_speculation(
        self,
        state: GenerationState,
        logits: torch.Tensor,
        speculated_input: torch.Tensor,
        *,
        computed_before: int,
        target_query_count: int,
        status: GenerationStatus,
        prompt_logits: Optional[torch.Tensor] = None,
    ) -> GenerationProposal:
        backend_state = state.backend_state
        sampler_output = state.sampler.sample(state.input_ids, logits)
        sampled_tokens = sampler_output.next_tokens
        if sampled_tokens.dim() == 3:
            sampled_tokens = sampled_tokens.squeeze(-1)
        verification = self.spec_decoder.verify(
            sampled_tokens,
            speculated_input,
            draft_kv_cache=backend_state.draft_kv_cache_manager,
            draft_num_computed=backend_state.draft_num_computed_tokens,
        )
        accepted = verification.accepted_tokens
        if verification.target_kv_cache_trim > 0:
            backend_state.kv_cache_manager.remove_cache(
                verification.target_kv_cache_trim
            )
        accepted_count = accepted.shape[-1]
        backend_state.accepted_speculative_tokens.append(accepted_count)
        computed = (
            computed_before + target_query_count - verification.target_kv_cache_trim
        )
        return GenerationProposal(
            token_ids=accepted,
            status=status,
            computed_prompt_tokens=computed,
            probs=(
                self._trim_sampler_tensor(sampler_output.probs, accepted_count)
                if state.sampling_config.return_probs
                else None
            ),
            logprobs=(
                self._trim_sampler_tensor(sampler_output.logprobs, accepted_count)
                if state.sampling_config.return_logprobs
                else None
            ),
            prompt_logprobs=self._prompt_logprobs(state, prompt_logits),
        )

    @staticmethod
    def _trim_sampler_tensor(
        tensor: Optional[torch.Tensor], accepted_count: int
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() == 3:
            return tensor[:, :accepted_count, :]
        return tensor

    @staticmethod
    def _prompt_logprobs(
        state: GenerationState, logits: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if logits is None or not state.sampling_config.return_input_logprobs:
            return None
        values = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        return values.squeeze(0)

    def _model_step(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        paged_attn_metadata=None,
    ):
        kwargs = (
            {"paged_attn_metadata": paged_attn_metadata}
            if paged_attn_metadata is not None
            else {}
        )
        return self.model(inputs, positions, kv_cache=kv_cache, **kwargs)

    def _finalize_proposal(
        self,
        state: GenerationState,
        proposal: GenerationProposal,
        keep_count: int,
        stop_reason: Optional[str] = None,
    ) -> None:
        proposed_count = proposal.token_ids.numel()
        discard_count = proposed_count - keep_count
        if discard_count > 0 and self.spec_decoder is not None:
            # Shared stopping only discards accepted tokens when the request is
            # terminal. Both target and draft contexts are released immediately,
            # so neither cache needs another asymmetric trim here.
            PACE_LLM_ASSERT(
                keep_count == state.remaining_tokens or stop_reason is not None,
                "discarding accepted speculative tokens must finish the request",
            )
            proposal.computed_prompt_tokens -= discard_count

        if (
            state.backend_state.prompt_start == 0
            and proposal.status
            in {
                GenerationStatus.PREFILL_IN_PROGRESS,
                GenerationStatus.PREFILL_COMPLETED,
            }
            and getattr(self._cache_backend, "prefix_cache_enabled", False)
        ):
            computed = min(proposal.computed_prompt_tokens, state.prompt_length)
            if computed > 0:
                num_blocks = computed // self._cache_backend._prefix_block_size
                self._cache_backend.insert_prefix(
                    state.backend_state.kv_cache_manager,
                    block_hashes=state.backend_state.block_hashes[:num_blocks],
                )

    def _release(self, state: GenerationState, reason: TerminationReason) -> None:
        backend_state = state.backend_state
        if backend_state is None:
            return
        backend_state.prompt_logits.clear()
        batch_group = backend_state.contiguous_batch_group
        if batch_group is not None:
            if batch_group.release():
                self._cache_backend.remove_context(batch_group.kv_cache_manager)
                batch_group.kv_cache_manager = None
            backend_state.kv_cache_manager = None
        elif backend_state.kv_cache_manager is not None:
            self._cache_backend.remove_context(backend_state.kv_cache_manager)
            backend_state.kv_cache_manager = None
        if (
            backend_state.draft_kv_cache_manager is not None
            and self.spec_decoder is not None
        ):
            self.spec_decoder.remove_draft_context(backend_state.draft_kv_cache_manager)
            backend_state.draft_kv_cache_manager = None

    def _build_output(self, states: Sequence[GenerationState]) -> GeneratorOutput:
        output = super()._build_output(states)
        accepted = [
            count
            for state in states
            for count in state.backend_state.accepted_speculative_tokens
        ]
        if accepted:
            output.speculative_stats = SpeculativeStats(
                total_speculated_tokens=sum(accepted),
                mean_accepted_tokens=sum(accepted) / len(accepted),
            )
        return output
