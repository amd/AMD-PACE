# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from pace.llm.attention import KVCacheType, KVCacheManager
from pace.llm.attention.paged.utils import (
    create_paged_kv_cache_manager,
    build_paged_attention_metadata,
)
from pace.llm.configs import (
    SpecDecodeConfig,
    PardSpecDecodeConfig,
)
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import init_model
from pace.llm.outputs import ModelOutput
from pace.llm.generation.validation import validate_model_inputs
from pace.utils.logging import PACE_LLM_INFO, PACE_LLM_ASSERT


@dataclass
class SpeculationOutput:
    """
    Output of the speculate step. Contains the extended input (original +
    speculated tokens).
    """

    extended_input: torch.Tensor


@dataclass
class VerificationOutput:
    """
    Output of the verify step. Contains the accepted tokens and
    instructions for the caller to adjust target model state.

    Attributes:
        accepted_tokens: Token IDs that passed verification.
        target_kv_cache_trim: Number of entries to remove from the target
            model's KV cache (0 means no removal needed).
    """

    accepted_tokens: torch.Tensor
    target_kv_cache_trim: int


class SpeculativeDecoder(ABC):
    """
    Abstract base class for speculative decoding algorithms.

    Subclasses implement a specific speculation strategy (e.g. draft-model
    based, Medusa heads, lookahead, etc.).

    A single decoder instance owns the draft model. Per-request cache and
    position state is supplied explicitly by the generation backend.
    """

    supports_mixed_batch = False

    @property
    @abstractmethod
    def num_speculative_tokens(self) -> int:
        """Number of speculative tokens generated per step."""
        ...

    @abstractmethod
    def prepare_draft_batch(
        self,
        max_seq_length: int,
        kv_cache_type: KVCacheType,
        batch_size: int,
    ) -> None:
        """Prepare shared draft-cache capacity before contexts are created."""
        ...

    @abstractmethod
    def create_draft_kv_cache(
        self,
        max_seq_length: int,
        kv_cache_type: KVCacheType,
    ) -> KVCacheManager:
        """Create a per-sequence draft KV cache (used by serving)."""
        ...

    def remove_draft_context(self, context) -> None:
        pass

    @abstractmethod
    def draft_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_kv_cache: KVCacheManager,
    ) -> None:
        """Run draft model forward on the full prompt to warm up KV cache.
        No sampling -- just fills the draft KV cache with prompt context."""
        ...

    @abstractmethod
    def prepare_draft_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Build the draft-model input from the last generated token.

        Encapsulates algorithm-specific input construction (e.g. PARD token
        padding) so the generation backend does not need to know the details.
        """
        ...

    @abstractmethod
    def speculate(
        self,
        model_input: torch.Tensor,
        draft_kv_cache: KVCacheManager,
        draft_num_computed: torch.Tensor,
        initial_positions: Optional[torch.Tensor] = None,
    ) -> SpeculationOutput:
        """Generate speculated tokens for a *single* sequence.

        When ``initial_positions`` is provided (prefill with left-padded
        inputs), padding-aware positions are constructed so the draft model
        does not attend to padding tokens.
        """
        ...

    @abstractmethod
    def verify(
        self,
        sampled_tokens: torch.Tensor,
        speculated_input: torch.Tensor,
        draft_kv_cache: KVCacheManager,
        draft_num_computed: torch.Tensor,
    ) -> VerificationOutput:
        """Verify speculated tokens for a *single* sequence.

        Cleans up the draft KV cache (passed-in or internal).  Returns
        instructions for the caller to adjust the *target* KV cache.
        """
        ...


# ---------------------------------------------------------------------------
# PARD (PARallel Draft) speculative decoding
# ---------------------------------------------------------------------------


class PardSpeculativeDecoder(SpeculativeDecoder):
    """
    Speculative decoder using the PARD (PARallel Draft) algorithm.

    Initialised **once** (loads the draft model).  Per-sequence state
    (draft KV cache, position counter) is created externally and passed
    in via ``speculate()`` / ``verify()`` / ``draft_prefill()``.

    Args:
        config: PARD-specific configuration.
        dtype: Torch dtype for the draft model.
        opconfig: Operator configuration for the draft model.
    """

    supports_mixed_batch = True

    def __init__(
        self,
        config: PardSpecDecodeConfig,
        dtype: torch.dtype = torch.bfloat16,
        opconfig=None,
        cache_type=None,
    ):
        from pace.llm.configs import OperatorConfig

        self.config = config
        self._use_paged = cache_type == KVCacheType.PAGED
        self.model_path = resolve_model_path(config.model_name_or_path)
        validate_model_inputs(self.model_path, None, dtype)

        if opconfig is None:
            opconfig = OperatorConfig().finalize(cache_type=cache_type)
        elif not getattr(opconfig, "_finalized", False):
            opconfig = opconfig.finalize(cache_type=cache_type)
        self.model = init_model(self.model_path, dtype=dtype, opconfig=opconfig)

        self.config.pard_token = getattr(
            self.model.config, "pard_token", self.config.pard_token
        )
        PACE_LLM_ASSERT(
            self.config.pard_token is not None,
            "PARD token must be set either in the model config or in the "
            "speculative decode config (pard_token). Cannot initialise "
            "PardSpeculativeDecoder without it.",
        )
        self._pard_token_list = [self.config.pard_token for _ in range(32)]
        self._draft_size = self.config.num_speculative_tokens

        self._draft_cache_backend = None

        PACE_LLM_INFO(f"PARD speculative decoder loaded, config: {self.config}")

    @property
    def num_speculative_tokens(self) -> int:
        return self._draft_size

    def prepare_draft_batch(
        self,
        max_seq_length: int,
        kv_cache_type: KVCacheType,
        batch_size: int,
    ) -> None:
        draft_max = max_seq_length + self._draft_size * 2
        if kv_cache_type == KVCacheType.PAGED:
            return

        from pace.llm.attention import create_cache

        if self._draft_cache_backend is None:
            cache_kwargs = {}
            if self.config.draft_kv_cache_memory_gb is not None:
                cache_kwargs["kv_cache_memory_gb"] = (
                    self.config.draft_kv_cache_memory_gb
                )
            self._draft_cache_backend = create_cache(
                kv_cache_type,
                model_config=self.model.config,
                **cache_kwargs,
            )
        self._draft_cache_backend.prepare_batch(draft_max, batch_size)

    def create_draft_kv_cache(
        self,
        max_seq_length: int,
        kv_cache_type: KVCacheType,
    ):
        # 2x draft_size headroom: draft_size for speculated tokens plus
        # draft_size buffer for edge cases where accepted tokens exceed one round.
        draft_max = max_seq_length + self._draft_size * 2

        if kv_cache_type == KVCacheType.PAGED:
            dtype = next(self.model.parameters()).dtype
            block_size = getattr(self.model.config, "block_size", 16)
            return create_paged_kv_cache_manager(
                self.model.config, draft_max, block_size, dtype, batch_size=1
            )

        self.prepare_draft_batch(max_seq_length, kv_cache_type, 1)
        return self._draft_cache_backend.create_context(
            self.model.config,
            draft_max,
            batch_size=1,
        )

    def remove_draft_context(self, context) -> None:
        """Release a draft KV cache context (used by server cleanup)."""
        if self._draft_cache_backend is not None:
            self._draft_cache_backend.remove_context(context)

    def draft_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_kv_cache: KVCacheManager,
    ) -> None:
        """Run draft model forward on full prompt (KV warm-up, no sampling)."""
        paged_attn_metadata = None
        if self._use_paged:
            draft_dtype = next(self.model.parameters()).dtype
            block_size = getattr(self.model.config, "block_size", 16)
            paged_attn_metadata = build_paged_attention_metadata(
                draft_kv_cache,
                self.model.config,
                torch.tensor([input_ids.shape[-1]], dtype=torch.long),
                draft_dtype,
                block_size,
            )
        pa_kwargs = (
            {"paged_attn_metadata": paged_attn_metadata}
            if paged_attn_metadata is not None
            else {}
        )
        self.model(input_ids, positions, kv_cache=draft_kv_cache, **pa_kwargs)

    def prepare_draft_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Append PARD padding tokens to the last generated token."""
        unused_tokenids = torch.tensor([self._pard_token_list])[
            :, : self._draft_size - 1
        ]
        return torch.cat([model_input, unused_tokenids], dim=-1)

    def speculate(
        self,
        model_input: torch.Tensor,
        draft_kv_cache: KVCacheManager,
        draft_num_computed: torch.Tensor,
        initial_positions: Optional[torch.Tensor] = None,
    ) -> SpeculationOutput:
        kv_cache = draft_kv_cache
        num_computed = draft_num_computed

        pard_input = self.prepare_draft_input(model_input)

        spec_count = pard_input.shape[-1] - model_input.shape[-1]
        if initial_positions is not None:
            # Prefill path: use padding-aware positions for the prompt tokens so
            # _compute_pad_lens inside the attention backend correctly detects
            # leading padding and masks those positions.  Sequential positions
            # ([0, 1, ..., prompt_len-1]) would make pad_len appear to be zero,
            # causing the draft model to attend to garbage pad tokens and
            # producing wrong speculated tokens and corrupting num_computed
            # for every subsequent decode step.
            actual_lengths = initial_positions[:, -1:] + 1  # [B, 1]
            spec_pos = actual_lengths + torch.arange(spec_count, dtype=torch.long)
            pard_positions = torch.cat([initial_positions, spec_pos], dim=-1)
            num_computed_delta = actual_lengths.squeeze(1) + spec_count
        else:
            pard_positions = self._compute_positions(pard_input.shape[-1], num_computed)
            num_computed_delta = pard_input.shape[-1]

        draft_paged_meta = None
        if getattr(self, "_use_paged", False):
            if initial_positions is not None:
                # Prefill with left-padding: pack the input to strip pad
                # tokens so the paged metadata query lengths reflect real
                # token counts, matching packed native prefill.
                per_seq_lengths = num_computed_delta
                packed_tokens = []
                packed_positions = []
                for i in range(pard_input.shape[0]):
                    seq_len_i = int(per_seq_lengths[i].item())
                    packed_tokens.append(pard_input[i, -seq_len_i:])
                    packed_positions.append(
                        torch.arange(
                            seq_len_i, dtype=torch.long, device=pard_input.device
                        )
                    )
                pard_input = torch.cat(packed_tokens).unsqueeze(0)
                pard_positions = torch.cat(packed_positions).unsqueeze(0)
                draft_query_len = per_seq_lengths
                draft_past = None
            else:
                draft_query_len = torch.tensor([pard_input.shape[-1]], dtype=torch.long)
                draft_past = torch.tensor(
                    [int(num_computed[0].item())], dtype=torch.long
                )
            draft_dtype = next(self.model.parameters()).dtype
            draft_block_size = getattr(self.model.config, "block_size", 16)
            draft_paged_meta = build_paged_attention_metadata(
                kv_cache,
                self.model.config,
                draft_query_len,
                draft_dtype,
                draft_block_size,
                past_lengths=draft_past,
            )

        pa_kwargs = (
            {"paged_attn_metadata": draft_paged_meta}
            if draft_paged_meta is not None
            else {}
        )
        speculated_output: ModelOutput = self.model(
            pard_input, pard_positions, kv_cache=kv_cache, **pa_kwargs
        )

        num_computed += num_computed_delta

        speculated_tokens = speculated_output.logits[:, -self._draft_size :].argmax(
            dim=-1
        )

        extended_input = torch.cat([model_input, speculated_tokens], dim=-1)
        return SpeculationOutput(extended_input=extended_input)

    def verify(
        self,
        sampled_tokens: torch.Tensor,
        speculated_input: torch.Tensor,
        draft_kv_cache: KVCacheManager,
        draft_num_computed: torch.Tensor,
    ) -> VerificationOutput:
        kv_cache = draft_kv_cache
        num_computed = draft_num_computed

        PACE_LLM_ASSERT(
            sampled_tokens.dim() == 2,
            f"sampled_tokens must be 2D (batch, seq), got {sampled_tokens.dim()}D",
        )
        PACE_LLM_ASSERT(
            sampled_tokens.shape[1] >= self._draft_size + 1,
            f"sampled_tokens must have at least draft_size+1={self._draft_size + 1} "
            f"tokens (draft verification + bonus), got {sampled_tokens.shape[1]}",
        )

        speculated_tokens = speculated_input[:, -self._draft_size :]

        check_len = self._draft_size
        matches = sampled_tokens[0, :check_len] == speculated_tokens[0, :check_len]
        num_accepted = matches.long().cumprod(0).sum().item()

        # Accepted draft tokens + target model's token at the rejection point
        keep_token_ids = sampled_tokens[:, : num_accepted + 1]

        remove_count = self._draft_size - keep_token_ids.shape[1]
        target_kv_cache_trim = remove_count + 1

        # Clean up draft KV cache (in-place on the passed-in cache)
        kv_cache.remove_cache(self._draft_size - 1)
        num_computed -= self._draft_size - 1

        return VerificationOutput(
            accepted_tokens=keep_token_ids,
            target_kv_cache_trim=target_kv_cache_trim,
        )

    @staticmethod
    def _compute_positions(num_tokens: int, num_computed: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(num_tokens, dtype=torch.long)
        return num_computed.unsqueeze(1) + offsets.unsqueeze(0)


# Factory
def create_speculative_decoder(
    config: SpecDecodeConfig,
    dtype: torch.dtype = torch.bfloat16,
    opconfig=None,
    cache_type=None,
) -> SpeculativeDecoder:
    """Create a :class:`SpeculativeDecoder` from a config object."""
    if isinstance(config, PardSpecDecodeConfig):
        return PardSpeculativeDecoder(
            config=config, dtype=dtype, opconfig=opconfig, cache_type=cache_type
        )
    raise ValueError(
        f"Unknown speculative decoding config type: {type(config).__name__}. "
        "Expected a subclass of SpecDecodeConfig with a registered decoder."
    )
