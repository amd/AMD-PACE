# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List

import torch

from pace.llm.attention import KVCacheType, KVCacheManager
from pace.llm.attention.paged.utils import (
    create_paged_kv_cache_manager,
    build_paged_attention_metadata,
)
from pace.llm.configs import (
    SpecDecodeConfig,
    PardSpecDecodeConfig,
    SamplingConfig,
    SamplingMode,
)
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import init_model
from pace.llm.outputs import ModelOutput, SpeculativeStats
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

    **Ownership model** -- A single ``SpeculativeDecoder`` instance is
    created once (like the target model).  Per-sequence state (draft KV
    cache, position counter) lives *outside* the decoder and is passed in
    to ``speculate`` / ``verify`` / ``draft_prefill``.

    For the offline ``Generator.generate()`` flow, ``prepare()`` creates
    internal per-run state that is used as the default when the external
    state parameters are not supplied.
    """

    @abstractmethod
    def prepare(
        self,
        input_prompts: torch.Tensor,
        sampling_config: SamplingConfig,
        kv_cache_type: KVCacheType,
    ) -> None:
        """Set up *internal* per-run state for the offline generate() flow."""
        ...

    @property
    @abstractmethod
    def num_speculative_tokens(self) -> int:
        """Number of speculative tokens generated per step."""
        ...

    @abstractmethod
    def create_draft_kv_cache(
        self, max_seq_length: int, kv_cache_type: KVCacheType
    ) -> KVCacheManager:
        """Create a per-sequence draft KV cache (used by serving)."""
        ...

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
    def draft_step(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Union[KVCacheManager, List[KVCacheManager]],
    ) -> ModelOutput:
        """Batched draft-model forward pass (thin wrapper)."""
        ...

    @abstractmethod
    def prepare_draft_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Build the draft-model input from the last generated token.

        Returns a tensor ready to be fed to ``draft_step()``.  This
        encapsulates algorithm-specific input construction (e.g. PARD
        token padding) so the serving layer does not need to know the
        details.
        """
        ...

    @abstractmethod
    def speculate(
        self,
        model_input: torch.Tensor,
        draft_kv_cache: Optional[KVCacheManager] = None,
        draft_num_computed: Optional[torch.Tensor] = None,
    ) -> SpeculationOutput:
        """Generate speculated tokens for a *single* sequence.

        When ``draft_kv_cache`` / ``draft_num_computed`` are supplied they
        are used (and mutated in-place).  Otherwise the internal state
        created by ``prepare()`` is used (offline flow).
        """
        ...

    @abstractmethod
    def verify(
        self,
        sampled_tokens: torch.Tensor,
        speculated_input: torch.Tensor,
        draft_kv_cache: Optional[KVCacheManager] = None,
        draft_num_computed: Optional[torch.Tensor] = None,
    ) -> VerificationOutput:
        """Verify speculated tokens for a *single* sequence.

        Cleans up the draft KV cache (passed-in or internal).  Returns
        instructions for the caller to adjust the *target* KV cache.
        """
        ...

    @abstractmethod
    def get_stats(self) -> Optional[SpeculativeStats]:
        """Return statistics for the completed generation run."""
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

    def __init__(
        self,
        config: PardSpecDecodeConfig,
        dtype: torch.dtype = torch.bfloat16,
        opconfig=None,
        cache_type=None,
    ):
        from pace.llm.configs import OperatorConfig

        from pace.llm.generator import validate_generator_inputs

        self.config = config
        self.model_path = resolve_model_path(config.model_name_or_path)
        validate_generator_inputs(self.model_path, None, dtype)

        if opconfig is None:
            opconfig = OperatorConfig().finalize(cache_type=cache_type)
        elif not getattr(opconfig, "_finalized", False):
            opconfig = opconfig.finalize(cache_type=cache_type)
        self.model = init_model(self.model_path, dtype=dtype, opconfig=opconfig)

        # Resolve PARD token (model config overrides user config) at init
        # time so it is available before prepare() is called.
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

        # Internal per-run state (populated by prepare() for offline flow)
        self._kv_cache_manager: Optional[KVCacheManager] = None
        self._draft_cache_backend = None
        self._num_computed_tokens: Optional[torch.Tensor] = None
        self._total_accepted: list[int] = []

        PACE_LLM_INFO(f"PARD speculative decoder loaded, config: {self.config}")

    # -- helpers to resolve internal vs external state -----------------------

    def _resolve_draft_state(self, draft_kv_cache, draft_num_computed):
        """Return (kv_cache, num_computed) using external args or internal."""
        kv = draft_kv_cache if draft_kv_cache is not None else self._kv_cache_manager
        nc = (
            draft_num_computed
            if draft_num_computed is not None
            else self._num_computed_tokens
        )
        assert kv is not None, (
            "No draft KV cache available. Either pass draft_kv_cache or "
            "call prepare() first."
        )
        return kv, nc

    # -- SpeculativeDecoder interface ----------------------------------------

    def prepare(
        self,
        input_prompts: torch.Tensor,
        sampling_config: SamplingConfig,
        kv_cache_type: KVCacheType,
    ) -> None:
        """Set up internal per-run state for the offline ``generate()`` flow."""
        PACE_LLM_ASSERT(
            sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH,
            "Speculative Decoding using PARD is only supported for greedy "
            f"search sampling mode but got {sampling_config.sampling_mode}",
        )
        PACE_LLM_ASSERT(
            input_prompts.shape[0] == 1,
            "Offline speculative decoding is only supported for batch size 1",
        )

        self._use_paged = kv_cache_type == KVCacheType.PAGED
        self._total_accepted = []
        max_seq_length = sampling_config.max_new_tokens + input_prompts.size(-1)
        cache_len = (
            max_seq_length + self._draft_size if self._use_paged else max_seq_length
        )
        self._kv_cache_manager = self.create_draft_kv_cache(cache_len, kv_cache_type)
        self._num_computed_tokens = torch.zeros(
            input_prompts.shape[0], dtype=torch.long
        )

        PACE_LLM_INFO(
            f"Using PARD for speculative decoding, using config: {self.config}"
        )

    @property
    def num_speculative_tokens(self) -> int:
        return self._draft_size

    def create_draft_kv_cache(self, max_seq_length: int, kv_cache_type: KVCacheType):
        from pace.llm.attention import create_cache

        # 2x draft_size headroom: draft_size for speculated tokens plus
        # draft_size buffer for edge cases where accepted tokens exceed one round.
        draft_max = max_seq_length + self._draft_size * 2

        if kv_cache_type == KVCacheType.PAGED:
            dtype = next(self.model.parameters()).dtype
            block_size = getattr(self.model.config, "block_size", 16)
            return create_paged_kv_cache_manager(
                self.model.config, draft_max, block_size, dtype, batch_size=1
            )

        # Lazy-init: create the backend once and reuse for all contexts.
        # This ensures remove_draft_context always targets the correct backend.
        if self._draft_cache_backend is None:
            cache_kwargs = {}
            if self.config.draft_kv_cache_memory_gb is not None:
                cache_kwargs["kv_cache_memory_gb"] = (
                    self.config.draft_kv_cache_memory_gb
                )
            else:
                cache_kwargs["max_total_tokens"] = draft_max * 2
            self._draft_cache_backend = create_cache(
                kv_cache_type,
                model_config=self.model.config,
                **cache_kwargs,
            )
        return self._draft_cache_backend.create_context(self.model.config, draft_max)

    def remove_draft_context(self, context) -> None:
        """Release a draft KV cache context (used by server cleanup)."""
        if self._draft_cache_backend is not None:
            self._draft_cache_backend.remove_context(context)

    def cleanup(self) -> None:
        """Release internal draft KV cache (called from generator finally)."""
        if self._kv_cache_manager is not None and self._draft_cache_backend is not None:
            self._draft_cache_backend.remove_context(self._kv_cache_manager)
        self._kv_cache_manager = None
        self._num_computed_tokens = None

    def draft_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_kv_cache: KVCacheManager,
        paged_attn_metadata=None,
    ) -> None:
        """Run draft model forward on full prompt (KV warm-up, no sampling)."""
        pa_kwargs = (
            {"paged_attn_metadata": paged_attn_metadata}
            if paged_attn_metadata is not None
            else {}
        )
        self.model(input_ids, positions, kv_cache=draft_kv_cache, **pa_kwargs)

    def draft_step(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Union[KVCacheManager, List[KVCacheManager]],
        paged_attn_metadata=None,
    ) -> ModelOutput:
        """Batched draft-model forward (same interface as Generator.step)."""
        pa_kwargs = (
            {"paged_attn_metadata": paged_attn_metadata}
            if paged_attn_metadata is not None
            else {}
        )
        return self.model(inputs, positions, kv_cache=kv_caches, **pa_kwargs)

    def prepare_draft_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Append PARD padding tokens to the last generated token."""
        unused_tokenids = torch.tensor([self._pard_token_list])[
            :, : self._draft_size - 1
        ]
        return torch.cat([model_input, unused_tokenids], dim=-1)

    def speculate(
        self,
        model_input: torch.Tensor,
        draft_kv_cache: Optional[KVCacheManager] = None,
        draft_num_computed: Optional[torch.Tensor] = None,
    ) -> SpeculationOutput:
        kv_cache, num_computed = self._resolve_draft_state(
            draft_kv_cache, draft_num_computed
        )

        pard_input = self.prepare_draft_input(model_input)

        pard_positions = self._compute_positions(pard_input.shape[-1], num_computed)

        draft_paged_meta = None
        if getattr(self, "_use_paged", False):
            draft_query_len = torch.tensor([pard_input.shape[-1]], dtype=torch.long)
            draft_past = torch.tensor([int(num_computed[0].item())], dtype=torch.long)
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

        num_computed += pard_input.shape[-1]

        speculated_tokens = speculated_output.logits[:, -self._draft_size :].argmax(
            dim=-1
        )

        extended_input = torch.cat([model_input, speculated_tokens], dim=-1)
        return SpeculationOutput(extended_input=extended_input)

    def verify(
        self,
        sampled_tokens: torch.Tensor,
        speculated_input: torch.Tensor,
        draft_kv_cache: Optional[KVCacheManager] = None,
        draft_num_computed: Optional[torch.Tensor] = None,
    ) -> VerificationOutput:
        kv_cache, num_computed = self._resolve_draft_state(
            draft_kv_cache, draft_num_computed
        )

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

        self._total_accepted.append(keep_token_ids.shape[1])

        return VerificationOutput(
            accepted_tokens=keep_token_ids,
            target_kv_cache_trim=target_kv_cache_trim,
        )

    def get_stats(self) -> Optional[SpeculativeStats]:
        if not self._total_accepted:
            return None
        return SpeculativeStats(
            total_speculated_tokens=sum(self._total_accepted),
            mean_accepted_tokens=sum(self._total_accepted) / len(self._total_accepted),
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
