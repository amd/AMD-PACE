# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.utils import GENERATION_CONFIG_NAME
from transformers.utils import logging as hf_logging

from pace.llm.attention.specs import resolve_text_config
from pace.llm.configs import SamplingConfig
from pace.llm.generation.base import GenerationBackend
from pace.llm.generation.types import (
    GenerationProposal,
    GenerationState,
    GenerationStatus,
    TerminationReason,
)
from pace.llm.generation.validation import validate_model_inputs
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import get_tokenizer
from pace.llm.outputs import GeneratorOutput, SpeculativeStats
from pace.llm.step.budget import (
    CHANNEL_ALIGNMENT,
    GIB,
    SplitPlan,
    available_vram_bytes,
    device_workspace_bytes,
    kv_bytes_per_token,
    plan_split,
    prefill_activation_bytes,
)
from pace.llm.step.hf_modules import find_decoder_layers
from pace.llm.step.loading import activation_of, load_model, move_to_device
from pace.llm.step.split import install_split
from pace.llm.step.tree import AttentionWindows, TreeDrafter, TreeVerifier
from pace.utils.device import describe_device, require_gpu
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO, PACE_LLM_WARNING

# Sampling settings applied against each candidate's own history, with the values that switch them off.
_HISTORY_DEPENDENT = (
    ("repetition_penalty", 1.0),
    ("frequency_penalty", 0.0),
    ("min_new_tokens", 0),
)


@dataclass
class StepStats(SpeculativeStats):
    """
    Counters for one request.

    `total_speculated_tokens` and `mean_accepted_tokens` keep the meaning the native backend gives them, the tokens committed by speculation rounds with the bonus included and their mean per round, so a caller reading only `SpeculativeStats` sees comparable figures. They are filled when the request ends.

    Args:
        steps: Decode rounds, speculative or plain.
        proposed: Candidate tokens the trees offered.
        accepted: Draft tokens the full model accepted.
        committed: Tokens the rounds produced, before the shared stopping logic trims the tail. Divided by `steps` to give tokens per step.
        generated: Tokens kept, including the one the prompt forward produced.
        plain_steps: Rounds decoded without speculation, either because it was disabled or because the auto policy stopped.
        prefill_seconds: Time in the prompt forward.
        decode_seconds: Time from the first round to the end of the request.
        draft_seconds: Time drafting, when timings are collected.
        verify_seconds: Time verifying, when timings are collected.
        residual_seconds: Time in the CPU FFN, when timings are collected.
        evictions: Times channels were moved to the CPU under memory pressure.
        speculation_stopped_at: Token index where the auto policy stopped speculating, or None if it never did.
    """

    total_speculated_tokens: int = 0
    mean_accepted_tokens: float = 0.0
    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    generated: int = 0
    plain_steps: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    residual_seconds: float = 0.0
    evictions: int = 0
    speculation_stopped_at: Optional[int] = None

    @property
    def draft_ms_per_step(self) -> float:
        return self.draft_seconds / self.steps * 1e3 if self.steps else 0.0

    @property
    def verify_ms_per_step(self) -> float:
        return self.verify_seconds / self.steps * 1e3 if self.steps else 0.0

    @property
    def residual_ms_per_step(self) -> float:
        """CPU FFN time per step."""
        return self.residual_seconds / self.steps * 1e3 if self.steps else 0.0

    @property
    def tokens_per_step(self) -> float:
        """
        Tokens committed per round, against 1.0 for plain decoding.

        Divides `committed` rather than `generated`, since the latter includes the prefill token, which cost no round, and excludes any tail trimmed past the stop condition, which cost a full one.
        """
        return self.committed / self.steps if self.steps else 0.0

    @property
    def accepted_per_step(self) -> float:
        """Draft tokens accepted per round, equal to tokens per step minus the bonus."""
        return self.accepted / self.steps if self.steps else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        """Kept tokens per second of decoding, leaving out the prefill token, which was produced before `decode_seconds` starts."""
        decoded = max(self.generated - 1, 0)
        return decoded / self.decode_seconds if self.decode_seconds else 0.0

    def describe(self) -> str:
        detail = (
            f"{self.generated} tokens returned, {self.committed} in {self.steps} steps, "
            f"{self.tokens_per_step:.2f} tokens/step "
            f"({self.accepted_per_step:.2f} accepted + 1 bonus), "
            f"{self.decode_tokens_per_second:.1f} tok/s decode, "
            f"prefill {self.prefill_seconds * 1e3:.0f} ms"
        )
        if self.plain_steps:
            detail += f", {self.plain_steps} plain steps"
        if self.speculation_stopped_at is not None:
            detail += f", speculation stopped at token {self.speculation_stopped_at}"
        if self.evictions:
            detail += f", {self.evictions} evictions"
        if self.draft_seconds or self.verify_seconds:
            detail += (
                f"; draft {self.draft_ms_per_step:.0f} ms/step, "
                f"verify {self.verify_ms_per_step:.0f} ms/step "
                f"(cpu ffn {self.residual_ms_per_step:.0f} ms)"
            )
        return detail


@dataclass
class StepRequestState:
    """
    STeP's decode state for one request, held on `GenerationState.backend_state`.

    The committed tokens live on the shared state; these counters record how far each cache has consumed them. They are tracked apart because they diverge: the verify cache is never windowed, so its slot count matches the absolute position, whereas the draft cache can be trimmed at the front.

    Args:
        verify_cache: Full-model KV cache.
        draft_cache: GPU-prefix KV cache, or None once speculation stops.
        recent: Rolling tokens per round, for the auto policy.
        speculating: Whether this request is still drafting.
        verify_seen: Committed tokens in the verify cache.
        draft_cached: Committed entries in the draft cache, which windowing can reduce.
        draft_seen: Absolute position the draft has consumed to.
        stats: Counters for this request.
        decode_started: When the first round began, so decode time spans the rounds rather than being summed over them.
    """

    verify_cache: Optional[DynamicCache]
    draft_cache: Optional[DynamicCache]
    recent: deque
    speculating: bool
    verify_seen: int = 0
    draft_cached: int = 0
    draft_seen: int = 0
    stats: StepStats = field(default_factory=StepStats)
    decode_started: Optional[float] = None


class StepGenerationBackend(GenerationBackend):
    """
    STeP: speculative decoding across a GPU channel prefix and a CPU residual.

    The salient FFN channels are held on the GPU and act as the draft model; the remainder are held on the CPU. Drafting runs the GPU prefix alone. Verification adds the CPU residual, reconstructing the full model, so accepted tokens are the full model's tokens. The draft is a subnetwork of the target.

    With `speculation="off"` the same split runs as plain tensor parallelism, one token per full-model forward. This is also the fallback when speculation ceases to be beneficial.

    One request runs at a time. STeP is for a model that does not fit in VRAM, and a second request's KV cache would come out of the channel prefix that makes the draft worth running.
    """

    def __init__(
        self,
        config,
        model_name_or_path,
        tokenizer_name_or_path=None,
        dtype: torch.dtype = torch.bfloat16,
        disable_tqdm: bool = False,
    ):
        """
        Args:
            config: A `StepGenerationBackendConfig`.
            model_name_or_path: HF hub id or local directory, calibrated or not.
            tokenizer_name_or_path: Tokenizer to use instead of the checkpoint's.
            dtype: Runtime dtype.
            disable_tqdm: Hide the progress bars for loading, splitting and moving the model.
        """
        require_gpu("STeP")
        config.check_residual_backend(dtype)
        self.config = config
        self.dtype = dtype
        self.device = torch.device(f"cuda:{config.gpu_index}")
        # Events record on the current device's stream, which has to be the device holding the prefix.
        torch.cuda.set_device(self.device)
        if config.cpu_threads:
            torch.set_num_threads(config.cpu_threads)
        if disable_tqdm:
            hf_logging.disable_progress_bar()

        model_path = resolve_model_path(model_name_or_path, native_only=False)
        tokenizer_path = (
            resolve_model_path(tokenizer_name_or_path, native_only=False)
            if tokenizer_name_or_path is not None
            else None
        )
        validate_model_inputs(model_path, tokenizer_path, dtype)
        self._model_path = os.fspath(model_path)

        PACE_LLM_INFO(describe_device())
        self._tokenizer = get_tokenizer(model_path, tokenizer_path)
        self._base_sampling_config = SamplingConfig()
        generation_config = os.path.join(self._model_path, GENERATION_CONFIG_NAME)
        if os.path.exists(generation_config):
            self._base_sampling_config = SamplingConfig.from_pretrained(
                generation_config
            )

        model = load_model(self._model_path, dtype).eval()
        self.parameters = sum(p.numel() for p in model.parameters())
        self.plan = self._plan_split(model.config)
        self.split = self.plan.split
        PACE_LLM_INFO(f"channel split: {self.split.describe()}")

        PACE_LLM_INFO(f"{self.parameters / 1e9:.1f}B parameters")
        text_config = resolve_text_config(model.config)
        PACE_LLM_ASSERT(
            config.tree_width <= text_config.vocab_size,
            f"tree_width {config.tree_width} exceeds the vocabulary of {text_config.vocab_size} tokens",
        )
        attention_windows = AttentionWindows.of(model.config)
        self.split_state, self.blocks = install_split(
            model,
            self.split.gpu_channels,
            self.device,
            activation_of(text_config),
            residual_backend=config.residual_backend,
            disable_tqdm=disable_tqdm,
        )
        itemsize = torch.empty((), dtype=dtype).element_size()
        # Each channel is a gate_proj row, an up_proj row and a down_proj column in every block.
        self._channel_bytes = 3 * text_config.hidden_size * itemsize * len(self.blocks)
        self._kv_bytes_per_token = kv_bytes_per_token(model.config, dtype)
        _, layers = find_decoder_layers(model)
        for layer in tqdm(
            layers if layers is not None else [],
            desc=f"Moving layers to {self.device}",
            unit="layer",
            disable=disable_tqdm,
        ):
            move_to_device(layer, self.device)
        self._model = move_to_device(model, self.device)
        self.drafter = TreeDrafter(
            self._model,
            config.tree_width,
            config.tree_depth,
            config.tree_mode,
            window=config.draft_kv_window,
            sink=config.draft_kv_sink,
            attention_windows=attention_windows,
        )
        self.verifier = TreeVerifier(self._model, attention_windows)
        # Draft quality is a property of the model and the ratio, not of a prompt, so the auto verdict persists across requests until reset.
        self.speculation_abandoned = False
        self._active: Optional[str] = None
        self._warnings_given: Set[str] = set()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_config(self):
        return self._model.config

    @property
    def supports_packing(self) -> bool:
        return False

    def reset(self) -> None:
        """
        Return evicted channels to the GPU and let `speculation="auto"` reconsider.

        Only valid between requests, since restoring channels changes the split under a running one.
        """
        PACE_LLM_ASSERT(
            self._active is None,
            f"cannot reset STeP while request {self._active} is running",
        )
        restored = [
            block.restore(self.split.gpu_channels - block.gpu_channels)
            for block in self.blocks
        ]
        if restored[0]:
            PACE_LLM_INFO(
                f"restored {restored[0]} channels per block to the GPU; {self.blocks[0].gpu_channels} resident, residual now {self.blocks[0].residual.describe()}"
            )
        self.speculation_abandoned = False

    def _plan_split(self, model_config) -> SplitPlan:
        budget = available_vram_bytes(self.config.gpu_index)
        if self.config.vram_budget_gb is not None:
            requested = int(self.config.vram_budget_gb * GIB)
            if requested > budget:
                PACE_LLM_WARNING(
                    f"vram_budget_gb={self.config.vram_budget_gb} is more than the {budget / GIB:.1f} GiB this process can use on {self.device} with 1 GiB kept free, so the split is planned against {budget / GIB:.1f} GiB"
                )
            budget = min(budget, requested)
        plan = plan_split(
            model_config,
            budget,
            self.config.max_seq_length,
            self.dtype,
            self.config.speculation,
            draft_ratio=self.config.draft_ratio,
            align=self.config.uses_pace_kernels(self.dtype),
            parameter_count=self.parameters,
            workspace=device_workspace_bytes(self.device, self.dtype),
        )
        PACE_LLM_INFO(
            f"planned against a {budget / GIB:.1f} GiB budget: {plan.memory.describe(plan.split.ratio)}"
        )
        return plan

    def _prepare_sampling_config(
        self,
        user_config: Optional[SamplingConfig],
        prompt_length: int,
    ) -> SamplingConfig:
        config = copy.deepcopy(self._base_sampling_config)
        config.merge_from(user_config, self.tokenizer)
        explicit = getattr(user_config, "_explicit_keys", frozenset())
        for name, value, off in self._drop_inherited_settings(config, explicit):
            if name not in self._warnings_given:
                self._warnings_given.add(name)
                PACE_LLM_WARNING(
                    f"the checkpoint's generation_config.json sets {name}={value}, which STeP cannot apply exactly, since it samples every draft-tree node against the committed history; generating with {name}={off}"
                )
        config.verify_max_new_tokens(
            prompt_length,
            resolve_text_config(self.model_config).max_position_embeddings,
        )
        config.finalize()
        return config

    @staticmethod
    def _drop_inherited_settings(
        sampling_config: SamplingConfig, explicit: FrozenSet[str]
    ) -> List[Tuple[str, object, object]]:
        """
        Switch off the history-dependent settings a request inherited from the checkpoint rather than set itself.

        Args:
            sampling_config: The request's settings merged over the checkpoint's generation_config.json.
            explicit: The settings the request passed.

        Returns:
            list: (name, inherited value, value used) for each setting switched off.
        """
        dropped = []
        for name, off in _HISTORY_DEPENDENT:
            value = getattr(sampling_config, name)
            if value != off and name not in explicit:
                setattr(sampling_config, name, off)
                dropped.append((name, value, off))
        return dropped

    @staticmethod
    def _check_sampling(sampling_config: SamplingConfig) -> None:
        """
        Refuse sampling settings STeP cannot reproduce exactly.

        Every node of the draft tree is sampled in one call against the committed history, so settings that depend on each candidate's own history would be applied against the wrong one. Those a request inherited from the checkpoint are already switched off, so any left were asked for.
        """
        history_dependent = [
            (name, getattr(sampling_config, name), off)
            for name, off in _HISTORY_DEPENDENT
            if getattr(sampling_config, name) != off
        ]
        refused = ", ".join(f"{name}={value}" for name, value, _ in history_dependent)
        wanted = ", ".join(f"{name}={off}" for name, _, off in history_dependent)
        PACE_LLM_ASSERT(
            not history_dependent,
            f"STeP samples every draft-tree node against the committed history, so it cannot apply {refused} exactly; leave unset or set {wanted}",
        )
        PACE_LLM_ASSERT(
            not sampling_config.return_input_logprobs,
            "STeP keeps only the last prompt position's logits, so return_input_logprobs is not supported",
        )

    @staticmethod
    def _single(states: Sequence[GenerationState]) -> GenerationState:
        """The one request in a batch, since STeP admits no more."""
        PACE_LLM_ASSERT(
            len(states) == 1,
            f"STeP generates one sequence at a time, got a batch of {len(states)}",
        )
        return states[0]

    def _prepare_batch(self, states: Sequence[GenerationState]) -> None:
        self._single(states)

    def _allocate(self, state: GenerationState) -> None:
        PACE_LLM_ASSERT(
            self._active is None,
            f"STeP generates one sequence at a time, and request {self._active} is still running",
        )
        self._check_sampling(state.sampling_config)
        if state.attention_mask is not None:
            PACE_LLM_ASSERT(
                bool(state.attention_mask.bool().all()),
                "STeP takes unpadded prompts, but the attention mask has padding",
            )
        if self.config.reset_between_requests:
            self.reset()

        speculating = (
            self.config.speculation != "off" and not self.speculation_abandoned
        )
        state.backend_state = StepRequestState(
            verify_cache=DynamicCache(),
            draft_cache=DynamicCache() if speculating else None,
            recent=deque(maxlen=self.config.speculation_window),
            speculating=speculating,
        )
        self._active = state.request_id

    def _prefill(
        self,
        states: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        state = self._single(states)
        chunk = (chunk_info or {}).get(state.request_id)
        if chunk is not None:
            start, size = chunk
            PACE_LLM_ASSERT(
                start == 0 and size >= state.prompt_length,
                "STeP prefills a prompt in one pass; serve it with the iterative scheduler, which does not chunk",
            )
        return {state.request_id: self._prefill_one(state)}

    @torch.no_grad()
    def _prefill_one(self, state: GenerationState) -> GenerationProposal:
        """Run the prompt forward through the full model and, when speculating, the draft."""
        request = state.backend_state
        stats = request.stats
        if (
            state.prompt_length > self.config.max_seq_length
            and "max_seq_length" not in self._warnings_given
        ):
            self._warnings_given.add("max_seq_length")
            PACE_LLM_WARNING(
                f"a {state.prompt_length}-token prompt is longer than the {self.config.max_seq_length} tokens the split was planned for, so channels may move to the CPU to make room. Raise max_seq_length to plan for prompts this long."
            )
        # max_seq_length does not cap the prompt, and its caches are built in one pass, so check for room first.
        caches = 2 if request.speculating else 1
        self._maybe_evict(
            state,
            state.prompt_length,
            cache_bytes=state.prompt_length * caches * self._kv_bytes_per_token,
            transient_bytes=prefill_activation_bytes(
                self._model.config,
                self.dtype,
                state.prompt_length,
                self.blocks[0].gpu_channels,
            ),
        )
        self.split_state.collect_timings = self.config.collect_timings
        self.split_state.reset_timings()

        input_ids = state.input_ids.to(self.device)
        start = time.perf_counter()
        self.split_state.verify = True
        # Only the last position is sampled, and full-prompt logits would cost vocab-sized rows per prompt token.
        outputs = self._model(
            input_ids=input_ids,
            past_key_values=request.verify_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        sampled = state.sampler.sample(
            state.input_ids, outputs.logits[:, -1, :].float()
        )

        if request.speculating:
            self.split_state.verify = False
            self._model(
                input_ids=input_ids,
                past_key_values=request.draft_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        stats.prefill_seconds = time.perf_counter() - start
        # Prefill runs the residual over the whole prompt, so including it would distort the per-step figures.
        self.split_state.reset_timings()

        request.verify_seen = state.prompt_length
        request.draft_cached = state.prompt_length
        request.draft_seen = state.prompt_length
        return self._proposal(
            state,
            sampled.next_tokens.reshape(-1).tolist(),
            GenerationStatus.PREFILL_COMPLETED,
            probs=sampled.probs,
            logprobs=sampled.logprobs,
        )

    def _decode(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        state = self._single(states)
        return {state.request_id: self._decode_one(state)}

    @torch.no_grad()
    def _decode_one(self, state: GenerationState) -> GenerationProposal:
        """
        Advance a request by one round.

        A speculation round proposes the accepted path plus one bonus token; a plain round proposes one. The shared commit trims the proposal against the stop condition and the remaining budget, while the statistics keep counting the untrimmed round, so tokens per step continues to measure what the round produced rather than what survived.
        """
        request = state.backend_state
        stats = request.stats
        if request.decode_started is None:
            request.decode_started = time.perf_counter()
        committed_ids = state.input_ids[0]

        if request.speculating:
            self.split_state.verify = False
            draft_pending = committed_ids[request.draft_seen :].tolist()
            mark = self._mark()
            tree, request.draft_cached = self.drafter.grow(
                draft_pending,
                request.draft_cache,
                request.draft_cached,
                request.draft_seen,
            )
            stats.draft_seconds += self._since(mark)
            request.draft_seen += len(draft_pending)

            self.split_state.verify = True
            mark = self._mark()
            result = self.verifier.verify(
                tree,
                request.verify_cache,
                request.verify_seen,
                committed_ids[request.verify_seen :].tolist(),
                state.sampler,
                state.input_ids,
            )
            stats.verify_seconds += self._since(mark)
            request.verify_seen = result.context_length
            committed = result.committed
            probs = result.probs.unsqueeze(0) if result.probs is not None else None
            logprobs = (
                result.logprobs.unsqueeze(0) if result.logprobs is not None else None
            )

            stats.steps += 1
            stats.proposed += result.proposed
            stats.accepted += result.accepted
            stats.committed += len(committed)
            request.recent.append(len(committed))
            request.speculating = self._still_worth_it(
                request.recent, stats, state.generated_count + len(committed)
            )
            if not request.speculating:
                request.draft_cache = None
                torch.cuda.empty_cache()
        else:
            pending = committed_ids[request.verify_seen :].tolist()
            sampled = self._plain_step(pending, state)
            request.verify_seen = state.token_count
            committed = sampled.next_tokens.reshape(-1).tolist()
            probs, logprobs = sampled.probs, sampled.logprobs
            stats.steps += 1
            stats.plain_steps += 1
            stats.committed += 1

        stats.decode_seconds = time.perf_counter() - request.decode_started
        stats.residual_seconds = self.split_state.residual_seconds
        self._maybe_evict(state, request.verify_seen)
        return self._proposal(
            state,
            committed,
            GenerationStatus.DECODING_IN_PROGRESS,
            probs=probs,
            logprobs=logprobs,
        )

    def _proposal(
        self,
        state: GenerationState,
        tokens,
        status: GenerationStatus,
        probs: Optional[torch.Tensor],
        logprobs: Optional[torch.Tensor],
    ) -> GenerationProposal:
        return GenerationProposal(
            token_ids=torch.tensor([tokens], dtype=torch.long),
            status=status,
            computed_prompt_tokens=state.backend_state.verify_seen,
            probs=probs if state.sampling_config.return_probs else None,
            logprobs=logprobs if state.sampling_config.return_logprobs else None,
        )

    def _mixed(
        self,
        prefills: Sequence[GenerationState],
        decodes: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        raise NotImplementedError(
            "STeP runs one request at a time and does not mix prefill with decode; serve it with the iterative scheduler, the server default"
        )

    def _finalize_proposal(
        self,
        state: GenerationState,
        proposal: GenerationProposal,
        keep_count: int,
        stop_reason: Optional[str] = None,
    ) -> None:
        request = state.backend_state
        if request is None:
            return
        # The verify cache must hold every committed token but the last, which the next round feeds in to predict the position after it. A trimmed commit can leave accepted tokens past that point.
        last = state.token_count + keep_count - 1
        if request.verify_cache is not None and request.verify_seen > last:
            request.verify_seen = last
            request.verify_cache.crop(last)
            proposal.computed_prompt_tokens = last
        request.stats.generated += keep_count

    def _release(self, state: GenerationState, reason: TerminationReason) -> None:
        request = state.backend_state
        if request is None:
            return
        # Rounds are timed without synchronizing, so the running total trails whatever the device has not finished.
        torch.cuda.synchronize()
        stats = request.stats
        if request.decode_started is not None:
            stats.decode_seconds = time.perf_counter() - request.decode_started
        stats.residual_seconds = self.split_state.residual_seconds
        rounds = stats.steps - stats.plain_steps
        stats.total_speculated_tokens = stats.committed - stats.plain_steps
        stats.mean_accepted_tokens = (
            stats.total_speculated_tokens / rounds if rounds else 0.0
        )
        request.verify_cache = None
        request.draft_cache = None
        if self._active == state.request_id:
            self._active = None

    def _build_output(self, states: Sequence[GenerationState]) -> GeneratorOutput:
        output = super()._build_output(states)
        output.speculative_stats = states[0].backend_state.stats
        return output

    def _maybe_evict(
        self,
        state: GenerationState,
        tokens: int,
        cache_bytes: int = 0,
        transient_bytes: int = 0,
    ) -> None:
        """
        Move channels to the CPU when the next step would take VRAM use past the budget.

        An eviction makes room for the longer of `max_seq_length` and twice the current length, the doubling capped at the request's own length limit. Each eviction copies the channels that stay on the GPU, so doubling holds a long request to a few evictions without taking room for tokens it may never generate. A split derived from the budget already holds `max_seq_length` tokens, so this happens once a sequence grows past that, or when an explicit ratio starts above the budget.

        Only FFN weights move. The KV cache requires no rebuilding, since its shape depends on layers, heads and head dimension, none of which change.

        Args:
            state: The running request, whose statistics count the eviction.
            tokens: Tokens in the KV cache once the step is taken.
            cache_bytes: KV cache the step builds, counted as already in use.
            transient_bytes: Activations the step needs while it runs.
        """
        budget = self.plan.budget_bytes
        allocated = torch.cuda.memory_allocated(self.config.gpu_index)
        used = allocated + cache_bytes + transient_bytes
        if used <= budget:
            return

        request = state.backend_state
        longest = max(
            self.config.max_seq_length,
            min(state.prompt_length + state.sampling_config.max_new_tokens, 2 * tokens),
        )
        caches = 2 if request.speculating else 1
        growth = max(0, longest - tokens) * caches * self._kv_bytes_per_token
        excess = allocated + cache_bytes - budget
        # Whole aligned blocks, since `evict` rounds a partial move down.
        block_bytes = self._channel_bytes * CHANNEL_ALIGNMENT
        # During a prefill each block moved also takes its share of the activations with it.
        block_activations = (
            prefill_activation_bytes(
                self._model.config, self.dtype, tokens, CHANNEL_ALIGNMENT
            )
            - prefill_activation_bytes(self._model.config, self.dtype, tokens, 0)
            if transient_bytes
            else 0
        )
        blocks = max(
            -(-(excess + transient_bytes) // (block_bytes + block_activations)),
            -(-(excess + growth) // block_bytes),
        )
        moved = [block.evict(blocks * CHANNEL_ALIGNMENT) for block in self.blocks]
        where = f"at {used / GIB:.1f} GiB of a {budget / GIB:.1f} GiB budget"
        if cache_bytes:
            where += (
                f", counting {cache_bytes / GIB:.2f} GiB of prompt cache not yet built"
            )
        if moved[0] == 0:
            if "eviction" not in self._warnings_given:
                self._warnings_given.add("eviction")
                PACE_LLM_WARNING(
                    f"no FFN channels are left to move {where}, so VRAM use will go past the budget. Lower max_new_tokens or max_seq_length, or raise vram_budget_gb."
                )
            return
        request.stats.evictions += 1
        torch.cuda.empty_cache()
        PACE_LLM_INFO(
            f"evicted {moved[0]} channels per block {where}, leaving room for {longest} tokens; {self.blocks[0].gpu_channels} remain on GPU, residual now {self.blocks[0].residual.describe()}"
        )

    def _mark(self) -> Optional[float]:
        """
        Start a timing interval, or None when timings are off.

        GPU work is queued rather than executed, so a correct interval requires a synchronize. Timings are therefore opt-in.
        """
        if not self.config.collect_timings:
            return None
        torch.cuda.synchronize()
        return time.perf_counter()

    def _since(self, mark: Optional[float]) -> float:
        if mark is None:
            return 0.0
        torch.cuda.synchronize()
        return time.perf_counter() - mark

    def _plain_step(self, pending: Sequence[int], state: GenerationState):
        """
        Decode one token with the full model and no speculation.

        This is plain tensor parallelism: the GPU prefix and CPU residual still partition the FFN, but nothing is drafted or verified.
        """
        self.split_state.verify = True
        tokens = torch.tensor([list(pending)], device=self.device)
        outputs = self._model(
            input_ids=tokens,
            past_key_values=state.backend_state.verify_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return state.sampler.sample(state.input_ids, outputs.logits[:, -1, :].float())

    def _still_worth_it(self, recent, stats: StepStats, produced: int) -> bool:
        """
        Decide whether speculation is still earning its cost.

        Every round commits at least the bonus token, so tokens per step cannot fall below 1. A value near 1 means the wider verify forward and the draft forwards yielded nothing.

        The test is a threshold on a rolling mean. The window is short because tokens per step is stable within a run, and every round spent deciding runs at the losing rate.
        """
        if self.config.speculation != "auto":
            return True
        if len(recent) < recent.maxlen:
            return True
        if sum(recent) / len(recent) >= self.config.min_tokens_per_step:
            return True
        stats.speculation_stopped_at = produced
        self.speculation_abandoned = True
        PACE_LLM_INFO(
            f"speculation yielded {sum(recent) / len(recent):.2f} tokens/step over the last {len(recent)} rounds, below the {self.config.min_tokens_per_step:.2f} threshold; falling back to plain decoding for this and subsequent requests. Call reset() or set reset_between_requests to reconsider."
        )
        return False
