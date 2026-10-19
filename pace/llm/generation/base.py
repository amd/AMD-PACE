# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import BatchEncoding, TextStreamer

from pace.llm.configs import SamplingConfig
from pace.llm.outputs import GeneratorOutput
from pace.llm.sampler import Sampler
from pace.llm.stopping_criteria import StoppingCriteria
from pace.utils.logging import PACE_LLM_ERROR
from pace.llm.generation.types import (
    GenerationBatchResult,
    GenerationDelta,
    GenerationProposal,
    GenerationRequest,
    GenerationState,
    GenerationStatus,
    TerminationReason,
)


class GenerationBackend(ABC):
    """Shared request lifecycle for generation implementations.

    An exception escaping a backend batch hook fails every request in that
    batch. Backends may isolate request-local failures by returning an ERROR
    proposal for only that request.
    """

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def model_config(self):
        raise NotImplementedError

    @property
    def supports_packing(self) -> bool:
        return True

    @abstractmethod
    def _prepare_sampling_config(
        self,
        user_config: Optional[SamplingConfig],
        prompt_length: int,
    ) -> SamplingConfig:
        raise NotImplementedError

    @abstractmethod
    def _allocate(self, state: GenerationState) -> None:
        raise NotImplementedError

    def _prepare_batch(self, states: Sequence[GenerationState]) -> None:
        pass

    @abstractmethod
    def _prefill(
        self,
        states: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        raise NotImplementedError

    def _decode_fixed_batch(
        self, states: Sequence[GenerationState]
    ) -> Dict[str, GenerationProposal]:
        raise NotImplementedError

    @abstractmethod
    def _mixed(
        self,
        prefills: Sequence[GenerationState],
        decodes: Sequence[GenerationState],
        chunk_info: Optional[Dict[str, Tuple[int, int]]],
    ) -> Dict[str, GenerationProposal]:
        raise NotImplementedError

    @abstractmethod
    def _finalize_proposal(
        self,
        state: GenerationState,
        proposal: GenerationProposal,
        keep_count: int,
        stop_reason: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _release(self, state: GenerationState, reason: TerminationReason) -> None:
        raise NotImplementedError

    def create_request(self, request: GenerationRequest) -> GenerationState:
        return self.create_requests([request])[0]

    def create_requests(
        self, requests: Sequence[GenerationRequest]
    ) -> List[GenerationState]:
        sampling_configs = {}
        states = [self._create_state(request, sampling_configs) for request in requests]
        batch_size = len(states)
        for state in states:
            state.allocation_batch_size = batch_size
        try:
            self._prepare_batch(states)
            for state in states:
                self._allocate(state)
        except Exception:
            self.release(states, TerminationReason.ERROR)
            raise
        return states

    def _create_state(
        self,
        request: GenerationRequest,
        sampling_configs: Optional[dict] = None,
    ) -> GenerationState:
        input_ids = request.input_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"each generation request must contain one prompt, got {tuple(input_ids.shape)}"
            )

        config_key = (id(request.sampling_config), input_ids.shape[-1])
        sampling_config = (
            sampling_configs.get(config_key) if sampling_configs is not None else None
        )
        if sampling_config is None:
            sampling_config = self._prepare_sampling_config(
                request.sampling_config, input_ids.shape[-1]
            )
            if sampling_configs is not None:
                sampling_configs[config_key] = sampling_config
        max_length = input_ids.shape[-1] + sampling_config.max_new_tokens
        token_buffer = torch.empty(
            (1, max_length),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        token_buffer[:, : input_ids.shape[-1]] = input_ids
        state = GenerationState(
            request_id=request.request_id,
            prompt_token_ids=input_ids,
            sampling_config=sampling_config,
            sampler=Sampler(sampling_config, input_ids),
            stopping_criteria=StoppingCriteria(
                sampling_config, input_ids, self.tokenizer
            ),
            eos_token_ids=set(sampling_config.eos_token_id.tolist()),
            token_buffer=token_buffer,
            token_count=input_ids.shape[-1],
            prompt_length=input_ids.shape[-1],
            logprobs_n=request.logprobs_n,
            attention_mask=request.attention_mask,
            infer_padding=request.infer_padding,
            accumulate_output_tensors=request.accumulate_output_tensors,
        )
        return state

    @torch.inference_mode()
    def prefill(
        self,
        states: Sequence[GenerationState],
        *,
        chunk_info: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> GenerationBatchResult:
        invalid = self._reject_duplicate_requests(states)
        if invalid is not None:
            return invalid
        try:
            proposals = self._prefill(states, chunk_info)
        except Exception as error:
            return self._fail_batch(states, error)
        return self._commit_batch(states, proposals)

    @torch.inference_mode()
    def decode(self, states: Sequence[GenerationState]) -> GenerationBatchResult:
        invalid = self._reject_duplicate_requests(states)
        if invalid is not None:
            return invalid
        try:
            proposals = self._decode(states)
        except Exception as error:
            return self._fail_batch(states, error)
        return self._commit_batch(states, proposals)

    @torch.inference_mode()
    def decode_fixed_batch(
        self, states: Sequence[GenerationState]
    ) -> GenerationBatchResult:
        active = [state for state in states if not state.finished]
        invalid = self._reject_duplicate_requests(active)
        if invalid is not None:
            return invalid
        try:
            proposals = self._decode_fixed_batch(states)
        except Exception as error:
            return self._fail_batch(active, error)
        return self._commit_batch(active, proposals)

    @torch.inference_mode()
    def mixed(
        self,
        prefills: Sequence[GenerationState],
        decodes: Sequence[GenerationState],
        *,
        chunk_info: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> GenerationBatchResult:
        states = [*prefills, *decodes]
        invalid = self._reject_duplicate_requests(states)
        if invalid is not None:
            return invalid
        try:
            proposals = self._mixed(prefills, decodes, chunk_info)
        except Exception as error:
            return self._fail_batch(states, error)
        return self._commit_batch(states, proposals)

    def _fail_batch(
        self, states: Sequence[GenerationState], error: Exception
    ) -> GenerationBatchResult:
        PACE_LLM_ERROR(f"generation batch failed: {error}")
        deltas = {}
        for state in states:
            state.status = GenerationStatus.ERROR
            state.finish_reason = "error"
            self.release(state, TerminationReason.ERROR)
            deltas[state.request_id] = GenerationDelta(
                request_id=state.request_id,
                token_ids=[],
                status=GenerationStatus.ERROR,
                finish_reason="error",
                error=str(error),
            )
        return GenerationBatchResult(deltas=deltas)

    def _reject_duplicate_requests(
        self, states: Sequence[GenerationState]
    ) -> Optional[GenerationBatchResult]:
        request_ids = [state.request_id for state in states]
        if len(request_ids) == len(set(request_ids)):
            return None
        return self._fail_batch(
            states,
            ValueError("duplicate request IDs in generation batch"),
        )

    def release(
        self,
        states: Union[GenerationState, Iterable[GenerationState]],
        reason: TerminationReason,
    ) -> None:
        if isinstance(states, GenerationState):
            states = [states]
        for state in states:
            if state.resources_released:
                continue
            self._release(state, reason)
            state.resources_released = True

    def _commit_batch(
        self,
        states: Sequence[GenerationState],
        proposals: Dict[str, GenerationProposal],
    ) -> GenerationBatchResult:
        deltas = {}
        by_id = {state.request_id: state for state in states}
        state_ids = set(by_id)
        proposal_ids = set(proposals)
        if state_ids != proposal_ids:
            missing = sorted(state_ids - proposal_ids)
            unknown = sorted(proposal_ids - state_ids)
            return self._fail_batch(
                states,
                RuntimeError(
                    "backend proposal IDs do not match request IDs: "
                    f"missing={missing}, unknown={unknown}"
                ),
            )

        materialized_tokens = None
        if all(
            proposal.token_ids.numel() == 1
            and proposal.status
            not in {
                GenerationStatus.PREFILL_IN_PROGRESS,
                GenerationStatus.CANCELLED,
                GenerationStatus.ERROR,
            }
            and not state.sampling_config.stop_strings
            for state, proposal in (
                (by_id[request_id], proposals[request_id])
                for request_id in proposal_ids
            )
        ):
            materialized_tokens = self._materialize_single_token_batch(
                states, proposals
            )

        for request_id, proposal in proposals.items():
            state = by_id[request_id]
            try:
                deltas[request_id] = self._commit_proposal(
                    state,
                    proposal,
                    proposed=(
                        materialized_tokens.get(request_id)
                        if materialized_tokens is not None
                        else None
                    ),
                )
            except Exception as error:
                PACE_LLM_ERROR(
                    f"request {request_id} generation commit failed: {error}"
                )
                state.status = GenerationStatus.ERROR
                state.finish_reason = "error"
                self.release(state, TerminationReason.ERROR)
                deltas[request_id] = GenerationDelta(
                    request_id=request_id,
                    token_ids=[],
                    status=GenerationStatus.ERROR,
                    finish_reason="error",
                    error=str(error),
                )
        return GenerationBatchResult(deltas=deltas)

    @staticmethod
    def _materialize_single_token_batch(
        states: Sequence[GenerationState],
        proposals: Dict[str, GenerationProposal],
    ) -> Dict[str, List[int]]:
        """Copy a one-token proposal batch to CPU with one synchronization."""
        token_values = (
            torch.cat(
                [proposals[state.request_id].token_ids.reshape(-1) for state in states]
            )
            .cpu()
            .tolist()
        )
        return {state.request_id: [token] for state, token in zip(states, token_values)}

    def _commit_proposal(
        self,
        state: GenerationState,
        proposal: GenerationProposal,
        proposed: Optional[List[int]] = None,
    ) -> GenerationDelta:
        if proposal.status in {
            GenerationStatus.CANCELLED,
            GenerationStatus.ERROR,
        }:
            self._finalize_proposal(state, proposal, 0)
            state.status = proposal.status
            state.finish_reason = proposal.finish_reason or proposal.status.name.lower()
            reason = (
                TerminationReason.CANCELLED
                if proposal.status == GenerationStatus.CANCELLED
                else TerminationReason.ERROR
            )
            self.release(state, reason)
            return GenerationDelta(
                request_id=state.request_id,
                token_ids=[],
                status=state.status,
                finish_reason=state.finish_reason,
                error=proposal.error,
            )

        if proposal.status == GenerationStatus.PREFILL_IN_PROGRESS:
            self._finalize_proposal(state, proposal, 0)
            state.num_computed_tokens = proposal.computed_prompt_tokens
            return GenerationDelta(
                request_id=state.request_id,
                token_ids=[],
                status=GenerationStatus.PREFILL_IN_PROGRESS,
                computed_prompt_tokens=proposal.computed_prompt_tokens,
                prompt_logprobs=proposal.prompt_logprobs,
            )

        if proposed is None:
            proposed = proposal.token_ids.reshape(-1).tolist()
        if (
            not proposed
            and state.remaining_tokens > 0
            and proposal.status != GenerationStatus.COMPLETED
        ):
            raise RuntimeError(
                f"backend proposed no tokens for active request {state.request_id}"
            )
        keep_count = min(len(proposed), state.remaining_tokens)
        candidates = proposal.token_ids.reshape(-1)[:keep_count].to(
            dtype=state.token_buffer.dtype,
            device=state.token_buffer.device,
        )
        candidate_end = state.token_count + keep_count
        state.token_buffer[0, state.token_count : candidate_end] = candidates
        stop_reason = None
        for count in range(1, keep_count + 1):
            preview = state.token_buffer[:, : state.token_count + count]
            if state.stopping_criteria.stop_now(preview, num_new_tokens=1).item():
                keep_count = count
                stop_reason = state.stopping_criteria.stop_reason
                break

        self._finalize_proposal(
            state,
            proposal,
            keep_count,
            stop_reason=stop_reason,
        )
        kept_tokens = proposed[:keep_count]
        state.append_tokens(kept_tokens)
        state.num_computed_tokens = max(
            state.num_computed_tokens, proposal.computed_prompt_tokens
        )
        kept_probs = self._trim_distributions(proposal.probs, keep_count)
        kept_logprobs = self._trim_distributions(proposal.logprobs, keep_count)
        if state.accumulate_output_tensors:
            if proposal.prompt_logprobs is not None:
                state.input_logprobs = proposal.prompt_logprobs
            if kept_probs is not None:
                state.accumulated_probs.append(kept_probs)
            if kept_logprobs is not None:
                state.accumulated_logprobs.append(kept_logprobs)

        finished = (
            proposal.status == GenerationStatus.COMPLETED
            or stop_reason is not None
            or state.remaining_tokens == 0
        )
        if finished:
            state.status = GenerationStatus.COMPLETED
            state.finish_reason = proposal.finish_reason or stop_reason or "length"
            if not state.defer_release:
                self.release(state, TerminationReason.COMPLETED)
        else:
            state.status = proposal.status

        return GenerationDelta(
            request_id=state.request_id,
            token_ids=kept_tokens,
            status=state.status,
            computed_prompt_tokens=proposal.computed_prompt_tokens,
            finish_reason=state.finish_reason,
            probs=kept_probs,
            logprobs=kept_logprobs,
            prompt_logprobs=proposal.prompt_logprobs,
        )

    # TODO: Add an OfflineEngine scheduler for continuous mixed batching.
    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[torch.Tensor, BatchEncoding],
        sampling_config: Optional[SamplingConfig] = None,
        *,
        text_streamer=None,
    ) -> GeneratorOutput:
        input_ids = prompts.input_ids if isinstance(prompts, BatchEncoding) else prompts
        attention_mask = (
            getattr(prompts, "attention_mask", None)
            if isinstance(prompts, BatchEncoding)
            else None
        )
        if input_ids.dim() != 2:
            raise ValueError(f"prompts must be 2D, got {input_ids.dim()}D")
        if isinstance(text_streamer, TextStreamer) and input_ids.shape[0] != 1:
            raise ValueError("text streaming only supports batch size 1")

        states: List[GenerationState] = []
        streamer_ended = False
        try:
            requests = []
            for index in range(input_ids.shape[0]):
                row_mask = (
                    attention_mask[index : index + 1]
                    if attention_mask is not None
                    else None
                )
                requests.append(
                    GenerationRequest(
                        request_id=str(uuid.uuid4()),
                        input_ids=input_ids[index : index + 1],
                        sampling_config=sampling_config,
                        attention_mask=row_mask,
                        accumulate_output_tensors=True,
                    )
                )
            states = self.create_requests(requests)
            fixed_batch = len(states) > 1 and not self.supports_packing
            if fixed_batch:
                for state in states:
                    state.defer_release = True
            if len(states) > 1:
                for state in states[1:]:
                    state.sampler.share_rng_with(states[0].sampler)

            if text_streamer is not None:
                text_streamer.put(input_ids)

            result = self.prefill(states)
            self._raise_on_errors(result)
            self._stream_result(result, text_streamer, states)
            active = [state for state in states if not state.finished]
            while active:
                result = (
                    self.decode_fixed_batch(states)
                    if fixed_batch
                    else self.decode(active)
                )
                self._raise_on_errors(result)
                self._stream_result(result, text_streamer, active)
                active = [state for state in active if not state.finished]

            if text_streamer is not None:
                streamer_ended = True
                text_streamer.end()
            return self._build_output(states)
        finally:
            try:
                if text_streamer is not None and not streamer_ended:
                    streamer_ended = True
                    text_streamer.end()
            finally:
                self.release(states, TerminationReason.CANCELLED)

    @staticmethod
    def _raise_on_errors(result: GenerationBatchResult) -> None:
        for delta in result.deltas.values():
            if delta.status == GenerationStatus.ERROR:
                raise RuntimeError(delta.error)

    @staticmethod
    def _stream_result(
        result: GenerationBatchResult,
        text_streamer,
        states: Sequence[GenerationState],
    ) -> None:
        if text_streamer is None:
            return
        rows = [result.deltas[state.request_id].token_ids for state in states]
        if not any(rows):
            return
        width = max(len(row) for row in rows)
        tokens = torch.full(
            (len(rows), width),
            int(states[0].sampling_config.pad_token_id),
            dtype=torch.long,
        )
        for index, row in enumerate(rows):
            if row:
                tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        text_streamer.put(tokens)

    def _build_output(self, states: Sequence[GenerationState]) -> GeneratorOutput:
        pad_id = int(states[0].sampling_config.pad_token_id)
        max_length = max(state.token_count for state in states)
        output = torch.full(
            (len(states), max_length),
            pad_id,
            dtype=states[0].token_buffer.dtype,
            device=states[0].token_buffer.device,
        )
        for index, state in enumerate(states):
            output[index, : state.token_count] = state.input_ids[0]

        decoded_text = None
        if states[0].sampling_config.return_text:
            decoded_text = [
                self.tokenizer.decode(row, skip_special_tokens=True) for row in output
            ]
        return GeneratorOutput(
            output_token_ids=output,
            decoded_text=decoded_text,
            input_logprobs=self._stack_optional(
                [state.input_logprobs for state in states]
            ),
            probs=self._stack_accumulated(states, "accumulated_probs"),
            logprobs=self._stack_accumulated(states, "accumulated_logprobs"),
        )

    @staticmethod
    def _stack_optional(values):
        if not values or any(value is None for value in values):
            return None
        if all(value.shape == values[0].shape for value in values):
            return torch.stack(values)
        trailing_shape = values[0].shape[1:]
        if any(value.shape[1:] != trailing_shape for value in values):
            raise ValueError("cannot combine prompt logprobs with incompatible shapes")
        max_length = max(value.shape[0] for value in values)
        result = torch.full(
            (len(values), max_length, *trailing_shape),
            -torch.inf,
            dtype=values[0].dtype,
            device=values[0].device,
        )
        for index, value in enumerate(values):
            result[index, max_length - value.shape[0] :] = value
        return result

    @staticmethod
    def _stack_accumulated(states, attribute):
        values = []
        for state in states:
            parts = getattr(state, attribute)
            if not parts:
                return None
            normalized = [
                part.unsqueeze(1) if part.dim() == 2 else part for part in parts
            ]
            concatenated = torch.cat(normalized, dim=1)
            values.append(concatenated.reshape(concatenated.shape[0], -1))
        max_width = max(value.shape[-1] for value in values)
        fill_value = 0.0 if attribute == "accumulated_probs" else -torch.inf
        padded = []
        for value in values:
            if value.shape[-1] == max_width:
                padded.append(value)
                continue
            result = torch.full(
                (value.shape[0], max_width),
                fill_value,
                dtype=value.dtype,
                device=value.device,
            )
            result[:, : value.shape[-1]] = value
            padded.append(result)
        return torch.cat(padded, dim=0)

    @staticmethod
    def _trim_distributions(
        values: Optional[torch.Tensor], keep_count: int
    ) -> Optional[torch.Tensor]:
        if values is None:
            return None
        if values.dim() == 3:
            return values[:, :keep_count, :]
        return values
