# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from pace.llm import SamplingConfig
from pace.llm.attention import AttentionBackendType, KVCacheType
from pace.llm.configs import LLMOperatorType, OperatorConfig, PardSpecDecodeConfig
from pace.llm.generation import (
    GenerationRequest,
    GenerationState,
    GenerationStatus,
    NativeGenerationBackendConfig,
    TerminationReason,
)
from pace.llm.ops import LLMBackendType
from pace.server.engine.utils import ModelConfig, PrefillRequest
from pace.utils.dtype import TorchDtypeResolver
from pace.utils.logging import PACE_DEBUG, PACE_INFO


def inference_config_to_sampling_config(
    infConfig: Optional[Dict[str, Any]],
) -> Tuple[SamplingConfig, int]:
    """Convert InferenceConfig to SamplingConfig.

    Strips the router-only ``logprobs_n`` key (top-N for OpenAI logprobs)
    before forwarding to SamplingConfig and returns it separately.
    """
    PACE_DEBUG(f"Converting infConfig to SamplingConfig: {infConfig}")
    logprobs_n = 0
    if infConfig is not None:
        cfg = dict(infConfig)
        logprobs_n = cfg.pop("logprobs_n", 0)
        return SamplingConfig(**cfg), logprobs_n
    return SamplingConfig(), logprobs_n


class ModelExecutor:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.model_config = None
        self.kv_cache_type = None
        self._backend = None

        self.prefill_queue: Dict[uuid.UUID, GenerationState] = {}
        self.decode_queue: Dict[uuid.UUID, GenerationState] = {}

    @property
    def cache_backend(self):
        return (
            getattr(self._backend, "cache_backend", None)
            if self._backend is not None
            else None
        )

    def load_model(self, mConfig: ModelConfig) -> None:
        """Load the model and tokenizer."""
        PACE_INFO(
            f"Loading model '{mConfig.modelId}' with dtype '{mConfig.dataType}'... kvCacheType: '{mConfig.kvCacheType}'"
        )
        if self._model is None:
            dtype = TorchDtypeResolver.resolve(mConfig.dataType)
            self.kv_cache_type = (
                KVCacheType(mConfig.kvCacheType.lower())
                if mConfig.kvCacheType is not None
                else (
                    KVCacheType.PAGED
                    if dtype == torch.float32
                    else KVCacheType.SLAB_POOL
                )
            )
            opConfig = OperatorConfig(
                **{
                    LLMOperatorType.Norm: getattr(
                        LLMBackendType,
                        mConfig.norm_backend.upper(),
                        LLMBackendType.NATIVE,
                    ),
                    LLMOperatorType.QKVProjection: getattr(
                        LLMBackendType,
                        mConfig.qkv_projection_backend.upper(),
                        LLMBackendType.TPP,
                    ),
                    LLMOperatorType.Attention: getattr(
                        AttentionBackendType,
                        mConfig.attention_backend.upper(),
                        AttentionBackendType.SLAB,
                    ),
                    LLMOperatorType.GatedDeltaNet: getattr(
                        LLMBackendType,
                        mConfig.gated_deltanet_backend.upper(),
                        LLMBackendType.NATIVE,
                    ),
                    LLMOperatorType.OutProjection: getattr(
                        LLMBackendType,
                        mConfig.out_projection_backend.upper(),
                        LLMBackendType.TPP,
                    ),
                    LLMOperatorType.MLP: getattr(
                        LLMBackendType, mConfig.mlp_backend.upper(), LLMBackendType.TPP
                    ),
                    LLMOperatorType.LMHead: getattr(
                        LLMBackendType,
                        mConfig.lm_head_backend.upper(),
                        LLMBackendType.NATIVE,
                    ),
                }
            )
            cache_kwargs = {}
            if mConfig.kv_cache_memory_gb is not None:
                cache_kwargs["kv_cache_memory_gb"] = mConfig.kv_cache_memory_gb
            cache_kwargs["dtype"] = dtype

            spec_cfg = None
            if mConfig.spec_config is not None:
                spec_type = mConfig.spec_config.get("type", "pard").lower()
                if spec_type != "pard":
                    raise ValueError(
                        f"Unknown speculative decoder type '{spec_type}' in spec_config. "
                        "Supported types: 'pard'."
                    )
                model_name = mConfig.spec_config.get("model_name")
                if not model_name:
                    raise ValueError(
                        "Invalid spec_config: missing required field 'model_name'."
                    )
                spec_cfg = PardSpecDecodeConfig(
                    model_name_or_path=model_name,
                    num_speculative_tokens=mConfig.spec_config.get(
                        "num_speculative_tokens", 12
                    ),
                    draft_kv_cache_memory_gb=mConfig.spec_config.get(
                        "draft_kv_cache_memory_gb"
                    ),
                )

            backend_config = (
                mConfig.generation_backend_config or NativeGenerationBackendConfig()
            )
            backend_kwargs = {
                "serving": True,
                "dtype": dtype,
                "disable_tqdm": True,
                "kv_cache_type": self.kv_cache_type,
                "spec_config": spec_cfg,
                "opconfig": opConfig,
                "cache_kwargs": cache_kwargs,
            }
            self._backend = backend_config.initialize_backend(
                mConfig.modelId, **backend_kwargs
            )
            self._model = self._backend.model
            self._tokenizer = self._backend.tokenizer
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self.model_config = self._backend.model_config

        else:
            PACE_INFO("Model already loaded, skipping load.")

    def create_sequence(
        self,
        req_id: uuid.UUID,
        input_token_ids: List[int],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> GenerationState:
        if req_id in self.prefill_queue or req_id in self.decode_queue:
            raise ValueError(f"request ID {req_id} is already active")
        user_config, logprobs_n = inference_config_to_sampling_config(generation_config)
        return self._backend.create_request(
            GenerationRequest(
                request_id=str(req_id),
                input_ids=torch.tensor([input_token_ids], dtype=torch.long),
                sampling_config=user_config,
                infer_padding=False,
                logprobs_n=logprobs_n,
            )
        )

    @staticmethod
    def _state_key(state: GenerationState) -> uuid.UUID:
        return uuid.UUID(state.request_id)

    def _delta_entry(self, state: GenerationState, delta) -> Dict[str, Any]:
        entry = {
            "token_ids": delta.token_ids,
            "status": delta.status.name,
            "num_tokens_generated": len(delta.token_ids),
        }
        if delta.status == GenerationStatus.PREFILL_IN_PROGRESS:
            entry["num_computed_tokens"] = delta.computed_prompt_tokens
            entry["actual_input_length"] = state.prompt_length
        if delta.finish_reason is not None:
            entry["stop_reason"] = delta.finish_reason
        logprobs = self._build_logprobs_object(
            delta.token_ids, delta.logprobs, state.logprobs_n
        )
        if logprobs is not None:
            entry["logprobs"] = logprobs
        prompt_logprobs = self._prompt_logprobs_from_tensor(
            state, delta.prompt_logprobs
        )
        if prompt_logprobs is not None:
            entry["prompt_logprobs"] = prompt_logprobs
        if delta.error is not None:
            entry["error"] = delta.error
        return entry

    def _prompt_logprobs_from_tensor(
        self,
        state: GenerationState,
        input_logprobs: Optional[torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """Build prompt logprobs from ``[prompt, vocab]`` distributions.

        The logits at position ``t`` predict prompt token ``t + 1``; the first
        prompt token therefore has no predecessor logprob.
        """
        if input_logprobs is None:
            return None
        prompt_start = getattr(state.backend_state, "prompt_start", 0)
        prompt_ids = state.prompt_token_ids[0, prompt_start:].tolist()
        token_logprobs: List[Optional[float]] = [None]
        top_logprobs: List[List[List[float]]] = [[]]
        for index, token_id in enumerate(prompt_ids[1:], start=1):
            row = input_logprobs[index - 1]
            token_logprobs.append(float(row[token_id].item()))
            if state.logprobs_n > 0:
                values, ids = torch.topk(row, min(state.logprobs_n, row.shape[-1]))
                top_logprobs.append(
                    [
                        [int(candidate), float(value)]
                        for value, candidate in zip(values.tolist(), ids.tolist())
                    ]
                )
            else:
                top_logprobs.append([])
        return {
            "token_ids": prompt_ids,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
        }

    def _backend_prefill(self, request_data: PrefillRequest) -> Dict[str, Any]:
        req_id = request_data.request_id
        if not isinstance(req_id, uuid.UUID):
            req_id = uuid.UUID(str(req_id))
        state = self.create_sequence(
            req_id,
            request_data.prompt,
            request_data.gen_config,
        )
        self.prefill_queue[req_id] = state
        delta = self._backend.prefill([state]).deltas[state.request_id]
        entry = self._delta_entry(state, delta)
        if delta.status == GenerationStatus.PREFILL_COMPLETED:
            self.prefill_queue.pop(req_id, None)
            state.status = GenerationStatus.DECODING_IN_PROGRESS
            self.decode_queue[req_id] = state
        elif delta.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.CANCELLED,
            GenerationStatus.ERROR,
        }:
            self.prefill_queue.pop(req_id, None)
        return {state.request_id: entry}

    def _backend_decode(self) -> Dict[str, Any]:
        if not self.decode_queue:
            return {}
        states = list(self.decode_queue.values())
        result = self._backend.decode(states)
        output = {}
        for state in states:
            delta = result.deltas[state.request_id]
            output[state.request_id] = self._delta_entry(state, delta)
            if delta.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.CANCELLED,
                GenerationStatus.ERROR,
            }:
                self.decode_queue.pop(self._state_key(state), None)
        return output

    def _backend_mixed(
        self,
        prefill_states: List[GenerationState],
        decode_states: List[GenerationState],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result = self._backend.mixed(
            prefill_states,
            decode_states,
            chunk_info=chunk_info_map,
        )
        prefill_output = {}
        decode_output = {}
        for state in prefill_states:
            delta = result.deltas[state.request_id]
            prefill_output[state.request_id] = self._delta_entry(state, delta)
            req_id = self._state_key(state)
            if delta.status == GenerationStatus.PREFILL_COMPLETED:
                self.prefill_queue.pop(req_id, None)
                state.status = GenerationStatus.DECODING_IN_PROGRESS
                self.decode_queue[req_id] = state
            elif delta.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.CANCELLED,
                GenerationStatus.ERROR,
            }:
                self.prefill_queue.pop(req_id, None)
        for state in decode_states:
            delta = result.deltas[state.request_id]
            decode_output[state.request_id] = self._delta_entry(state, delta)
            if delta.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.CANCELLED,
                GenerationStatus.ERROR,
            }:
                self.decode_queue.pop(self._state_key(state), None)
        return prefill_output, decode_output

    def _backend_remove(self, seq_indexes: List[uuid.UUID]) -> None:
        for seq_id in seq_indexes:
            state = self.prefill_queue.pop(seq_id, None)
            if state is None:
                state = self.decode_queue.pop(seq_id, None)
            if state is not None:
                self._backend.release(state, TerminationReason.CANCELLED)

    @staticmethod
    def _build_logprobs_object(
        token_ids: List[int],
        logprobs_tensor: Optional[torch.Tensor],
        logprobs_n: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a numeric logprobs dict from token IDs and logprob values."""
        if logprobs_tensor is None:
            return None

        if logprobs_tensor.dim() == 3:
            logprobs_tensor = logprobs_tensor.squeeze(0)
        if logprobs_tensor.dim() == 1:
            logprobs_tensor = logprobs_tensor.unsqueeze(0)

        out_token_ids: List[int] = []
        token_logprobs: List[float] = []
        top_logprobs: List[List[List[float]]] = []

        for index, row in enumerate(logprobs_tensor):
            token_id = token_ids[index] if index < len(token_ids) else 0
            out_token_ids.append(int(token_id))
            token_logprobs.append(float(row[token_id].item()))

            if logprobs_n > 0:
                values, ids = torch.topk(row, min(logprobs_n, row.shape[-1]))
                top_logprobs.append(
                    [
                        [int(candidate), float(value)]
                        for value, candidate in zip(values.tolist(), ids.tolist())
                    ]
                )
            else:
                top_logprobs.append([])

        return {
            "token_ids": out_token_ids,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
        }

    def prefill(self, request_data: PrefillRequest) -> Any:
        if self._model is None:
            raise RuntimeError("Model must be loaded before running prefill.")
        return self._backend_prefill(request_data)

    def decode(self):
        if self._model is None:
            raise RuntimeError("Model must be loaded before running decode.")
        return self._backend_decode()

    def prefill_decode_mixed_batch(
        self,
        prefill_sequences: List[GenerationState],
        decode_sequences: List[GenerationState],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Model must be loaded before running mixed batch.")
        return self._backend_mixed(prefill_sequences, decode_sequences, chunk_info_map)

    def remove_sequences(self, seq_indexes: List[uuid.UUID]) -> None:
        return self._backend_remove(seq_indexes)


model_executor = ModelExecutor()
