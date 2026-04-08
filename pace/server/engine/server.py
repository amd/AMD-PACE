# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
import uuid
import traceback

from typing import Callable, List, Dict, Any, Tuple
from pace.utils.logging import (
    PACE_INFO,
    PACE_DEBUG,
    PACE_ERROR,
    PACE_WARNING,
    PACE_ASSERT,
)
from transformers import AutoTokenizer
from pace.llm import LLMModel
from pace.llm.sampler import Sampler
from pace.llm.attention import (
    AttentionBackendType,
    KVCacheType,
    KVCacheManager,
    create_cache,
)
from pace.llm.configs import OperatorConfig, PardSpecDecodeConfig
from pace.llm import SamplingConfig
from pace.llm.configs import LLMOperatorType, SamplingMode
from pace.llm.ops import LLMBackendType
from pace.llm.outputs import SamplerOutput
from pace.llm.speculative import (
    SpeculativeDecoder,
    create_speculative_decoder,
)
from typing import Optional
from pace.llm.stopping_criteria import StoppingCriteria
from pace.server.engine.utils import ModelConfig, TorchDtypeResolver, PrefillRequest
from enum import Enum, auto


def inference_config_to_sampling_config(
    infConfig: Optional[Dict[str, Any]],
) -> SamplingConfig:
    """Convert InferenceConfig to SamplingConfig."""
    PACE_DEBUG(f"Converting infConfig to SamplingConfig: {infConfig}")
    if infConfig is not None:
        return SamplingConfig(**infConfig)
    return SamplingConfig()


class State(Enum):
    """
    Enumeration for the possible states of a sequence.
    """

    PREFILL = auto()
    DECODING = auto()


class Sequence:
    """
    Contains information and state about an incoming sequence request.

    This class holds the input string, tracks its current processing state
    (e.g., WAITING, PREFILL, DECODING), and assigns a unique ID to each
    request for easy identification. It also contains a KV_Manager object
    to manage its KV Cache and respective sampling parameters.

    Attributes:
        id (uuid.UUID): A unique identifier for the sequence request.
        input_request (List[str]): The data buffer containing the input,
                                   represented as a list of strings.
        state (State): The current processing state of the sequence.
        kv_manager (pace.kv_cache.kv_manager.KVManager): The KV Cache manager for this sequence.
    """

    def __init__(
        self,
        req_id: uuid.UUID,
        input_request: str,
        tokenizer: Any,
        model_config: Any,
        cache_backend,
        _prepare_sampling_config: Callable,
        genConfig: Optional[Dict[str, Any]] = None,
        mlperf_mode: bool = False,
        spec_headroom: int = 0,
    ):
        """
        Initializes a new Sequence instance.

        Args:
            req_id (uuid.UUID): Unique identifier for this request.
            input_request (str): The input data for the sequence.
            tokenizer: The tokenizer instance.
            model_config: The model configuration.
            cache_backend: The cache backend (ContiguousCache or PagedCache).
            genConfig (Optional[Dict]): Generation configuration parameters.
            mlperf_mode (bool): If True, skip text decoding (only return token_ids)
                               for MLPerf Server scenario.
        """
        if not isinstance(input_request, (str, list)):
            raise TypeError("input_request must be a list of strings.")

        self.id: uuid.UUID = req_id
        self.input_request: str = input_request
        self.state: State = State.PREFILL
        self.mlperf_mode: bool = mlperf_mode  # MLPerf Server: response token_id only

        try:
            self.input_encoded = tokenizer.encode_plus(
                self.input_request, return_tensors="pt", padding="longest"
            )
        except Exception as e:
            PACE_INFO(f"[ERROR] Failed to encode input: {e}")
            raise

        try:
            user_sampling_config = (
                inference_config_to_sampling_config(genConfig) if genConfig else None
            )

            self.sampling_config: SamplingConfig = _prepare_sampling_config(
                user_sampling_config=user_sampling_config,
                initial_decoder_input_length=self.input_encoded.input_ids.shape[-1],
                model_max_new_tokens=model_config.max_position_embeddings,
            )
        except Exception as e:
            PACE_WARNING(f"Failed to prepare sampling config: {e}")
            PACE_WARNING(f"Exception type: {type(e)}")
            traceback.print_exc()
            raise

        self.output_str = []

        input_len = self.input_encoded.input_ids.size(-1)
        max_seq_length = self.sampling_config.max_new_tokens + input_len

        # Pre-allocate token buffer to avoid torch.cat on every decode step.
        self._token_buffer = torch.zeros(
            1,
            max_seq_length,
            dtype=torch.long,
            device=self.input_encoded.input_ids.device,
        )
        self._token_buffer[0, :input_len] = self.input_encoded.input_ids[0]
        self._token_len = input_len
        self.input_encoded.input_ids = self._token_buffer[:, :input_len]

        PACE_INFO(f"Sampling config: {self.sampling_config}")
        self.sampler = Sampler(self.sampling_config, self.input_encoded.input_ids)

        self.kv_cache_manager = cache_backend.create_context(
            model_config,
            max_seq_length,
            spec_headroom=spec_headroom,
            token=str(req_id),
        )

        self.stopping_criteria = StoppingCriteria(
            self.sampling_config, self.input_encoded.input_ids, tokenizer
        )

        # Position tracking
        self.num_computed_tokens = 0

        # Draft-model state (populated during prefill when spec decoding is enabled)
        self.draft_kv_cache_manager: Optional[KVCacheManager] = None
        self.draft_num_computed_tokens: Optional[torch.Tensor] = None

        PACE_INFO(
            f"Sequence object(kv_cache + metadata) created for request id{self.id}"
        )

    def append_token(self, token_id: int):
        """Append a token to the pre-allocated buffer (no torch.cat)."""
        PACE_ASSERT(
            self._token_len < self._token_buffer.shape[1],
            f"Token buffer overflow: {self._token_len} >= {self._token_buffer.shape[1]}",
        )
        self._token_buffer[0, self._token_len] = token_id
        self._token_len += 1
        self.input_encoded.input_ids = self._token_buffer[:, : self._token_len]

    def compute_positions(self, num_tokens: int) -> torch.Tensor:
        """Compute positions for next model step. Returns [1, num_tokens]."""
        return torch.arange(
            self.num_computed_tokens,
            self.num_computed_tokens + num_tokens,
            dtype=torch.long,
        ).unsqueeze(0)

    def set_prefill(self) -> None:
        """Sets the sequence state to PREFILL."""
        PACE_DEBUG(
            f"Sequence {self.id}: State changed from {self.state.name} -> PREFILL"
        )
        self.state = State.PREFILL

    def set_decoding(self) -> None:
        """Sets the sequence state to DECODING."""
        PACE_DEBUG(
            f"Sequence {self.id}: State changed from {self.state.name} -> DECODING"
        )
        self.state = State.DECODING

    def __str__(self) -> str:
        """Provides a human-readable string representation of the object."""
        return (
            f"Sequence:\n"
            f"  ID     : {self.id}\n"
            f"  State  : {self.state.name}\n"
            f"  Request: {''.join(self.input_request)}"
        )

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the object."""
        return str(self)


class ModelExecutor:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.model_config = None
        self.kv_cache_type = None
        self._cache_backend = None
        self._spec_decoder: Optional[SpeculativeDecoder] = None

        self.prefill_queue: Dict[uuid.UUID, Sequence] = {}
        self.decode_queue: Dict[uuid.UUID, Sequence] = {}
        self._all_greedy: bool = True  # empty queue is trivially all-greedy
        self._all_mlperf: bool = True  # empty queue is trivially all-mlperf

    def add_to_prefill_queue(self, reqs: List[Sequence]) -> None:
        """
        Creates Sequence objects for new prompts and adds them to the prefill queue.
        """
        for req in reqs:
            # Assuming a single prompt per sequence for now
            req.set_prefill()
            self.prefill_queue[req.id] = req

    def move_to_decode_queue(self, req: Sequence) -> None:
        """
        Moves a sequence from the prefill stage to the decode queue.
        """
        PACE_DEBUG(f"Moving request {req.id} to decode queue")
        req.set_decoding()
        self.prefill_queue.pop(req.id, None)
        self.decode_queue[req.id] = req
        self._all_greedy = self._all_greedy and req.sampler._greedy_fast_path
        self._all_mlperf = self._all_mlperf and req.mlperf_mode

    @staticmethod
    def batch_input_ids(input_ids_list, pad_token=0):
        # input_ids_list: list of tensors, each of shape (1, seq_len)
        max_len = max(ids.shape[1] for ids in input_ids_list)
        batch_size = len(input_ids_list)

        # Initialize a tensor of shape (batch_size, max_len) with pad_token_id
        batch_tensor = torch.full(
            (batch_size, max_len), pad_token, dtype=input_ids_list[0].dtype
        )

        for i, ids in enumerate(input_ids_list):
            seq_len = ids.shape[1]
            # Left padding: place the sequence at the end of the tensor
            batch_tensor[i, max_len - seq_len :] = ids.squeeze(0)

        return batch_tensor

    def load_model(self, mConfig: ModelConfig) -> None:
        """Load the model and tokenizer."""
        PACE_INFO(
            f"Loading model '{mConfig.modelId}' with dtype '{mConfig.dataType}'... kvCacheType: '{mConfig.kvCacheType}'"
        )
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(mConfig.modelId)
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Build OperatorConfig from individual backend fields
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
                        AttentionBackendType.JIT,
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
            self.kv_cache_type = KVCacheType(mConfig.kvCacheType.lower())

            self._model = LLMModel.for_serving(
                mConfig.modelId,
                dtype=TorchDtypeResolver.resolve(mConfig.dataType),
                kv_cache_type=self.kv_cache_type,
                opconfig=opConfig,
            )

            cache_kwargs = {}
            if mConfig.kv_cache_memory_gb is not None:
                cache_kwargs["kv_cache_memory_gb"] = mConfig.kv_cache_memory_gb
            cache_kwargs["dtype"] = TorchDtypeResolver.resolve(mConfig.dataType)
            self.model_config = self._model.get_config()
            self._cache_backend = create_cache(
                self.kv_cache_type,
                model_config=self.model_config,
                **cache_kwargs,
            )

            # Load speculative decoder if configured
            if mConfig.spec_config is not None:
                spec_type = mConfig.spec_config.get("type", "pard").lower()
                if spec_type == "pard":
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
                else:
                    raise ValueError(
                        f"Unknown speculative decoder type '{spec_type}' in spec_config. "
                        "Supported types: 'pard'."
                    )
                self._spec_decoder = create_speculative_decoder(
                    spec_cfg,
                    dtype=TorchDtypeResolver.resolve(mConfig.dataType),
                    opconfig=opConfig,
                    cache_type=self.kv_cache_type,
                )
                PACE_INFO(f"Speculative decoder loaded: {spec_cfg}")
        else:
            PACE_INFO("Model already loaded, skipping load.")

    def prefill(
        self,
        request_data: PrefillRequest,
        chunked: bool = False,
        chunk_sizes: List[Tuple[int, int]] = [(0, -1)],
    ) -> Any:
        """Run prefill step for the given prompts. Creates Sequence from request_data (same as mlperf_scripts)."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be loaded before running prefill."
            )
        prompt = request_data.prompt
        reqID = request_data.request_id
        generation_config = request_data.gen_config

        if not isinstance(reqID, uuid.UUID):
            reqID = (
                uuid.UUID(str(reqID))
                if isinstance(reqID, str)
                else uuid.UUID(int(reqID))
            )

        PACE_DEBUG(f"Config: {generation_config}")

        # Compute spec headroom for KV cache sizing (same as generator)
        spec_headroom = 0
        if self._spec_decoder is not None:
            spec_headroom = self._spec_decoder.num_speculative_tokens + 1

        req: Sequence = Sequence(
            req_id=reqID,
            input_request=prompt,
            tokenizer=self._tokenizer,
            model_config=self.model_config,
            cache_backend=self._cache_backend,
            _prepare_sampling_config=self._model.generator._prepare_sampling_config,
            genConfig=generation_config,
            mlperf_mode=request_data.mlperf_mode,
            spec_headroom=spec_headroom,
        )

        result_buffer: Dict[str, Dict[str, Any]] = {}
        self.prefill_queue[req.id] = req

        try:
            # Pass only input_ids with positions
            input_ids = req.input_encoded.input_ids
            positions = req.compute_positions(input_ids.shape[-1])

            meta = self._cache_backend.build_prefill_metadata(
                req.kv_cache_manager, input_ids.shape[-1]
            )
            pa_kwargs = {"paged_attn_metadata": meta} if meta is not None else {}

            model_out = self._model.step(
                input_ids,
                positions,
                req.kv_cache_manager,
                **pa_kwargs,
            )
        except Exception as e:
            PACE_ERROR(f"Model step failed: {e}")
            result_buffer[str(req.id)] = {
                "output": "",
                "status": "ERROR",
                "error": str(e),
            }
            self.remove_sequences([req.id])
            return result_buffer

        # Update position counter after prefill
        req.num_computed_tokens += req.input_encoded.input_ids.shape[-1]

        # If speculative decoding is enabled, warm up the draft KV cache
        if self._spec_decoder is not None:
            PACE_ASSERT(
                req.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH,
                "Speculative decoding is only supported for greedy "
                f"search sampling mode but got {req.sampling_config.sampling_mode}",
            )
            max_seq_length = (
                req.sampling_config.max_new_tokens
                + req.input_encoded.input_ids.size(-1)
            )
            draft_cache_len = max_seq_length
            if self.kv_cache_type == KVCacheType.PAGED:
                draft_cache_len += self._spec_decoder.num_speculative_tokens
            req.draft_kv_cache_manager = self._spec_decoder.create_draft_kv_cache(
                draft_cache_len, self.kv_cache_type
            )
            batch_size = req.input_encoded.input_ids.size(0)
            req.draft_num_computed_tokens = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=req.input_encoded.input_ids.device,
            )
            draft_meta = self._cache_backend.build_prefill_metadata(
                req.draft_kv_cache_manager,
                req.input_encoded.input_ids.shape[-1],
            )
            draft_pa = (
                {"paged_attn_metadata": draft_meta} if draft_meta is not None else {}
            )
            self._spec_decoder.draft_prefill(
                req.input_encoded.input_ids,
                positions,
                req.draft_kv_cache_manager,
                **draft_pa,
            )
            req.draft_num_computed_tokens += req.input_encoded.input_ids.shape[-1]

        next_token_logits = model_out.logits[:, -1, :].clone().float()

        sampler_output: SamplerOutput = req.sampler.sample(
            req.input_encoded.input_ids, next_token_logits
        )
        output_token_id = sampler_output.next_tokens
        req.append_token(output_token_id.item())
        # MLPerf mode: skip text decoding, only store token_id
        if req.mlperf_mode:
            req.output_str.append(str(output_token_id[0].item()))
        else:
            req.output_str.append(
                self._tokenizer.decode(output_token_id[0], skip_special_tokens=True)
            )

        token_id_int = output_token_id[0].item()
        decoded_token = req.output_str[-1]

        if req.stopping_criteria.stop_now(req.input_encoded.input_ids).item():
            PACE_INFO(
                f"Sequence {req.id} completed during prefill "
                f"due to stopping criteria."
            )
            self.remove_sequences([req.id])
            result_buffer[str(req.id)] = {
                "output": decoded_token,
                "token_id": token_id_int,
                "status": "COMPLETED",
            }
            return result_buffer

        self.move_to_decode_queue(req)

        # Use string key for result buffer (scheduler expects string keys)
        result_buffer[str(req.id)] = {
            "output": decoded_token,
            "token_id": token_id_int,
            "status": "PREFILL_COMPLETED",
        }

        PACE_INFO(
            f"Prefill completed for request {req.id}. Current output: {req.output_str[-1]}"
        )
        return result_buffer

    def decode(self):
        """Run decode step (standard or speculative)."""
        if self._model is None:
            raise RuntimeError("Model must be loaded before running decode.")

        # Create a copy of the keys to avoid modifying dict during iteration
        decode_queue_keys = list(self.decode_queue.keys())

        if not decode_queue_keys:
            PACE_INFO("No sequences in decode queue, returning empty result")
            return {}

        if self._spec_decoder is not None:
            return self._speculative_decode(decode_queue_keys)
        return self._standard_decode(
            decode_queue_keys,
            all_greedy=self._all_greedy,
            all_mlperf=self._all_mlperf,
        )

    # Standard (non-speculative) decode
    def _standard_decode(self, decode_queue_keys, all_greedy, all_mlperf=False):
        """Original single-token decode loop."""
        current_batch_size = len(decode_queue_keys)
        PACE_DEBUG(f"Current decode batch size: {current_batch_size}")
        result_buffer: Dict[uuid.UUID, Dict[str, Any]] = {}
        batched_input_ids = []
        batch_positions_list = []
        batch_kv_cache_managers_list = []
        try:
            for req_id in decode_queue_keys:
                req = self.decode_queue[req_id]
                batched_input_ids.append(req.input_encoded.input_ids[:, -1:])
                batch_positions_list.append(req.compute_positions(1))
                batch_kv_cache_managers_list.append(req.kv_cache_manager)
            batch_input_ids = self.batch_input_ids(batched_input_ids)
            # Batch positions: cat for tensor path, list for list path
            if len(batch_positions_list) == 1:
                batch_positions = batch_positions_list[0]
            else:
                batch_positions = torch.cat(batch_positions_list, dim=0)
        except Exception as e:
            PACE_WARNING(f"Failed to prepare batch data: {e}")
            raise
        try:
            merged = self._cache_backend.merge_contexts(batch_kv_cache_managers_list)
            meta = getattr(merged, "paged_attn_metadata", None)
            pa_kwargs = {"paged_attn_metadata": meta} if meta is not None else {}
            model_out = self._model.step(
                batch_input_ids,
                batch_positions,
                merged,
                **pa_kwargs,
            )
            next_token_logits = model_out.logits[:, -1, :].clone().float()

            if all_greedy:
                PACE_ASSERT(
                    next_token_logits is not None
                    and torch.isnan(next_token_logits).sum() == 0,
                    "NaN values found in logits during greedy decode step, failing decode to prevent"
                    " invalid token generation. Please check the model and input data for issues.",
                )
                next_tokens_batch = next_token_logits.argmax(dim=-1)
            else:
                next_tokens_batch = torch.empty(
                    len(decode_queue_keys), dtype=torch.long
                )
                for i, req_id in enumerate(decode_queue_keys):
                    req = self.decode_queue[req_id]
                    sampler_output = req.sampler.sample(
                        req.input_encoded.input_ids, next_token_logits[i : i + 1]
                    )
                    next_tokens_batch[i] = sampler_output.next_tokens[0, 0]

            # Batch decode tokens (skip entirely if all requests are mlperf).
            decoded_tokens = (
                None
                if all_mlperf
                else self._tokenizer.batch_decode(
                    next_tokens_batch.tolist(), skip_special_tokens=True
                )
            )

            # Update per-sequence state.
            completed_ids = []
            for i, req_id in enumerate(decode_queue_keys):
                req = self.decode_queue[req_id]
                token_id_int = int(next_tokens_batch[i].item())
                req.append_token(token_id_int)
                req.num_computed_tokens += 1

                if req.mlperf_mode:
                    decoded_token = ""
                else:
                    decoded_token = decoded_tokens[i]
                    req.output_str.append(decoded_token)

                # Check if this sequence should stop
                if req.stopping_criteria.stop_now(req.input_encoded.input_ids).item():
                    PACE_INFO(f"Stopping criteria met for sequence {req.id}.")
                    result_buffer[req.id] = {
                        "output": decoded_token,
                        "token_id": token_id_int,
                        "status": "COMPLETED",
                        "stop_reason": "stop_criteria",
                    }
                    completed_ids.append(req_id)
                    continue
                req.kv_cache_manager = batch_kv_cache_managers_list[i]

                result_buffer[str(req.id)] = {
                    "output": decoded_token,
                    "token_id": token_id_int,
                    "status": "DECODING_IN_PROGRESS",
                }

            # Batch remove completed sequences.
            if completed_ids:
                self.remove_sequences(completed_ids)
        except Exception as e:
            PACE_WARNING(f"Decode step failed for batch decode. Exception: {e}")
            traceback.print_exc()
            for failed_req_id in decode_queue_keys:
                PACE_WARNING(
                    f"Marking sequence {failed_req_id} as failed due to decode error."
                )
                result_buffer[failed_req_id] = {
                    "output": "",
                    "status": "ERROR",
                    "error": str(e),
                }
                if failed_req_id in self.decode_queue:
                    self.remove_sequences([failed_req_id])

        return result_buffer

    # Speculative decode  (3-phase: draft → target → verify)
    def _speculative_decode(self, decode_queue_keys):
        """Speculative decode with batched draft and target forwards."""
        sd = self._spec_decoder
        draft_size = sd.num_speculative_tokens
        batch_size = len(decode_queue_keys)
        PACE_DEBUG(f"Speculative decode batch size: {batch_size}")
        result_buffer: Dict[uuid.UUID, Dict[str, Any]] = {}

        try:
            reqs = [self.decode_queue[rid] for rid in decode_queue_keys]

            # Phase 1: Per-sequence draft-model forward
            # Each sequence may have a different number of accepted tokens
            # since the last draft step, so pard_input lengths can vary.
            # Run draft forwards per-sequence; the draft model is small.
            speculated_tokens_list = []
            for req in reqs:
                nc = req.draft_num_computed_tokens
                n_missing = req.input_encoded.input_ids.shape[-1] - int(nc[0].item())
                draft_context = req.input_encoded.input_ids[:, -n_missing:]
                pard_input = sd.prepare_draft_input(draft_context)

                offsets = torch.arange(pard_input.shape[-1], dtype=torch.long)
                pos = nc.unsqueeze(1) + offsets.unsqueeze(0)

                draft_past = int(nc[0].item())
                draft_meta = self._cache_backend.build_prefill_metadata(
                    req.draft_kv_cache_manager,
                    pard_input.shape[-1],
                    past_len=draft_past,
                )
                draft_pa = (
                    {"paged_attn_metadata": draft_meta}
                    if draft_meta is not None
                    else {}
                )
                draft_out = sd.draft_step(
                    pard_input, pos, req.draft_kv_cache_manager, **draft_pa
                )
                req.draft_num_computed_tokens += pard_input.shape[-1]
                spec_toks = draft_out.logits[:, -draft_size:].argmax(dim=-1)
                speculated_tokens_list.append(spec_toks)

            # Phase 2: Batched target-model forward
            target_inputs = []
            target_positions_list = []
            target_kv_caches = []
            speculated_inputs_list = []

            for i, req in enumerate(reqs):
                last_token = req.input_encoded.input_ids[:, -1:]
                extended = torch.cat([last_token, speculated_tokens_list[i]], dim=-1)
                speculated_inputs_list.append(extended)
                target_inputs.append(extended)

                pos = req.compute_positions(extended.shape[-1])
                target_positions_list.append(pos)
                target_kv_caches.append(req.kv_cache_manager)

            batched_target_input = self.batch_input_ids(target_inputs)
            if batch_size == 1:
                batched_target_pos = target_positions_list[0]
            else:
                batched_target_pos = torch.cat(target_positions_list, dim=0)

            target_kv_arg = target_kv_caches[0] if batch_size == 1 else target_kv_caches
            target_q_len = target_inputs[0].shape[-1]
            merged_target = self._cache_backend.merge_contexts(
                target_kv_caches, query_len=target_q_len
            )
            target_meta = getattr(merged_target, "paged_attn_metadata", None)
            target_pa = (
                {"paged_attn_metadata": target_meta} if target_meta is not None else {}
            )
            target_out = self._model.step(
                batched_target_input,
                batched_target_pos,
                target_kv_arg,
                **target_pa,
            )

            # Update target position counters
            for i, req in enumerate(reqs):
                req.num_computed_tokens += target_inputs[i].shape[-1]

            # Phase 3: Per-sequence verify
            for i, req_id in enumerate(decode_queue_keys):
                req = reqs[i]
                logits_i = target_out.logits[i : i + 1, -draft_size - 1 :, :]
                logits_i = logits_i.clone().float()

                sampler_output: SamplerOutput = req.sampler.sample(
                    req.input_encoded.input_ids, logits_i
                )
                next_tokens = sampler_output.next_tokens
                if next_tokens.dim() == 3:
                    next_tokens = next_tokens.squeeze(-1)

                result = sd.verify(
                    next_tokens,
                    speculated_inputs_list[i],
                    draft_kv_cache=req.draft_kv_cache_manager,
                    draft_num_computed=req.draft_num_computed_tokens,
                )
                accepted = result.accepted_tokens

                # Trim target KV cache
                if result.target_kv_cache_trim > 0:
                    req.kv_cache_manager.remove_cache(result.target_kv_cache_trim)
                    req.num_computed_tokens -= result.target_kv_cache_trim

                # Append all accepted tokens (vectorized slice-assign).
                # Clamp to remaining buffer capacity to avoid overflow.
                n_acc = accepted.shape[-1]
                remaining = req._token_buffer.shape[1] - req._token_len
                n_acc = min(n_acc, remaining)
                if n_acc < accepted.shape[-1]:
                    accepted = accepted[:, :n_acc]
                num_accepted = n_acc

                if n_acc > 0:
                    req._token_buffer[0, req._token_len : req._token_len + n_acc] = (
                        accepted[0]
                    )
                    req._token_len += n_acc
                    req.input_encoded.input_ids = req._token_buffer[:, : req._token_len]

                # Decode and stream all accepted tokens
                decoded_tokens = self._tokenizer.decode(
                    accepted[0], skip_special_tokens=True
                )

                # Check stopping criteria
                if req.stopping_criteria.stop_now(
                    req.input_encoded.input_ids, num_new_tokens=num_accepted
                ).item():
                    PACE_INFO(f"Stopping criteria met for sequence {req.id}.")
                    result_buffer[req.id] = {
                        "output": decoded_tokens,
                        "status": "COMPLETED",
                        "num_tokens_generated": num_accepted,
                    }
                    self.remove_sequences([req_id])
                    continue

                PACE_DEBUG(
                    f"Speculative decode for {req.id}: accepted {num_accepted} "
                    f"tokens: '{decoded_tokens}'"
                )
                req.output_str.append(decoded_tokens)
                result_buffer[req.id] = {
                    "output": decoded_tokens,
                    "status": "DECODING_IN_PROGRESS",
                    "num_tokens_generated": num_accepted,
                }

        except Exception as e:
            PACE_WARNING(f"Speculative decode failed. Exception: {e}")
            traceback.print_exc()
            for failed_req_id in decode_queue_keys:
                PACE_WARNING(
                    f"Marking sequence {failed_req_id} as failed due to decode error."
                )
                result_buffer[str(failed_req_id)] = {
                    "output": "",
                    "status": "ERROR",
                    "error": str(e),
                }
                if failed_req_id in self.decode_queue:
                    self.remove_sequences([failed_req_id])

        return result_buffer

    def remove_sequences(self, seq_indexes: List[uuid.UUID]) -> None:
        """Remove sequences from both prefill and decode queues."""
        PACE_INFO(
            f"Removing sequence(metadata +kv_cache) for request ids "
            f"{', '.join(str(seq_id) for seq_id in seq_indexes)}"
        )
        for seq_id in seq_indexes:
            seq = None
            queue_name = None
            if seq_id in self.prefill_queue:
                seq = self.prefill_queue.pop(seq_id)
                queue_name = "prefill"
            elif seq_id in self.decode_queue:
                seq = self.decode_queue.pop(seq_id)
                queue_name = "decode"

            if seq is not None:
                seq.kv_cache_manager.remove_cache(len(seq.kv_cache_manager))
                self._cache_backend.remove_context(seq.kv_cache_manager)
                if (
                    hasattr(seq, "draft_kv_cache_manager")
                    and seq.draft_kv_cache_manager is not None
                    and self._spec_decoder is not None
                ):
                    self._spec_decoder.remove_draft_context(seq.draft_kv_cache_manager)
                    seq.draft_kv_cache_manager = None
                PACE_INFO(f"Removed sequence {seq_id} from {queue_name} queue.")

        # Only recompute if flag was False — removing might restore True.
        # If already True, removing a qualifying request can't flip it.
        if not self._all_greedy:
            self._all_greedy = (
                all(r.sampler._greedy_fast_path for r in self.decode_queue.values())
                if self.decode_queue
                else True
            )
        if not self._all_mlperf:
            self._all_mlperf = (
                all(r.mlperf_mode for r in self.decode_queue.values())
                if self.decode_queue
                else True
            )


model_executor = ModelExecutor()
