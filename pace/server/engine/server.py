# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
import uuid
import traceback

from typing import Callable, List, Dict, Any, Optional, Tuple
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
from pace.llm.stopping_criteria import StoppingCriteria
from pace.server.engine.unified_batch import UnifiedBatchBuilder
from pace.server.engine.utils import ModelConfig, TorchDtypeResolver, PrefillRequest
from enum import Enum, auto


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
        input_token_ids: List[int],
        model_config: Any,
        cache_backend,
        _prepare_sampling_config: Callable,
        genConfig: Optional[Dict[str, Any]] = None,
        spec_headroom: int = 0,
        tokenizer=None,
    ):
        """
        Initializes a new Sequence instance.

        Args:
            req_id (uuid.UUID): Unique identifier for this request.
            input_token_ids (List[int]): Pre-tokenized prompt as token IDs.
            model_config: The model configuration.
            cache_backend: The cache backend (ContiguousCache or PagedCache).
            _prepare_sampling_config: Callable to prepare sampling config.
            genConfig (Optional[Dict]): Generation configuration parameters.
            spec_headroom (int): Extra headroom for speculative decoding KV cache.
            tokenizer: Optional tokenizer for stop-string matching.
        """
        if not isinstance(input_token_ids, list):
            raise TypeError("input_token_ids must be a list of integers.")

        self.id: uuid.UUID = req_id
        self.input_token_ids: List[int] = input_token_ids
        self.state: State = State.PREFILL

        input_ids_tensor = torch.tensor([input_token_ids], dtype=torch.long)
        input_len = input_ids_tensor.size(-1)

        try:
            if genConfig:
                user_sampling_config, logprobs_n = inference_config_to_sampling_config(
                    genConfig
                )
            else:
                user_sampling_config, logprobs_n = None, 0

            self.logprobs_n: int = logprobs_n

            self.sampling_config: SamplingConfig = _prepare_sampling_config(
                user_sampling_config=user_sampling_config,
                initial_decoder_input_length=input_len,
                model_max_new_tokens=model_config.max_position_embeddings,
            )
        except Exception as e:
            PACE_WARNING(f"Failed to prepare sampling config: {e}")
            PACE_WARNING(f"Exception type: {type(e)}")
            traceback.print_exc()
            raise

        max_seq_length = self.sampling_config.max_new_tokens + input_len

        self._token_buffer = torch.zeros(
            1,
            max_seq_length,
            dtype=torch.long,
            device=input_ids_tensor.device,
        )
        self._token_buffer[0, :input_len] = input_ids_tensor[0]
        self._token_len = input_len

        class _InputEncoded:
            """Lightweight stand-in for the tokenizer BatchEncoding."""

            pass

        self.input_encoded = _InputEncoded()
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

        self.num_computed_tokens = 0

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
            f"  Tokens : {self._token_len}"
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
                        AttentionBackendType.SLAB,
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
                if (
                    spec_cfg.draft_kv_cache_memory_gb is None
                    and mConfig.kv_cache_memory_gb is not None
                ):

                    def _kv(c):
                        h = getattr(c, "num_attention_heads", 1) or 1
                        kv_h = getattr(c, "num_key_value_heads", None) or h
                        return (
                            getattr(c, "num_hidden_layers", 0)
                            * kv_h
                            * getattr(c, "hidden_size", 0)
                            / h
                        )

                    t, d = _kv(self.model_config), _kv(self._spec_decoder.model.config)
                    if t and d:
                        spec_cfg.draft_kv_cache_memory_gb = (
                            mConfig.kv_cache_memory_gb * d / t
                        )
        else:
            PACE_INFO("Model already loaded, skipping load.")

    @staticmethod
    def _build_logprobs_object(
        token_ids: List[int],
        logprobs_tensor: Optional[torch.Tensor],
        logprobs_n: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a *numeric* logprobs dict (token IDs + values, no strings).

        Detokenization is router-native: the engine emits token IDs and the
        router decodes them into strings and computes ``text_offset``. This
        avoids double detokenization and keeps ``tokens``/``top_logprobs``/
        ``text_offset`` consistent with the response ``text`` (which the
        router also detokenizes). Mirrors vLLM's ``LogprobsTensors`` (IDs +
        values) feeding the ``LogprobsProcessor``.

        Returns ``None`` when logprobs are not requested (tensor is None).
        When ``logprobs_n == 0`` the per-token ``top_logprobs`` list is empty.

        Shape emitted::

            {
              "token_ids":      [int, ...],
              "token_logprobs": [float, ...],
              "top_logprobs":   [[[tok_id, logprob], ...], ...],
            }

        Shape expectations for *logprobs_tensor*:
          - 2-D ``[1, vocab]`` for a single-token decode step, or
          - 2-D ``[num_tokens, vocab]`` for multi-token (speculative) steps.
        """
        if logprobs_tensor is None:
            return None

        if logprobs_tensor.dim() == 3:
            logprobs_tensor = logprobs_tensor.squeeze(0)
        if logprobs_tensor.dim() == 1:
            logprobs_tensor = logprobs_tensor.unsqueeze(0)

        out_token_ids: List[int] = []
        token_logprobs: List[float] = []
        top_logprobs: List[List[List[float]]] = []

        num_tokens = logprobs_tensor.shape[0]
        for t in range(num_tokens):
            row = logprobs_tensor[t]
            tid = token_ids[t] if t < len(token_ids) else 0
            out_token_ids.append(int(tid))
            token_logprobs.append(float(row[tid].item()))

            if logprobs_n > 0:
                topk_vals, topk_ids = torch.topk(row, min(logprobs_n, row.shape[-1]))
                top_logprobs.append(
                    [
                        [int(i), float(v)]
                        for v, i in zip(topk_vals.tolist(), topk_ids.tolist())
                    ]
                )
            else:
                top_logprobs.append([])

        return {
            "token_ids": out_token_ids,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
        }

    @staticmethod
    def _build_prompt_logprobs_object(
        prompt_ids: List[int],
        full_logits: torch.Tensor,
        logprobs_n: int,
    ) -> Dict[str, Any]:
        """Build a *numeric* prompt (echo) logprobs dict from prefill logits.

        Mirrors vLLM's ``prompt_logprobs``: logits at position ``t`` predict
        token ``t+1``, so ``prompt_logprobs[0]`` is ``None`` and
        ``prompt_logprobs[i] = logprob(token[i] | logits[i-1])``. Emits token
        IDs (no strings) — the router detokenizes.

        ``full_logits`` may be ``[1, seq_len, vocab]`` or ``[seq_len, vocab]``
        and must cover the whole prompt (``seq_len == len(prompt_ids)``).
        """
        if full_logits.dim() == 3:
            full_logits = full_logits.squeeze(0)
        input_lp = torch.log_softmax(full_logits, dim=-1, dtype=torch.float32)
        inp_token_ids: List[int] = []
        inp_token_logprobs: List[Optional[float]] = []
        inp_top_logprobs: List[List[List[float]]] = []
        for i, tid in enumerate(prompt_ids):
            inp_token_ids.append(int(tid))
            if i == 0:
                inp_token_logprobs.append(None)
                inp_top_logprobs.append([])
            else:
                row = input_lp[i - 1]
                inp_token_logprobs.append(float(row[tid].item()))
                if logprobs_n > 0:
                    topk_vals, topk_ids = torch.topk(
                        row, min(logprobs_n, row.shape[-1])
                    )
                    inp_top_logprobs.append(
                        [
                            [int(k), float(v)]
                            for v, k in zip(topk_vals.tolist(), topk_ids.tolist())
                        ]
                    )
                else:
                    inp_top_logprobs.append([])
        return {
            "token_ids": inp_token_ids,
            "token_logprobs": inp_token_logprobs,
            "top_logprobs": inp_top_logprobs,
        }

    def prefill(
        self,
        request_data: PrefillRequest,
        chunked: bool = False,
        chunk_sizes: List[Tuple[int, int]] = [(0, -1)],
    ) -> Any:
        """Run prefill step for pre-tokenized prompts (List[int]).

        The router is responsible for all tokenization/detokenization.
        The engine only works with token IDs.
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before running prefill.")
        prompt_token_ids = request_data.prompt
        reqID = request_data.request_id
        generation_config = request_data.gen_config

        if not isinstance(reqID, uuid.UUID):
            reqID = (
                uuid.UUID(str(reqID))
                if isinstance(reqID, str)
                else uuid.UUID(int(reqID))
            )

        PACE_DEBUG(f"Config: {generation_config}")

        spec_headroom = 0
        if self._spec_decoder is not None:
            spec_headroom = self._spec_decoder.num_speculative_tokens + 1

        req: Sequence = Sequence(
            req_id=reqID,
            input_token_ids=prompt_token_ids,
            model_config=self.model_config,
            cache_backend=self._cache_backend,
            _prepare_sampling_config=self._model.generator._prepare_sampling_config,
            genConfig=generation_config,
            spec_headroom=spec_headroom,
            tokenizer=self._tokenizer,
        )

        result_buffer: Dict[str, Dict[str, Any]] = {}
        self.prefill_queue[req.id] = req

        try:
            input_ids = req.input_encoded.input_ids
            full_seq_len = input_ids.shape[-1]
            prefix_hit_len = self._match_and_claim_prefix(req)

            if prefix_hit_len > 0:
                suffix_ids = input_ids[:, prefix_hit_len:]
                positions = torch.arange(
                    prefix_hit_len, full_seq_len, dtype=torch.long
                ).unsqueeze(0)
                meta = self._cache_backend.build_prefill_metadata(
                    req.kv_cache_manager,
                    suffix_ids.shape[-1],
                    past_len=prefix_hit_len,
                )
                pa_kwargs = {"paged_attn_metadata": meta} if meta is not None else {}
                model_out = self._model.step(
                    suffix_ids, positions, req.kv_cache_manager, **pa_kwargs
                )
            else:
                positions = req.compute_positions(full_seq_len)
                meta = self._cache_backend.build_prefill_metadata(
                    req.kv_cache_manager, full_seq_len
                )
                pa_kwargs = {"paged_attn_metadata": meta} if meta is not None else {}
                model_out = self._model.step(
                    input_ids, positions, req.kv_cache_manager, **pa_kwargs
                )

            if getattr(self._cache_backend, "prefix_cache_enabled", False):
                self._cache_backend.insert_prefix(
                    req.kv_cache_manager, input_ids[0].tolist()
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

        req.num_computed_tokens += req.input_encoded.input_ids.shape[-1]

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
            # The draft cache is fresh and prefills the full prompt, so it needs
            # full-length positions even when the target model reused a cached
            # prefix (where `positions` above is suffix-only).
            draft_positions = req.compute_positions(full_seq_len)
            self._spec_decoder.draft_prefill(
                req.input_encoded.input_ids,
                draft_positions,
                req.draft_kv_cache_manager,
                **draft_pa,
            )
            req.draft_num_computed_tokens += req.input_encoded.input_ids.shape[-1]

        # Compute prompt (input) logprobs from the full prefill logits when
        # echo+logprobs is requested. These are emitted as a *separate*
        # ``prompt_logprobs`` field (mirroring vLLM's prompt_logprobs vs.
        # sample logprobs split) so the router can assemble the echo/
        # loglikelihood response explicitly instead of concatenating and
        # trimming forced tokens.
        # Logits at position t predict token t+1, so prompt_logprobs[0] is
        # None and prompt_logprobs[i] = logprob(token[i] | logits[i-1]).
        prompt_logprobs_obj = None
        if (
            req.sampling_config.return_logprobs
            and req.sampling_config.return_input_logprobs
        ):
            prompt_ids = req.input_encoded.input_ids.squeeze(0).tolist()
            prompt_logprobs_obj = self._build_prompt_logprobs_object(
                prompt_ids, model_out.logits, req.logprobs_n
            )

        next_token_logits = model_out.logits[:, -1, :].clone().float()

        sampler_output: SamplerOutput = req.sampler.sample(
            req.input_encoded.input_ids, next_token_logits
        )
        output_token_id = sampler_output.next_tokens
        req.append_token(output_token_id.item())

        token_id_int = output_token_id[0].item()

        logprobs_obj = self._build_logprobs_object(
            [token_id_int],
            sampler_output.logprobs,
            req.logprobs_n,
        )

        if req.stopping_criteria.stop_now(req.input_encoded.input_ids).item():
            PACE_INFO(
                f"Sequence {req.id} completed during prefill "
                f"due to stopping criteria."
            )
            self.remove_sequences([req.id])
            entry: Dict[str, Any] = {
                "token_ids": [token_id_int],
                "status": "COMPLETED",
                "num_tokens_generated": 1,
                "stop_reason": req.stopping_criteria.stop_reason,
            }
            if logprobs_obj is not None:
                entry["logprobs"] = logprobs_obj
            if prompt_logprobs_obj is not None:
                entry["prompt_logprobs"] = prompt_logprobs_obj
            result_buffer[str(req.id)] = entry
            return result_buffer

        self.move_to_decode_queue(req)

        entry = {
            "token_ids": [token_id_int],
            "status": "PREFILL_COMPLETED",
            "num_tokens_generated": 1,
        }
        if logprobs_obj is not None:
            entry["logprobs"] = logprobs_obj
        if prompt_logprobs_obj is not None:
            entry["prompt_logprobs"] = prompt_logprobs_obj
        result_buffer[str(req.id)] = entry

        PACE_INFO(
            f"Prefill completed for request {req.id}. First token_id: {token_id_int}"
        )
        return result_buffer

    def decode(self):
        """Run decode step (standard or speculative)."""
        if self._model is None:
            raise RuntimeError("Model must be loaded before running decode.")

        decode_queue_keys = list(self.decode_queue.keys())

        if not decode_queue_keys:
            PACE_INFO("No sequences in decode queue, returning empty result")
            return {}

        if self._spec_decoder is not None:
            return self._speculative_decode(decode_queue_keys)
        return self._standard_decode(
            decode_queue_keys,
            all_greedy=self._all_greedy,
        )

    def _standard_decode(self, decode_queue_keys, all_greedy):
        """Standard single-token decode loop. Returns only token_ids."""
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
            sampler_outputs: List[Optional[torch.Tensor]] = []
            if all_greedy:
                logits_last = model_out.logits[:, -1, :]
                PACE_ASSERT(
                    logits_last is not None and torch.isnan(logits_last).sum() == 0,
                    "NaN values found in logits during greedy decode step, "
                    "failing decode to prevent invalid token generation. "
                    "Please check the model and input data for issues.",
                )
                next_tokens_batch = logits_last.argmax(dim=-1)
                # Greedy decode skips the sampler, so compute logprobs directly
                # from the batch logits when any request asked for them.
                if any(
                    self.decode_queue[k].sampling_config.return_logprobs
                    for k in decode_queue_keys
                ):
                    greedy_logprobs = torch.log_softmax(
                        logits_last, dim=-1, dtype=torch.float32
                    )
                    sampler_outputs = [
                        greedy_logprobs[i] for i in range(len(decode_queue_keys))
                    ]
            else:
                next_token_logits = model_out.logits[:, -1, :].clone().float()
                next_tokens_batch = torch.empty(
                    len(decode_queue_keys), dtype=torch.long
                )
                for i, req_id in enumerate(decode_queue_keys):
                    req = self.decode_queue[req_id]
                    sampler_output = req.sampler.sample(
                        req.input_encoded.input_ids, next_token_logits[i : i + 1]
                    )
                    next_tokens_batch[i] = sampler_output.next_tokens[0, 0]
                    sampler_outputs.append(sampler_output.logprobs)

            completed_ids = []
            for i, req_id in enumerate(decode_queue_keys):
                req = self.decode_queue[req_id]
                token_id_int = int(next_tokens_batch[i].item())
                req.append_token(token_id_int)
                req.num_computed_tokens += 1

                logprobs_obj = None
                if req.sampling_config.return_logprobs:
                    logprobs_obj = self._build_logprobs_object(
                        [token_id_int],
                        sampler_outputs[i] if sampler_outputs else None,
                        req.logprobs_n,
                    )

                if req.stopping_criteria.stop_now(req.input_encoded.input_ids).item():
                    PACE_INFO(f"Stopping criteria met for sequence {req.id}.")
                    entry: Dict[str, Any] = {
                        "token_ids": [token_id_int],
                        "status": "COMPLETED",
                        "num_tokens_generated": 1,
                        "stop_reason": req.stopping_criteria.stop_reason,
                    }
                    if logprobs_obj is not None:
                        entry["logprobs"] = logprobs_obj
                    result_buffer[str(req.id)] = entry
                    completed_ids.append(req_id)
                    continue
                req.kv_cache_manager = batch_kv_cache_managers_list[i]

                entry = {
                    "token_ids": [token_id_int],
                    "status": "DECODING_IN_PROGRESS",
                    "num_tokens_generated": 1,
                }
                if logprobs_obj is not None:
                    entry["logprobs"] = logprobs_obj
                result_buffer[str(req.id)] = entry

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
        """Speculative decode with batched draft and target forwards.

        Reuses ``_run_draft_phase`` for drafting and
        ``_verify_spec_decode_results`` for verification.  The target
        forward here is decode-only (uniform K+1 query length).
        """
        batch_size = len(decode_queue_keys)
        PACE_DEBUG(f"Speculative decode batch size: {batch_size}")
        result_buffer: Dict[str, Dict[str, Any]] = {}

        try:
            reqs = [self.decode_queue[rid] for rid in decode_queue_keys]

            # Phase 1: Draft (shared implementation)
            decode_token_map, speculated_inputs_list = self._run_draft_phase(reqs)

            # Phase 2: Batched target-model forward (decode-only, uniform K+1)
            target_inputs = []
            target_positions_list = []
            target_kv_caches = []

            for req in reqs:
                extended = decode_token_map[str(req.id)]
                target_inputs.append(extended)
                pos = req.compute_positions(extended.shape[-1])
                target_positions_list.append(pos)
                target_kv_caches.append(req.kv_cache_manager)

            batched_target_input = self.batch_input_ids(target_inputs)
            if batch_size == 1:
                batched_target_pos = target_positions_list[0]
            else:
                batched_target_pos = torch.cat(target_positions_list, dim=0)

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
                merged_target,
                **target_pa,
            )

            # Update target position counters
            for i, req in enumerate(reqs):
                req.num_computed_tokens += target_inputs[i].shape[-1]

            # Extract full logits per sequence (uniform batch: direct indexing)
            decode_full_logits = [
                target_out.logits[i : i + 1] for i in range(batch_size)
            ]

            # Phase 3: Verify (shared implementation)
            self._verify_spec_decode_results(
                reqs, decode_full_logits, speculated_inputs_list, result_buffer
            )

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

    def _run_draft_phase(
        self,
        decode_sequences: List[Sequence],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Run per-sequence draft model forwards for speculative decoding.

        Returns:
            decode_token_map: ``{req_id_str: [last_accepted, spec0..specK-1]}``
                tensors for ``build_mixed_batch``.
            speculated_inputs_list: per-sequence ``[1, K+1]`` tensors needed
                for the verify phase.
        """
        sd = self._spec_decoder
        draft_size = sd.num_speculative_tokens
        decode_token_map: Dict[str, torch.Tensor] = {}
        speculated_inputs_list: List[torch.Tensor] = []

        for req in decode_sequences:
            nc = req.draft_num_computed_tokens
            n_missing = req.input_encoded.input_ids.shape[-1] - int(nc[0].item())
            if n_missing <= 0:
                n_missing = 1
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
                {"paged_attn_metadata": draft_meta} if draft_meta is not None else {}
            )
            draft_out = sd.draft_step(
                pard_input, pos, req.draft_kv_cache_manager, **draft_pa
            )
            req.draft_num_computed_tokens += pard_input.shape[-1]
            spec_toks = draft_out.logits[:, -draft_size:].argmax(dim=-1)

            last_token = req.input_encoded.input_ids[:, -1:]
            extended = torch.cat([last_token, spec_toks], dim=-1)
            speculated_inputs_list.append(extended)
            decode_token_map[str(req.id)] = extended

        return decode_token_map, speculated_inputs_list

    def _verify_spec_decode_results(
        self,
        decode_sequences: List[Sequence],
        decode_full_logits: List[torch.Tensor],
        speculated_inputs_list: List[torch.Tensor],
        decode_results: Dict[str, Any],
    ) -> None:
        """Run per-sequence verify against target logits for speculative decode.

        Args:
            decode_sequences: Decode sequences that participated in the forward.
            decode_full_logits: List of ``[1, K+1, vocab_size]`` logit tensors,
                one per decode sequence (all K+1 positions from the packed forward).
            speculated_inputs_list: Per-sequence ``[1, K+1]`` token tensors
                (last accepted + K draft tokens) from the draft phase.
            decode_results: Dict to populate with verification outcomes.
        """
        sd = self._spec_decoder
        draft_size = sd.num_speculative_tokens
        completed_ids = []

        for i, req in enumerate(decode_sequences):
            logits_i = decode_full_logits[i]
            if logits_i.dim() == 2:
                logits_i = logits_i.unsqueeze(0)
            logits_i = logits_i[:, -draft_size - 1 :, :].clone().float()
            PACE_ASSERT(
                torch.isnan(logits_i).sum() == 0,
                "NaN values found in logits during speculative decode step.",
            )

            # Reshape [1, K+1, vocab] -> [K+1, vocab] so the sampler
            # processes each candidate position as a batch element, avoiding
            # the 3D scatter bug in top_p/top_k preprocessors.
            n_cand = logits_i.shape[1]
            logits_2d = logits_i.squeeze(0).reshape(n_cand, -1)
            input_rep = req.input_encoded.input_ids.repeat(n_cand, 1)
            sampler_output: SamplerOutput = req.sampler.sample(input_rep, logits_2d)
            # Sampler returns [K+1, 1]; verify expects [1, K+1]
            next_tokens = sampler_output.next_tokens.reshape(1, -1)

            result = sd.verify(
                next_tokens,
                speculated_inputs_list[i],
                draft_kv_cache=req.draft_kv_cache_manager,
                draft_num_computed=req.draft_num_computed_tokens,
            )
            accepted = result.accepted_tokens

            if result.target_kv_cache_trim > 0:
                req.kv_cache_manager.remove_cache(result.target_kv_cache_trim)
                req.num_computed_tokens -= result.target_kv_cache_trim

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

            accepted_ids = (
                accepted[0, :num_accepted].tolist() if num_accepted > 0 else []
            )

            logprobs_obj = None
            if (
                req.sampling_config.return_logprobs
                and sampler_output.logprobs is not None
            ):
                accepted_logprobs = sampler_output.logprobs
                if accepted_logprobs.dim() == 3:
                    accepted_logprobs = accepted_logprobs.squeeze(0)
                accepted_logprobs = accepted_logprobs[:num_accepted]
                logprobs_obj = self._build_logprobs_object(
                    accepted_ids,
                    accepted_logprobs,
                    req.logprobs_n,
                )

            if req.stopping_criteria.stop_now(
                req.input_encoded.input_ids, num_new_tokens=num_accepted
            ).item():
                PACE_INFO(f"Stopping criteria met for sequence {req.id}.")
                entry: Dict[str, Any] = {
                    "token_ids": accepted_ids,
                    "status": "COMPLETED",
                    "num_tokens_generated": num_accepted,
                    "stop_reason": req.stopping_criteria.stop_reason,
                }
                if logprobs_obj is not None:
                    entry["logprobs"] = logprobs_obj
                decode_results[str(req.id)] = entry
                completed_ids.append(req.id)
                continue

            PACE_DEBUG(
                f"Speculative decode for {req.id}: accepted {num_accepted} tokens"
            )
            entry = {
                "token_ids": accepted_ids,
                "status": "DECODING_IN_PROGRESS",
                "num_tokens_generated": num_accepted,
            }
            if logprobs_obj is not None:
                entry["logprobs"] = logprobs_obj
            decode_results[str(req.id)] = entry

        if completed_ids:
            self.remove_sequences(completed_ids)

    def _standard_decode_results(
        self,
        decode_sequences: List[Sequence],
        logits_list: List[torch.Tensor],
        decode_results: Dict[str, Any],
    ) -> None:
        """Process single-token decode results (no speculative decoding).

        Emits per-request sample ``logprobs`` (numeric IDs + values) when the
        request set ``return_logprobs``, matching the iterative ``_standard_decode``
        contract so logprobs behave identically under continuous batching.
        """
        all_greedy = self._all_greedy or all(
            s.sampler._greedy_fast_path for s in decode_sequences
        )
        want_logprobs = any(s.sampling_config.return_logprobs for s in decode_sequences)

        # Per-request log-prob rows ([1, vocab]) used to build logprobs objects.
        logprob_rows: List[Optional[torch.Tensor]] = [None] * len(decode_sequences)

        if all_greedy:
            stacked = torch.stack(logits_list).float()
            batch_tokens = torch.argmax(stacked, dim=-1).squeeze(-1)
            # Greedy skips the sampler, so derive logprobs from the batch logits.
            if want_logprobs:
                greedy_lp = torch.log_softmax(stacked, dim=-1, dtype=torch.float32)
                for j in range(len(decode_sequences)):
                    logprob_rows[j] = greedy_lp[j]
        else:
            batch_tokens = torch.empty(len(decode_sequences), dtype=torch.long)
            for j, req in enumerate(decode_sequences):
                out = req.sampler.sample(
                    req.input_encoded.input_ids,
                    (
                        logits_list[j].float().unsqueeze(0)
                        if logits_list[j].dim() == 1
                        else logits_list[j].float()
                    ),
                )
                t = out.next_tokens
                batch_tokens[j] = t[0, 0] if t.dim() == 2 else t[0]
                if req.sampling_config.return_logprobs:
                    logprob_rows[j] = out.logprobs

        completed_ids = []
        for j, req in enumerate(decode_sequences):
            tok_val = int(batch_tokens[j].item())
            req.append_token(tok_val)

            logprobs_obj = None
            if req.sampling_config.return_logprobs:
                logprobs_obj = self._build_logprobs_object(
                    [tok_val], logprob_rows[j], req.logprobs_n
                )

            is_done = req.stopping_criteria.stop_now(req.input_encoded.input_ids).item()

            if is_done:
                entry: Dict[str, Any] = {
                    "token_ids": [tok_val],
                    "status": "COMPLETED",
                    "stop_reason": req.stopping_criteria.stop_reason,
                }
                if logprobs_obj is not None:
                    entry["logprobs"] = logprobs_obj
                decode_results[str(req.id)] = entry
                completed_ids.append(req.id)
            else:
                entry = {
                    "token_ids": [tok_val],
                    "status": "DECODING_IN_PROGRESS",
                }
                if logprobs_obj is not None:
                    entry["logprobs"] = logprobs_obj
                decode_results[str(req.id)] = entry

        if completed_ids:
            self.remove_sequences(completed_ids)

    def prefill_decode_mixed_batch(
        self,
        prefill_sequences: List[Sequence],
        decode_sequences: List[Sequence],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run mixed prefill+decode batch using unified paged attention.

        Only supported for non-contiguous cache backends (PAGED, SLAB_POOL).
        Returns only token IDs -- all detokenization is done at the router.

        When speculative decoding is active, decode sequences contribute
        K+1 tokens each (last accepted + K speculated) into the same unified
        forward as prefills, using variable query lengths via
        ``merge_contexts_mixed``.  Post-forward, decode logits are verified
        against draft predictions and prefill logits are sampled normally.

        Args:
            prefill_sequences: New requests needing prefill.
            decode_sequences: Existing requests in decode phase.
            chunk_info_map: Optional ``{req_id_str: (chunk_start, chunk_size)}``
                for chunked prefill.

        Returns:
            ``(prefill_results, decode_results)`` dicts keyed by request ID str.
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before running mixed batch.")

        prefill_results: Dict[str, Any] = {}
        decode_results: Dict[str, Any] = {}
        if not prefill_sequences and not decode_sequences:
            return prefill_results, decode_results

        try:
            if not hasattr(self, "_unified_builder") or self._unified_builder is None:
                self._unified_builder = UnifiedBatchBuilder(
                    self._cache_backend, self.model_config
                )
                PACE_INFO(
                    f"[MIXED_BATCH] UnifiedBatchBuilder initialized "
                    f"(cache_type={self.kv_cache_type})"
                )

            # Prefix-cache reuse: claim cached blocks for first-chunk prefills
            # and advance their chunk start past the reused prefix so the batch
            # builder computes only the uncached suffix.
            if prefill_sequences and getattr(
                self._cache_backend, "prefix_cache_enabled", False
            ):
                chunk_info_map = self._apply_mixed_prefix_reuse(
                    prefill_sequences, chunk_info_map
                )

            decode_only = not prefill_sequences
            spec_active = self._spec_decoder is not None
            n_dec = len(decode_sequences)
            decode_token_map: Optional[Dict[str, torch.Tensor]] = None
            speculated_inputs_list: Optional[List[torch.Tensor]] = None

            # --- Phase 1: Draft (when spec decode is active) ---
            if spec_active and decode_sequences:
                decode_token_map, speculated_inputs_list = self._run_draft_phase(
                    decode_sequences
                )

            # --- Build unified batch ---
            if decode_only and not spec_active:
                batch = self._unified_builder.build_decode_only_fast(decode_sequences)
            else:
                batch = self._unified_builder.build_mixed_batch(
                    prefill_sequences,
                    decode_sequences,
                    chunk_info_map,
                    decode_token_map=decode_token_map,
                )

            PACE_DEBUG(
                f"[MIXED_BATCH] prefills={len(prefill_sequences)} "
                f"decodes={n_dec} "
                f"total_tokens={batch.total_tokens} "
                f"query_lens={batch.query_lens} "
                f"decode_only={decode_only} spec_active={spec_active}"
            )

            # --- Phase 2: Unified target model forward ---
            self._ensure_pool_headroom(
                batch, len(prefill_sequences), len(decode_sequences)
            )
            step_kwargs = {}
            if batch.paged_attn_metadata is not None:
                step_kwargs["paged_attn_metadata"] = batch.paged_attn_metadata

            model_out = self._model.step(
                batch.packed_input_ids,
                batch.packed_positions,
                batch.merged_kv_cache,
                **step_kwargs,
            )

            # --- Extract per-request logits ---
            if spec_active and decode_sequences:
                full_logits = batch.extract_full_logits_per_request(model_out.logits)
                decode_full_logits = full_logits[:n_dec]
                per_request_logits = batch.extract_logits_per_request(model_out.logits)
            else:
                decode_full_logits = None
                per_request_logits = batch.extract_logits_per_request(model_out.logits)

        except Exception as e:
            PACE_ERROR(
                f"prefill_decode_mixed_batch forward failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            for req in prefill_sequences:
                prefill_results[str(req.id)] = {
                    "status": "ERROR",
                    "error": str(e),
                }
                self.remove_sequences([req.id])
            for req in decode_sequences:
                decode_results[str(req.id)] = {
                    "status": "ERROR",
                    "error": str(e),
                }
                self.remove_sequences([req.id])
            return prefill_results, decode_results

        # --- Post-forward: per-request processing (isolated failures) ---

        # Update num_computed_tokens for decode seqs
        for i, req in enumerate(decode_sequences):
            req.num_computed_tokens += batch.query_lens[i]

        # Full per-request prefill logits are only needed for prompt (echo)
        # logprobs. Extract them once when at least one prefill asked for them.
        full_per_request_logits: Optional[List[torch.Tensor]] = None
        if any(
            r.sampling_config.return_logprobs
            and r.sampling_config.return_input_logprobs
            for r in prefill_sequences
        ):
            full_per_request_logits = batch.extract_full_logits_per_request(
                model_out.logits
            )

        # Process prefill results (per-request isolation)
        for i, req in enumerate(prefill_sequences):
            try:
                logits_idx = n_dec + i
                req_id_str = str(req.id)

                ci = chunk_info_map.get(req_id_str) if chunk_info_map else None
                if ci is not None:
                    chunk_start, chunk_size = ci
                    total_prompt_len = req.input_encoded.input_ids.shape[-1]
                    actual_chunk_len = min(chunk_size, total_prompt_len - chunk_start)
                    is_final_chunk = chunk_start + actual_chunk_len >= total_prompt_len
                    if not is_final_chunk:
                        req.num_computed_tokens = chunk_start + actual_chunk_len
                        self._incremental_prefix_insert(req)
                        prefill_results[req_id_str] = {
                            "status": "PREFILL_IN_PROGRESS",
                            "num_computed_tokens": req.num_computed_tokens,
                            "actual_input_length": total_prompt_len,
                        }
                        continue

                next_token_logits = per_request_logits[logits_idx].float()
                # Prompt logprobs need the full prompt's logits in one step;
                # only pass them when this request's query covered the whole
                # prompt (i.e. non-chunked prefill), else echo logprobs are
                # skipped to avoid partial/misaligned results.
                full_logits_i = None
                if full_per_request_logits is not None:
                    prompt_len = req.input_encoded.input_ids.shape[-1]
                    candidate = full_per_request_logits[logits_idx]
                    if candidate.shape[1] == prompt_len:
                        full_logits_i = candidate.float()
                self._apply_prefill_result(
                    req, next_token_logits, prefill_results, full_logits_i
                )
            except Exception as e:
                PACE_ERROR(
                    f"Prefill post-processing failed for {req.id}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                prefill_results[str(req.id)] = {
                    "status": "ERROR",
                    "error": str(e),
                }
                self.remove_sequences([req.id])

        # Process decode results
        if decode_sequences and spec_active:
            self._verify_spec_decode_results(
                decode_sequences,
                decode_full_logits,
                speculated_inputs_list,
                decode_results,
            )
        elif decode_sequences:
            self._standard_decode_results(
                decode_sequences,
                [per_request_logits[j] for j in range(n_dec)],
                decode_results,
            )

        return prefill_results, decode_results

    def _apply_mixed_prefix_reuse(
        self,
        prefill_sequences: List[Sequence],
        chunk_info_map: Optional[Dict[str, Tuple[int, int]]],
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Claim cached prefix blocks for new prefills and bump their chunk start.

        Only first-chunk prefills (``chunk_start == 0``) qualify: reuse replaces
        the prompt head and can only be claimed before any prompt tokens are
        computed. For an ``N``-token cached prefix, matched blocks are assigned
        zero-copy and the chunk start advances to ``N`` so the batch builder
        computes only the uncached suffix ``[N, prompt_len)``. Returns the
        (possibly updated) chunk-info map.
        """
        local: Dict[str, Tuple[int, int]] = (
            dict(chunk_info_map) if chunk_info_map else {}
        )
        dedup = len(prefill_sequences) > 1
        changed = False
        # Register in-flight (chunked) prefills' blocks before new prefills look
        # up the cache, so a same-step follower reuses the leader's full prefix.
        if dedup:
            for req in prefill_sequences:
                ci = local.get(str(req.id))
                if ci and ci[0] != 0:
                    self._preregister_batch_prefix(req, ci)
        for req in prefill_sequences:
            rid = str(req.id)
            ci = local.get(rid)
            chunk_start = ci[0] if ci else 0
            if chunk_start != 0 or getattr(req, "_prefix_claimed", False):
                continue
            req._prefix_claimed = True
            hit = self._match_and_claim_prefix(req)
            if hit > 0:
                total_prompt_len = req.input_encoded.input_ids.shape[-1]
                chunk_size = ci[1] if ci else (total_prompt_len - hit)
                local[rid] = (hit, chunk_size)
                req.num_computed_tokens = hit
                changed = True
            if dedup:
                self._preregister_batch_prefix(req, local.get(rid) or ci)
        if changed:
            return local
        return chunk_info_map

    def _ensure_pool_headroom(self, batch, n_prefill: int, n_decode: int) -> None:
        backend = self._cache_backend
        mgr = getattr(backend, "_manager", None)
        if mgr is None or not getattr(backend, "prefix_cache_enabled", False):
            return
        need = -(-batch.total_tokens // mgr.block_size) + n_prefill + n_decode
        deficit = need - mgr.get_free_blocks()
        if deficit > 0:
            backend.evict_prefix_blocks(deficit)

    def _preregister_batch_prefix(
        self, req: Sequence, ci: Optional[Tuple[int, int]]
    ) -> None:
        backend = self._cache_backend
        total_prompt_len = req.input_encoded.input_ids.shape[-1]
        chunk_start, chunk_size = ci if ci else (0, total_prompt_len)
        computed_end = min(chunk_start + chunk_size, total_prompt_len)
        num_full = computed_end // backend._prefix_block_size
        if num_full <= 0:
            return
        token_list = req.input_encoded.input_ids[0].tolist()
        backend.preinsert_computed_blocks(req.kv_cache_manager, token_list, num_full)

    def _match_and_claim_prefix(self, req: Sequence) -> int:
        """Match and zero-copy claim a cached prefix for a prefill request.

        Shared by the single-request ``prefill()`` and the continuous-batch
        path so both enforce the same reuse semantics. Returns the number of
        reused (block-aligned) prefix tokens, or 0; at least one block is always
        left uncached so the forward still produces logits.
        """
        backend = self._cache_backend
        if not getattr(backend, "prefix_cache_enabled", False):
            return 0
        input_ids = req.input_encoded.input_ids
        full_seq_len = input_ids.shape[-1]
        token_list = input_ids[0].tolist()
        matched_blocks, num_matched = backend.try_prefix_match(token_list)
        if num_matched <= 0:
            return 0

        block_size = backend._prefix_block_size
        if num_matched >= full_seq_len:
            num_matched -= block_size
        if num_matched <= 0:
            return 0
        num_blocks = num_matched // block_size
        matched_blocks = matched_blocks[:num_blocks]
        if not matched_blocks:
            return 0

        backend.claim_prefix_blocks(
            req.kv_cache_manager, token_list, matched_blocks, num_matched
        )
        PACE_INFO(
            f"[PREFIX] Request {req.id}: reusing {len(matched_blocks)} blocks "
            f"({num_matched} tokens), computing suffix of "
            f"{full_seq_len - num_matched} tokens"
        )
        return num_matched

    def _incremental_prefix_insert(self, req: Sequence) -> None:
        """Register computed prefix blocks in the cache after a prefill step.

        Inserts one parent-chained hash per *full* block over tokens
        ``[0, req.num_computed_tokens)`` (``hash(block_i)`` folds in
        ``hash(block_{i-1})`` plus block ``i``'s token IDs; trailing partial
        block skipped). Dedup prevents re-registering prior-chunk blocks. Both
        paged and slab reuse these blocks for later requests. Populates the
        cache on the continuous-batch path.
        """
        backend = self._cache_backend
        if backend is None or not getattr(backend, "prefix_cache_enabled", False):
            return
        token_list = req.input_encoded.input_ids[0, : req.num_computed_tokens].tolist()
        if not token_list:
            return
        inserted = backend.insert_prefix(req.kv_cache_manager, token_list)
        if inserted > 0:
            PACE_DEBUG(
                f"[PREFIX] Incremental insert: {inserted} new blocks after "
                f"{req.num_computed_tokens} computed tokens for request {req.id}"
            )

    def _apply_prefill_result(
        self,
        req: Sequence,
        next_token_logits: torch.Tensor,
        result_buffer: Dict[str, Any],
        full_prefill_logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Sample from prefill logits, append token, move to decode or mark complete.

        When speculative decoding is active, also warms up the draft KV cache
        so the sequence is ready for spec decode steps.

        Emits the first sample token's ``logprobs`` (when ``return_logprobs``)
        and, when ``full_prefill_logits`` covering the whole prompt is provided
        and ``return_input_logprobs`` is set, the echo ``prompt_logprobs`` —
        mirroring the iterative ``prefill`` contract.
        """
        prompt_len = req.input_encoded.input_ids.shape[-1]
        req.num_computed_tokens = prompt_len
        self._incremental_prefix_insert(req)

        prompt_ids_snapshot = req.input_encoded.input_ids.squeeze(0).tolist()

        # Warm up draft KV cache BEFORE appending the sampled token,
        # matching master's prefill() ordering.  draft_num_computed must
        # equal the original prompt length so the first decode step sees
        # n_missing = 1 (the newly appended token).
        if self._spec_decoder is not None:
            PACE_ASSERT(
                req.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH,
                "Speculative decoding requires greedy search sampling mode "
                f"but got {req.sampling_config.sampling_mode}",
            )
            max_seq_length = req.sampling_config.max_new_tokens + prompt_len
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
            positions = req.compute_positions(prompt_len)
            draft_meta = self._cache_backend.build_prefill_metadata(
                req.draft_kv_cache_manager,
                prompt_len,
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
            req.draft_num_computed_tokens += prompt_len

        sampler_output: SamplerOutput = req.sampler.sample(
            req.input_encoded.input_ids, next_token_logits
        )
        output_token_id = sampler_output.next_tokens
        req.append_token(output_token_id.item())

        token_id_int = output_token_id[0].item()
        req_id_str = str(req.id)

        logprobs_obj = None
        if req.sampling_config.return_logprobs:
            logprobs_obj = self._build_logprobs_object(
                [token_id_int], sampler_output.logprobs, req.logprobs_n
            )

        prompt_logprobs_obj = None
        if (
            req.sampling_config.return_logprobs
            and req.sampling_config.return_input_logprobs
            and full_prefill_logits is not None
        ):
            prompt_logprobs_obj = self._build_prompt_logprobs_object(
                prompt_ids_snapshot, full_prefill_logits, req.logprobs_n
            )

        if req.stopping_criteria.stop_now(req.input_encoded.input_ids).item():
            PACE_INFO(f"Sequence {req.id} completed during prefill.")
            self.remove_sequences([req.id])
            entry: Dict[str, Any] = {
                "token_ids": [token_id_int],
                "status": "COMPLETED",
                "stop_reason": req.stopping_criteria.stop_reason,
            }
            if logprobs_obj is not None:
                entry["logprobs"] = logprobs_obj
            if prompt_logprobs_obj is not None:
                entry["prompt_logprobs"] = prompt_logprobs_obj
            result_buffer[req_id_str] = entry
            return

        self.move_to_decode_queue(req)
        entry = {
            "token_ids": [token_id_int],
            "status": "PREFILL_COMPLETED",
        }
        if logprobs_obj is not None:
            entry["logprobs"] = logprobs_obj
        if prompt_logprobs_obj is not None:
            entry["prompt_logprobs"] = prompt_logprobs_obj
        result_buffer[req_id_str] = entry

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

        if not self._all_greedy:
            self._all_greedy = (
                all(r.sampler._greedy_fast_path for r in self.decode_queue.values())
                if self.decode_queue
                else True
            )


model_executor = ModelExecutor()
