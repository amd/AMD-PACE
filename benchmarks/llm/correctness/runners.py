# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import gc
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    OperatorConfig,
    LLMOperatorType,
    LLMBackendType,
    PardSpecDecodeConfig,
)
from pace.llm.attention import AttentionBackendType
from pace.utils.logging import suppress_logging_fn


def _convert_operators(raw: Dict[str, str]) -> Dict:
    """Convert string operator config (from JSON) to typed enums."""
    converted = {}
    for key, value in raw.items():
        op_key = LLMOperatorType(key.lower())
        if op_key == LLMOperatorType.Attention:
            converted[op_key] = AttentionBackendType(value.lower())
        else:
            converted[op_key] = LLMBackendType(value.lower())
    return converted


TokensTextLogprobs = Tuple[List[int], str, Optional[List[Dict[int, float]]]]


def setup_tokenizer(tokenizer):
    """Configure tokenizer for left-padded batch generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def extract_topk_logprobs(
    per_step_logprobs: torch.Tensor, num_logprobs: int
) -> List[Dict[int, float]]:
    """Extract top-k logprobs per step from a [num_steps, vocab_size] tensor.

    Returns a list of dicts mapping token_id -> logprob for each step.
    """
    results = []
    for step_idx in range(per_step_logprobs.shape[0]):
        topk_vals, topk_ids = torch.topk(per_step_logprobs[step_idx], k=num_logprobs)
        step_dict = {tid.item(): tv.item() for tid, tv in zip(topk_ids, topk_vals)}
        results.append(step_dict)
    return results


def trim_at_eos(
    token_ids: List[int],
    logprobs_list: List[Dict[int, float]],
    eos_token_ids: List[int],
    pad_token_id: int,
) -> Tuple[List[int], List[Dict[int, float]]]:
    """Trim a single sequence's tokens and logprobs at the first EOS or pad token.

    EOS token is included; pad tokens are excluded.
    """
    for i, tok in enumerate(token_ids):
        if tok in eos_token_ids:
            return token_ids[: i + 1], logprobs_list[: i + 1]
        if tok == pad_token_id:
            return token_ids[:i], logprobs_list[:i]
    return token_ids, logprobs_list


class HfRunner:
    """HuggingFace Transformers reference runner for correctness comparison."""

    def __init__(self, model_id: str, dtype: torch.dtype = torch.bfloat16):
        """Initialize the HfRunner.

        Args:
            model_id: HuggingFace model identifier or local path.
            dtype: Data type for model weights.
        """
        self.model_id = model_id
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        """Load the model and tokenizer, configure for greedy generation."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        setup_tokenizer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        )
        self.model.eval()
        return self

    def __exit__(self, *args):
        """Release model and tokenizer memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        batch_size: int = 1,
    ) -> List[TokensTextLogprobs]:
        """Generate with greedy decoding and return per-prompt (tokens, text, logprobs)."""
        all_results: List[TokensTextLogprobs] = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch_prompts, return_tensors="pt", padding="longest"
            )
            prompt_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output = self.model.generate(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            # output.logits: tuple of max_steps tensors, each [batch, vocab_size]
            hf_logits = torch.stack(output.logits, dim=1)
            hf_logprobs = torch.log_softmax(hf_logits.float(), dim=-1)

            eos_ids = self.tokenizer.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]
            pad_id = self.tokenizer.pad_token_id

            for seq_idx in range(len(batch_prompts)):
                gen_ids = output.sequences[seq_idx, prompt_len:].tolist()
                # [steps, vocab]
                seq_logprobs = hf_logprobs[seq_idx]

                topk = extract_topk_logprobs(seq_logprobs, num_logprobs)
                gen_ids, topk = trim_at_eos(gen_ids, topk, eos_ids, pad_id)

                decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_results.append((gen_ids, decoded, topk))

        return all_results


class PaceRunner:
    """PACE LLM runner for correctness comparison."""

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
        llm_operators: Optional[Dict[str, str]] = None,
        kv_cache_type: str = "DYNAMIC",
        spec_config: Optional[Dict] = None,
    ):
        """Initialize the PaceRunner.

        Args:
            model_id: HuggingFace model identifier or local path.
            dtype: Data type for model weights.
            llm_operators: Operator-to-backend mapping (e.g. {"Attention": "JIT"}).
                           Defaults to all NATIVE if not provided.
            kv_cache_type: KV cache type, "DYNAMIC", "BMC" or "SLAB_POOL".
            spec_config: Speculative decoding config dict with keys
                         "model_name" and "num_speculated_tokens", or None.
        """
        self.model_id = model_id
        self.dtype = dtype
        self.llm_operators = llm_operators or {}
        self.kv_cache_type = kv_cache_type
        self.spec_config = spec_config
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        """Load the PACE model and tokenizer, configure for greedy generation."""
        cache_type_str = self.kv_cache_type.upper()
        if cache_type_str == "BMC":
            kv_cache = KVCacheType.BMC
        elif cache_type_str == "SLAB_POOL":
            kv_cache = KVCacheType.SLAB_POOL
        elif cache_type_str == "PAGED":
            kv_cache = KVCacheType.PAGED
        else:
            kv_cache = KVCacheType.DYNAMIC

        spec = None
        if self.spec_config is not None:
            spec = PardSpecDecodeConfig(
                model_name_or_path=self.spec_config["model_name"],
                num_speculative_tokens=self.spec_config["num_speculated_tokens"],
            )

        opconfig = (
            OperatorConfig(**_convert_operators(self.llm_operators))
            if self.llm_operators
            else None
        )

        self.model = LLMModel(
            self.model_id,
            dtype=self.dtype,
            kv_cache_type=kv_cache,
            opconfig=opconfig,
            spec_config=spec,
            disable_tqdm=True,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.model_config = self.model.get_config()
        setup_tokenizer(self.tokenizer)
        return self

    def __exit__(self, *args):
        """Release model and tokenizer memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()

    @suppress_logging_fn
    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        batch_size: int = 1,
    ) -> List[TokensTextLogprobs]:
        """Generate with greedy decoding and return per-prompt (tokens, text, logprobs)."""
        all_results: List[TokensTextLogprobs] = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch_prompts, return_tensors="pt", padding="longest"
            )
            prompt_len = encoded["input_ids"].shape[1]

            sampling_config = SamplingConfig(
                max_new_tokens=max_tokens,
                temperature=0,
                return_logprobs=True,
            )
            output = self.model.generate(encoded, sampling_config)

            vocab_size = self.model_config.vocab_size
            bs = len(batch_prompts)

            # Reshape flat logprobs: [batch, vocab*steps] -> [batch, steps, vocab]
            # [batch, vocab_size * num_steps]
            raw_logprobs = output.logprobs
            per_step = raw_logprobs.view(bs, -1, vocab_size)

            eos_ids = self.tokenizer.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]
            pad_id = self.tokenizer.pad_token_id

            for seq_idx in range(bs):
                gen_ids = output.output_token_ids[seq_idx, prompt_len:].tolist()
                # [steps, vocab]
                seq_logprobs = per_step[seq_idx]

                topk = extract_topk_logprobs(seq_logprobs, num_logprobs)
                gen_ids, topk = trim_at_eos(gen_ids, topk, eos_ids, pad_id)

                # Spec decode can overshoot max_tokens; truncate to match HF
                if len(gen_ids) > max_tokens:
                    gen_ids = gen_ids[:max_tokens]
                    topk = topk[:max_tokens]

                decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_results.append((gen_ids, decoded, topk))

        return all_results
