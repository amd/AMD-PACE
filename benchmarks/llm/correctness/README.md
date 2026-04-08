# AMD PACE LLM Correctness Testing

Validates that PACE produces the same tokens as HuggingFace Transformers under greedy decoding. For each model, the test generates tokens from both PACE and HF on a fixed set of prompts, then compares output tokens and top-k logprobs. Inspired by [vLLM's model correctness testing](https://docs.vllm.ai/en/latest/contributing/model/tests.html).

## Usage

```bash
# Standard correctness test (PACE vs HuggingFace)
python test_correctness.py --config correctness_config.json

# Speculative decoding correctness test (PACE+spec vs PACE)
python test_correctness.py --config correctness_spec_config.json
```

## Configuration

All settings are specified in a JSON config file:

```json
{
    "model_args": {
        "model_name": "facebook/opt-125m",
        "dtype": "bf16",
        "llm_operators": {
            "Norm": "NATIVE",
            "QKVProjection": "TPP",
            "Attention": "JIT",
            "OutProjection": "TPP",
            "MLP": "TPP",
            "LMHead": "TPP"
        },
        "spec_config": null
    },
    "generation_args": {
        "kv_cache_type": "BMC",
        "max_tokens": 32,
        "batch_size": [1, 8],
        "num_logprobs": 5
    }
}
```

### Parameters

- `model_args.model_name`: HuggingFace model name, local path, or `"all"` to test every registered architecture.
- `model_args.dtype`: `"bf16"` or `"fp32"`.
- `model_args.llm_operators`: Operator-to-backend mapping. Valid backends: `NATIVE`, `TPP`, `JIT`, etc.
- `model_args.spec_config`: Speculative decoding config (`model_name` + `num_speculated_tokens`), or `null`.
- `generation_args.kv_cache_type`: `"BMC"` or `"DYNAMIC"`.
- `generation_args.max_tokens`: Number of tokens to generate per prompt.
- `generation_args.batch_size`: List of batch sizes to test (e.g. `[1, 8]`).
- `generation_args.num_logprobs`: Top-k logprobs for cross-checking on token mismatch.

## How It Works

**Standard mode** (`spec_config: null`): For each model and batch size:

1. Load HuggingFace model, generate greedy outputs with logprobs, then release.
2. Load PACE model with the specified backends/cache, generate the same, then release.
3. Compare: tokens must match exactly. If a token differs, both backends' chosen tokens must appear in each other's top-k logprobs.

**Spec decode mode** (`spec_config` provided): For each model and batch size:

1. Load PACE model **without** spec decode, generate greedy outputs, then release.
2. Load PACE model **with** spec decode, generate the same, then release.
3. Compare: tokens must match exactly (speculative decoding is mathematically equivalent to standard greedy under correct verification).

## Registered Models

When `model_name` is set to `"all"`, the following architectures are tested:

| Architecture | Default Model |
|---|---|
| OPTForCausalLM | facebook/opt-125m |
| GPTJForCausalLM | EleutherAI/gpt-j-6b |
| LlamaForCausalLM | NousResearch/Llama-3.2-1B |
| Qwen2ForCausalLM | Qwen/Qwen2.5-0.5B |
| Phi3ForCausalLM | microsoft/Phi-4-mini-instruct |
| Gemma3ForCausalLM | unsloth/gemma-3-270m |
| GptOssForCausalLM | openai/gpt-oss-20b |
