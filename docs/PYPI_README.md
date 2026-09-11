# AMD PACE

High-performance LLM inference and serving on AMD EPYC CPUs. PACE combines a PyTorch C++ extension, CPU-native KV-cache backends, fused operators, and a production-ready serving stack. Check out the [GitHub repository](https://github.com/amd/amd-pace) for more information.


## Highlights

- **Inference server** - `pace-server` provides a router/engine serving stack with a `continuous_batch` scheduler that combines chunked prefills and decodes, multi-instance NUMA-aware execution, and built-in Prometheus and scheduler metrics.
- **FusedMoE** - BF16 routed and shared Mixture-of-Experts execution with an optimized TPP expert backend.
- **Automatic prefix caching** - `pace-server` reuses complete prompt-prefix blocks across requests with `SLAB_POOL` and `PAGED`.
- **Broad model support** - Llama 3 and text-only Llama 4, Qwen2/2.5/3 (including Qwen3 MoE), Phi3/4, Gemma 3 and text-only Gemma 4, GPT-J, OPT, and GPT-OSS under one operator and backend framework.
- **Speculative decoding (PARD)** - built-in parallel-draft speculation for greedy decoding.
- **PACE Agent** - the repository also includes a LangGraph-based runtime for profiling, deterministic replay, and optimization of non-model agent work. See the [PACE Agent README](https://github.com/amd/amd-pace/tree/main/packages/pace_agent).

## Requirements

- Linux x86_64 with AVX512F + AVX512_BF16 (AMD Zen4 or newer)
- Python 3.10 – 3.14

## Install

```bash
# 1. CPU PyTorch (the +cpu build is not on PyPI; needs PyTorch's index).
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu

# 2. amd-pace
pip install amd-pace
```

## Quick example

Inference server (router + engine, OpenAI-compatible endpoint):

```bash
pace-server --server_model meta-llama/Llama-3.1-8B --kv_cache_type SLAB_POOL --serve_type continuous_batch
```

For offline programmatic generation, `pace.llm.LLMModel` resolves the tokenizer
and applies default operator backends unless they are explicitly overridden.
See the runnable scripts at
[`examples/`](https://github.com/amd/amd-pace/tree/main/examples) --
[`pace_llm_basic.py`](https://github.com/amd/amd-pace/blob/main/examples/pace_llm_basic.py)
is the smallest starting point.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of these, please kindly file an issue on the PACE GitHub page [here](https://github.com/amd/amd-pace/issues).

## License

AMD PACE is licensed under the MIT License. See the [LICENSE](https://github.com/amd/amd-pace/blob/main/LICENSE) file for details. Third-party notices are in [NOTICE](https://github.com/amd/amd-pace/blob/main/NOTICE).
