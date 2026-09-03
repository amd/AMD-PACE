# AMD PACE - AMD Platform Aware Compute Engine

AMD PACE is a high-performance LLM inference engine built from the ground up for AMD EPYC CPUs. It is a PyTorch C++ extension that combines custom AVX512 kernels, CPU-native KV cache management, and a production-ready serving stack to deliver maximum throughput on AMD server-class hardware.

🔥 **Check out our latest blog post: [AMD PACE - A vLLM Plugin for CPU Inference](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace-integrates-with-vllm.html)** — how PACE plugs into vLLM as a CPU platform plugin: a single `pip install pace-vllm`, no application code changes, routing vLLM's CPU worker through PACE's SLAB attention, tuned Linear/RMSNorm operators, and fused MLP on 5th Gen AMD EPYC™ processors.

📦 **Now available on PyPI** — `pip install amd-pace` (and `pip install pace-vllm` for the [vLLM plugin](packages/pace_vllm/README.md)). Pre-built **manylinux wheels** for Python 3.10 – 3.13; no compiler required.

> NOTE: AMD PACE is designed and tested for systems with AVX512 or higher support. On systems lacking AVX512, performance may degrade significantly due to fallback to slower reference implementations, or the library might not function as intended.

## Highlights

* **vLLM Plugin (`pace-vllm`)** — A drop-in [vLLM platform plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) that routes vLLM's CPU worker through PACE's kernels and SlabPool KV cache, with no changes to your existing vLLM scripts. Replaces vLLM's CPU attention backend, KV cache, and Linear/RMSNorm layers with PACE equivalents; under compile mode also fuses gated/ungated MLP blocks into a single `pace::libxsmm_fused_mlp` call. Lets you keep the vLLM API and get PACE's CPU performance underneath. More in [packages/pace_vllm/README.md](packages/pace_vllm/README.md).

* **SlabPool Attention** — A CPU-native KV cache and attention backend engineered for AMD EPYC processors. SlabPool manages all sequences in a single pre-allocated BF16 tensor with O(1) slab allocation, L2-aware block sizing, and a unified attention dispatcher that selects the optimal kernel path per sequence — GQA-aware decode with online softmax, multi-token decode, or tiled prefill — all within one OMP dispatch. Handles offline and online inference, continuous batching, sliding window, and sink attention through a single entry point. More in [SlabAttention.md](docs/SlabAttention.md).

* **Inference Server** — `pace-server` provides a full serving stack with a router/engine architecture, continuous batching, multi-instance NUMA-aware execution, and built-in metrics. The launcher automatically partitions CPU cores across engine instances and binds memory to the local NUMA node for optimal data locality. More in [InferenceServer.md](docs/InferenceServer.md).

* **Paged Attention** — An implementation of [vLLM-style paged KV cache](https://github.com/vllm-project/vllm) on CPU. Memory is allocated in fixed-size pages, eliminating fragmentation from variable-length sequences and enabling efficient memory sharing. Fully integrated with the PACE serving stack and all supported models.

* **Fused AVX512 Kernels** — PACE ships a suite of fused operators that eliminate intermediate memory traffic and keep data in registers/cache: fused Add+RMSNorm and Add+LayerNorm, fused RoPE, fused QKV projections, and a fused MLP kernel (via TPP/libXSMM). These are the default operators for all supported models.

* **Broad Model Support** — Llama (up to 3.3), Qwen2/2.5, Phi3/4, Gemma 3, GPT-J, OPT, and GPT-OSS, all running in BF16 with the same operator and backend framework. Adding a new architecture is a single-file effort. More in [LLM.md](docs/LLM.md).

* **Speculative Decoding (PARD)** — Built-in PARallel Draft Model Adaptation that runs a smaller draft model ahead of the target, verifying speculated tokens in a single parallel forward pass. PARD can deliver up to 5x throughput improvement over standard autoregressive decoding. More in [SpeculativeDecoding.md](docs/SpeculativeDecoding.md).

## Contents

* [Installation](#installation)
* [vLLM Plugin (`pace-vllm`)](packages/pace_vllm/README.md)
* [PACE Agent (`pace-agent`)](packages/pace_agent/README.md)
* [Inference Server](docs/InferenceServer.md)
* [More about AMD PACE](docs/Plugin.md)
* [Models Supported](#models-supported)
* [Examples](#examples)
* [Performance Guide](docs/PerformanceGuide.md)
* [Benchmarks](#benchmarks)
* [SlabPool Attention](docs/SlabAttention.md)
* [Contributing to AMD PACE](#contributing)
* [Tests](tests/README.md)
* [Known Limitations](#known-limitations)
* [External Dependencies](#external-dependencies)
* [Resources](#resources)

## Installation

AMD PACE provides pre-built **manylinux wheels** on PyPI — the simplest install path, no compiler needed. Building from source is supported for developers and bleeding-edge users.

### From PyPI (recommended)

1. **Create a Python 3.12 env with miniforge.** Install miniforge from [here](https://conda-forge.org/miniforge/), then:
   ```
   conda create -n pace-env-py3.12 python=3.12 -y
   conda activate pace-env-py3.12
   ```

   > AMD PACE is tested on Python 3.10 – 3.13; Python 3.12 is the most thoroughly exercised version.

2. **Install CPU PyTorch.** The `+cpu` build is not published on PyPI, so it needs PyTorch's index:
   ```
   pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu
   ```

3. **Install amd-pace:**
   ```
   pip install amd-pace
   ```

### From source (developers)

> NOTE: Building from source requires gcc>=12 and make. On Ubuntu: `sudo apt install build-essential gcc-12 g++-12`.

1. **Create an env** — see [From PyPI](#from-pypi-recommended) above.

2. **Install the required dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Build AMD PACE from source:**
    ```
    pip install -r build_requirements.txt [-v] .
    ```
    This will build AMD PACE and install it in the current environment. The `-v` option is optional and can be used to enable verbose output during the build process.

    > NOTE: It uses the new way of building packages with `pip`, for more details refer to [PEP 517](https://peps.python.org/pep-0517/). The `build_requirements.txt` should be passed in during installation to ensure that the build environment is set up correctly, please refer to [PEP 518](https://peps.python.org/pep-0518/) for more details.

    For developers who need to build AMD PACE frequently, using pip with `--no-build-isolation` is recommended to avoid unnecessary overhead of creating isolated environments for each build. This speeds up the build process significantly. Make sure to have all the required dependencies installed in your environment before using this option.
    ```
    pip install --no-build-isolation [-v] .
    ```

    > NOTE: Building AMD PACE, especially the oneDNN component, can require significant memory. If your system does not have enough RAM, the build process may fail or your machine may run out of memory.

## Models Supported
The following models are supported by AMD PACE:
1. [INT8 DLRMv2](docs/DLRMv2.md)
1. [Large Language Models](docs/LLM.md)
    1. [Speculative Decoding](docs/SpeculativeDecoding.md)
    1. [LLM Benchmarks](benchmarks/llm/performance/README.md)
    1. [LLM Evaluation](benchmarks/llm/accuracy/README.md)
    1. [MLPerf LLM Inference (Server scenario)](docs/MLPerf.md)

## Examples
The `examples/` directory contains runnable scripts and notebooks to get started with PACE:

* [PACE LLM Basic](examples/pace_llm_basic.py) - basic offline generation example
* [PACE LLM Streamer](examples/pace_llm_streamer.py) - streaming text generation example
* [PACE GPT-OSS Chat Notebook](examples/pace_gpt-oss_chat.ipynb) - offline GPT-OSS chat workflow with chat templating and final-answer extraction
* [PACE Sarvam Translate Quickstart](examples/pace_sarvam_translate_quickstart.ipynb) - offline translation notebook for Sarvam Translate
* [PACE Server Basic](examples/pace_server_basic.py) - minimal inference server example
* [Server Playbook](examples/playbook_server.ipynb) - end-to-end inference server notebook
* [Speculative Server Playbook](examples/playbook_server_speculative.ipynb) - speculative decoding server notebook

## Benchmarks
Benchmarks for AMD PACE are available in the `benchmarks` directory. The benchmarks include:
* [LLM Performance](benchmarks/llm/performance/README.md)
* [LLM Accuracy](benchmarks/llm/accuracy/README.md)
* [MLPerf LLM Inference (Server scenario)](docs/MLPerf.md)

## Verbose
To enable verbose mode, set the environment variable `PACE_LOG_LEVEL`. The following levels are supported:

| Level   |  Environment Variable           |
|---------|---------------------------------|
| Debug   | `export PACE_LOG_LEVEL=debug`   |
| Profile | `export PACE_LOG_LEVEL=profile` |
| Info    | `export PACE_LOG_LEVEL=info`    |
| Warning | `export PACE_LOG_LEVEL=warning` |
| Error   | `export PACE_LOG_LEVEL=error`   |
| None    | `export PACE_LOG_LEVEL=none`    |

NOTE: By default, the log level is set to `info`.

## Contributing
We welcome contributions to AMD PACE! Please see [docs/Contributing.md](docs/Contributing.md) for guidelines on adding operators, creating core functions, code style, testing, and submitting PRs.

## Known Limitations

The following feature combinations are not yet supported in this release and may not work as expected:

| Feature | Unsupported Configuration | Supported Alternative |
|---------|--------------------------|----------------------|
| GPT-OSS | JIT attention backend | SLAB_POOL cache with SLAB or PAGED attention backend |
| Qwen2 | JIT attention backend | SLAB_POOL cache with SLAB or PAGED attention backend |
| AMX-enabled CPUs | libXSMM kernels fail on the AMX instruction path | Set `LIBXSMM_TARGET=cpx` to force the AVX-512 (CPX) path and disable AMX |

## External Dependencies

### Build-time (C++ libraries, built from source during installation)

| Library  | Version | Description |
|----------|---------|-------------|
| PyTorch  | v2.12.0 | Core framework (also a runtime dependency) |
| oneDNN   | v3.11   | JIT-compiled kernels for attention, norms, and linear ops |
| FBGEMM   | v1.2.0  | Quantized EmbeddingBag kernels |
| libXSMM  | [c14cbc6](https://github.com/libxsmm/libxsmm/commit/c14cbc6f8bc7964f8c5190a3a16b8cace03e5889) | Tensor Processing Primitives (TPP) for MLPs and linear ops |
| AOCL-DLP | [3256da1](https://github.com/amd/aocl-dlp/commit/3256da129b879d09301c966026a603f3f86e1233) | AMD AOCL Deep Learning Primitives for GEMMs |

### Runtime (Python packages)

| Package | Version |
|---------|---------|
| transformers | 5.8.1 |
| safetensors | >= 0.5.2 |
| huggingface-hub | 1.6.0 |
| fastapi | 0.115.12 |
| uvicorn | 0.34.2 |
| prometheus_client | 0.23.1 |

See `runtime_requirements.txt` for the full list of runtime dependencies installed by `pip install amd-pace`; `requirements.txt` is the developer aggregator (runtime + build + bench/script + lint/test tools).

## Resources

* [AMD PACE Blog: A vLLM Plugin for CPU Inference](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace-integrates-with-vllm.html)
* [AMD PACE Blog: High-Performance Platform Aware Compute Engine](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace---high-performance-platform-aware-compute-engine.html)
* [PARD (Parallel Draft) Speculative Decoding](https://github.com/AMD-AGI/PARD)
