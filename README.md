# AMD PACE - AMD Platform Aware Compute Engine

AMD PACE is a high-performance LLM inference engine and agentic AI runtime built from the ground up for AMD EPYC CPUs. It combines a PyTorch C++ extension, CPU-native KV cache management, a production-ready serving stack, and PACE Agent's LangGraph orchestration and profiling tools.

🔥 **Check out our latest blog post: [Agentic Flow with AMD PACE](https://www.amd.com/en/developer/resources/technical-articles/2026/agentic-flow-with-amd-pace.html)** — how PACE extends from optimized LLM inference into a LangGraph-based runtime for deterministic agent execution, local/remote model deployment, custom agent graphs, and optimized tool execution.

📦 **Now available on PyPI** — `pip install amd-pace` (and `pip install pace-vllm` for the [vLLM plugin](packages/pace_vllm/README.md)). Pre-built **manylinux wheels** support Python 3.10 – 3.14; no compiler required.

> NOTE: AMD PACE is designed and tested for systems with AVX512 or higher support. On systems lacking AVX512, performance may degrade significantly due to fallback to slower reference implementations, or the library might not function as intended.

## Highlights

* **vLLM Plugin (`pace-vllm`)** — A drop-in [vLLM platform plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) that routes vLLM's CPU worker through PACE's attention, SlabPool KV cache, Linear/RMSNorm, and eligible unquantized FusedMoE implementations. Compile mode also fuses eligible MLP blocks, and vLLM prefix caching is integrated through `PaceSlabConnector`. More in [packages/pace_vllm/README.md](packages/pace_vllm/README.md).

* **FusedMoE** — A routed Mixture-of-Experts operator with configurable routing, shared experts, and optimized TPP execution. It supports the MoE model paths used by Llama 4, Qwen3 MoE, Gemma 4, and GPT-OSS, and is integrated into `pace-vllm`.

* **Inference Server** — `pace-server` provides a router/engine serving stack with a `continuous_batch` scheduler that combines chunked prefills and decodes, multi-instance NUMA-aware execution, OpenAI-shaped text and chat completion endpoints, and built-in Prometheus and scheduler metrics. More in [InferenceServer.md](docs/InferenceServer.md).

* **Automatic Prefix Caching** — `pace-server` reuses complete prompt-prefix blocks across requests with its `SLAB_POOL` and `PAGED` KV caches. It is enabled by default and can be disabled with `PACE_PREFIX_CACHE=0`.

* **Broad Model Support** — Llama 3 and text-only Llama 4, Qwen2/2.5/3 (including Qwen3 MoE), Phi3/4, Gemma 3 and text-only Gemma 4, GPT-J, OPT, and GPT-OSS, with shared operator and backend infrastructure. See [LLM.md](docs/LLM.md).

* **PACE Agent** — A LangGraph-based agent runtime for deterministic replay, profiling, and optimization of non-model work such as retrieval, serialization, context assembly, orchestration, and tool I/O. It includes WebVoyager and GAIA workflows and supports local or remote OpenAI-compatible model endpoints. See [packages/pace_agent/README.md](packages/pace_agent/README.md).

* **Speculative Decoding (PARD)** — Built-in PARallel Draft Model Adaptation that runs a smaller draft model ahead of the target and verifies speculated tokens in parallel. More in [SpeculativeDecoding.md](docs/SpeculativeDecoding.md).

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
* [Citation](#citation)

## Installation

AMD PACE provides pre-built **manylinux wheels** on PyPI — the simplest install path, no compiler needed. Building from source is supported for developers and bleeding-edge users.

### From PyPI (recommended)

1. **Create a Python 3.12 env with miniforge.** Install miniforge from [here](https://conda-forge.org/miniforge/), then:
   ```
   conda create -n pace-env-py3.12 python=3.12 -y
   conda activate pace-env-py3.12
   ```

   > AMD PACE is tested on Python 3.10 – 3.14; Python 3.12 is the most thoroughly exercised version.

2. **Install CPU PyTorch.** The `+cpu` build is not published on PyPI, so it needs PyTorch's index:
   ```
   pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
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

3. **Build and install AMD PACE from source:**
    ```
    pip install -r build_requirements.txt
    pip install --no-build-isolation -v .
    ```
    This builds AMD PACE and installs it in the current environment.

    > NOTE: It uses the new way of building packages with `pip`, for more details refer to [PEP 517](https://peps.python.org/pep-0517/). The `build_requirements.txt` should be passed in during installation to ensure that the build environment is set up correctly, please refer to [PEP 518](https://peps.python.org/pep-0518/) for more details.

    For developers who need to build AMD PACE frequently, using pip with `--no-build-isolation` is recommended to avoid unnecessary overhead of creating isolated environments for each build. This speeds up the build process significantly. Make sure to have all the required dependencies installed in your environment before using this option.
    ```
    pip install --no-build-isolation [-v] .
    ```

    > NOTE: Building AMD PACE, especially the oneDNN component, can require significant memory. If your system does not have enough RAM, the build process may fail or your machine may run out of memory.

## PACE Agent

PACE Agent profiles and optimizes the CPU work around LLM calls in agentic workflows. It
supports LangGraph agents, deterministic record/replay, benchmark reporting, provider
comparisons, WebVoyager, and GAIA workflows.

Install it from the repository:

```bash
pip install --no-build-isolation ./packages/pace_agent
pace-agent check
```

See the [PACE Agent README](packages/pace_agent/README.md) for configuration, browser,
GAIA, and sandbox setup.

## Models Supported
The following models are supported by AMD PACE:
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

The following limitations are enforced by the current implementation:

| Feature | Unsupported Configuration | Supported Alternative |
|---------|--------------------------|----------------------|
| GPT-OSS | JIT attention backend | SLAB_POOL cache with SLAB or PAGED attention backend |
| Qwen2 | JIT attention backend | SLAB_POOL cache with SLAB or PAGED attention backend |
| Llama 4 / Gemma 4 | Vision and audio inputs | Only text inputs are supported |
| `continuous_batch` scheduler | `BMC` or `DYNAMIC` KV cache | `SLAB_POOL` or `PAGED` KV cache |
| PARD speculative decoding | Random sampling | Greedy decoding |

## External Dependencies

### Build-time (C++ libraries, built from source during installation)

| Library  | Version | Description |
|----------|---------|-------------|
| PyTorch  | v2.13.0 | Core framework (also a runtime dependency) |
| oneDNN   | v3.11   | JIT-compiled kernels for attention, norms, and linear ops |
| libXSMM  | v2.0.0 | Tensor Processing Primitives (TPP) for MLPs and linear ops |
| AOCL-DLP | [v5.3](https://github.com/amd/aocl-dlp/releases/tag/5.3) | AMD AOCL Deep Learning Primitives for GEMMs |

### Runtime (Python packages)

| Package | Version |
|---------|---------|
| transformers | 5.12.1 |
| safetensors | >= 0.8.0 |
| huggingface-hub | 1.21.0 |
| fastapi | 0.138.2 |
| uvicorn | 0.49.0 |
| prometheus_client | 0.25.0 |

See `runtime_requirements.txt` for the full list of runtime dependencies installed by `pip install amd-pace`; `requirements.txt` is the developer aggregator (runtime + build + bench/script + lint/test tools).

## Resources

* [AMD PACE Blog: Agentic Flow with AMD PACE](https://www.amd.com/en/developer/resources/technical-articles/2026/agentic-flow-with-amd-pace.html)
* [AMD PACE Blog: A vLLM Plugin for CPU Inference](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace-integrates-with-vllm.html)
* [AMD PACE Blog: High-Performance Platform Aware Compute Engine](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace---high-performance-platform-aware-compute-engine.html)
* [PARD (Parallel Draft) Speculative Decoding](https://github.com/AMD-AGI/PARD)

## Citation

If you use AMD PACE in your research, please cite it as:

```bibtex
@software{amd_pace,
  author  = {{Advanced Micro Devices, Inc.}},
  title   = {{AMD PACE: AMD Platform Aware Compute Engine}},
  year    = {2026},
  url     = {https://github.com/amd/amd-pace}
}
```
