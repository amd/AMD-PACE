# Performance Guide for AMD PACE

There are multiple factors that can affect the performance of models AMD PACE. This guide provides tips and best practices to help you optimize your setup for the best performance.

## Contents
1. [Standard Practices](#standard-practices)
2. [AMD PACE Operators](#amd-pace-operators)
3. [KV Cache Selection](#kv-cache-selection)
4. [Multi-Instance Serving](#multi-instance-serving)
5. [Profiling](#profiling)
6. [Benchmarking](#benchmarking)
7. [Environment Variables Reference](#environment-variables-reference)

## Standard Practices

The standard practices for PyTorch have been mentioned here: [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). These practices are applicable to AMD PACE as well. Some of them are reiterated below:

#### Utilize Non-Uniform Memory Access (NUMA) Controls
On multi-socket machines, NUMA (Non-Uniform Memory Access) controls can improve performance by optimizing memory locality. For deep learning workloads, binding a process to a specific NUMA node avoids cross-socket memory access, which can reduce overhead and increase throughput.

For core-level control, use `numactl` to bind both CPU and memory:

```bash
numactl --physcpubind=0-<num_cores> --membind=<numa_node> python <pytorch_script>
```

#### Utilize OpenMP
OpenMP is utilized to bring better performance for parallel computation tasks.

`OMP_NUM_THREADS` is the easiest switch to use for accelerating computations. It determines the number of threads used for OpenMP computations. With the following command, PyTorch will run the task on N OpenMP threads.

```bash
export OMP_NUM_THREADS=N
```

`OMP_WAIT_POLICY` is used to control the wait policy for OpenMP threads. AMD PACE recommends setting it to `active` for better performance.

```bash
export OMP_WAIT_POLICY=active
```

#### Optimal Memory Allocator
For deep learning workloads, `jemalloc` or `tcmalloc` can provide better performance than the default `malloc` function by reusing memory more efficiently. AMD PACE recommends using `tcmalloc` for better performance.

To install `tcmalloc` in your current conda environment, run:

```bash
conda install gperftools
```

To use `tcmalloc`, set the environment variable `LD_PRELOAD` to point to the `libtcmalloc.so` library:

```bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:$LD_PRELOAD"
```

#### THP (Transparent Huge Pages)
Transparent Huge Pages (THP) can improve performance by reducing the overhead of page table management. For models supported by AMD PACE, it is recommended to set THP to `always` for better performance.

To check the current THP setting, run:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
```

To set THP to `always`, run (make sure you have the necessary permissions):

```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

For more information on THP, refer to the [Documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-configuring_transparent_huge_pages).


## AMD PACE Operators

AMD PACE provides a set of optimized operators that can significantly improve performance for specific tasks. These operators are designed to take advantage of AMD hardware capabilities, such as AVX512 instructions and efficient memory access patterns.

PACE employs a modular operator framework that separates an operator's interface (Frontend) from its computational logic (Backend). This design allows for multiple, interchangeable backend implementations for a single operator. For example, a `Linear` operation can be executed by a `NATIVE` backend (using standard PyTorch), a `JIT` backend (using a just-in-time compiled kernel from oneDNN), or a `TPP` backend (using Tensor Processing Primitives). For a detailed explanation of the operator framework, see the [Python Operators documentation](./PythonOps.md).

### How to Use AMD PACE Operators
To leverage PACE operators with fine-grained control, they can be instantiated directly along with a preferred backend during instantiation of LLMModel. By default, all the operators will use the `NATIVE` backend (except Attention, which will use `JIT`), which is the standard PyTorch implementation.

For LLM models, there are six main operator types that can be configured:
* `Norm`: Normalization operator
* `QKVProjection`: Query-Key-Value projection operator
* `Attention`: Attention mechanism operator
* `OutProjection`: Output projection operator
* `MLP`: Multi-Layer Perceptron operator
* `LMHead`: Language Model Head operator

To configure these operators, you can use the `OperatorConfig` class:

```python
from pace.llm import LLMModel, OperatorConfig, LLMOperatorType, LLMBackendType
from pace.llm.attention import AttentionBackendType

opconfig = OperatorConfig(
    **{
        LLMOperatorType.Norm: LLMBackendType.JIT,
        LLMOperatorType.QKVProjection: LLMBackendType.TPP,
        LLMOperatorType.Attention: AttentionBackendType.JIT,
        LLMOperatorType.OutProjection: LLMBackendType.TPP,
        LLMOperatorType.MLP: LLMBackendType.TPP,
        LLMOperatorType.LMHead: LLMBackendType.TPP,
    }
)
model = LLMModel(
    model_name_or_path="your_model_name",
    opconfig=opconfig,
    # other parameters
)
```

When using the inference server, the same configuration can be passed as a JSON string via `--op_config`:

```bash
pace-server \
    --server_model your_model_name \
    --op_config '{
        "norm_backend": "JIT",
        "qkv_projection_backend": "TPP",
        "attention_backend": "JIT",
        "out_projection_backend": "TPP",
        "mlp_backend": "TPP",
        "lm_head_backend": "TPP"
    }'
```

> Note: The `Attention` operator uses `AttentionBackendType` (JIT, NATIVE, SLAB, PAGED) instead of `LLMBackendType`. This is because attention has its own backend routing with features like sliding window and sinks that are attention-specific.

For a more detailed example, please see the [PACE LLM Basic Example](../examples/pace_llm_basic.py).

### Backend Guide

| Backend | Description |
|---------|-------------|
| **NATIVE** | Standard PyTorch ops. Default for all operator types except Attention. |
| **JIT** | Uses oneDNN JIT-compiled kernels. Effective for dynamic shapes. |
| **TPP** | Uses libXSMM Tensor Processing Primitives. Blocks weight matrices at model load time. Default block size is 32 (`LIBXSMM_BLOCK_SIZE`). |
| **IMBPS** | Iterative MLP with parameter splits. Improves TTFT by splitting MLP parameters. Control splits via `IMBPS_BLOCK_SIZE`. |
| **AOCLDLP** | AOCL Deep Learning Primitives backend. |

**Recommended configuration for best performance:** Norm → JIT, Attention → SLAB (server) or JIT (offline), everything else → TPP.


## KV Cache Selection

PACE supports multiple KV cache types. The choice affects memory management, attention backend compatibility, and overall throughput. See [LLM.md](LLM.md#kv-cache-backends) for the full compatibility matrix.

| Cache Type | When to Use |
|------------|-------------|
| `DYNAMIC` | Simple offline inference, small batches, quick experiments |
| `BMC` | Offline inference with longer sequences. Pre-allocates in parts for better memory management. Control splits with `PACE_BMC_NUM_SPLITS`. For serving with BMC, always use multi-instance if you have enough cores and RAM. |
| `SLAB_POOL` | **Production serving.** Pre-allocates a fixed memory pool. Requires `--kv_cache_memory_gb`. Best for throughput. See [SlabAttention.md](SlabAttention.md). |
| `PAGED` | Paged attention with block-level memory management. |

For the inference server, `SLAB_POOL` is the recommended cache type:

```bash
pace-server \
    --server_model your_model_name \
    --kv_cache_type SLAB_POOL \
    --kv_cache_memory_gb 350 \
    --op_config '{"attention_backend": "SLAB"}'
```


## Multi-Instance Serving

The inference server supports multiple engine instances for higher throughput. The launcher automatically partitions CPU cores across instances and sets NUMA bindings.

```bash
pace-server \
    --server_model your_model_name \
    --num_engine_instances 2 \
    --kv_cache_type SLAB_POOL \
    --kv_cache_memory_gb 350
```

### How It Works

1. The launcher detects the number of sockets and physical cores via sysfs.
2. Cores are split evenly: `cores_per_instance = physical_cores_per_socket / num_instances`.
3. Each engine process is launched with `numactl --physcpubind=<range> --membind=<node>`.
4. `OMP_NUM_THREADS` is set per engine to match its core count.
5. `OMP_WAIT_POLICY=active` is set for each engine.
6. The router process connects to all engine instances and distributes requests.

### Overriding NUMA Bindings

By default, the launcher partitions socket 0 cores. For custom or cross-socket bindings:

```bash
pace-server \
    --server_model your_model_name \
    --num_engine_instances 2 \
    --numa_physcpubind "0-95;96-191" \
    --numa_membind "0;1"
```

- `--numa_physcpubind`: Semicolon-separated per-instance CPU ranges. A single value is broadcast to all instances.
- `--numa_membind`: Semicolon-separated per-instance NUMA node(s). If not set, auto-derived from the core ranges.

For offline (non-server) multi-instance inference, see the [Multi-Instance Example](../examples/multi_instance_pace.py).


## Profiling

AMD PACE has a built-in profiling mechanism that logs per-op wall-time durations. It is controlled by the `PACE_LOG_LEVEL` environment variable.

### Enabling Profiling

```bash
export PACE_LOG_LEVEL=profile
python your_script.py
```

This activates `PROFILE_PACE_FUNCTION` scopes in the C++ ops. When each op completes, a log line is emitted with the op name and duration in milliseconds. Additional info (shapes, sizes) is logged via `PROFILE_ADD_INFO` macros where implemented.

For server metrics (TTFT, TPOT, throughput), see [InferenceServerMonitoring.md](InferenceServerMonitoring.md).


## Benchmarking

### Offline Benchmarks

Use the offline benchmark infrastructure to measure throughput and latency:

```bash
cd benchmarks/llm/performance
python benchmark_llm_offline.py --config benchmark_config.json
```

The config JSON controls the framework, model, dtype, operator backends, KV cache type, and data generation parameters. See [benchmarks/llm/performance/README.md](../benchmarks/llm/performance/README.md) for the full configuration reference.

### Accuracy / Correctness

- Accuracy evaluation: [benchmarks/llm/accuracy/README.md](../benchmarks/llm/accuracy/README.md)
- Correctness comparison: `benchmarks/llm/correctness/`
- MLPerf server scenario: [MLPerf.md](MLPerf.md)


## Environment Variables Reference

### System / OpenMP

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | all cores | Number of OpenMP threads. Set automatically by the launcher for server instances. |
| `OMP_WAIT_POLICY` | (system) | Set to `active` for best latency. Set automatically by the launcher. |
| `LD_PRELOAD` | (unset) | Set to `libtcmalloc.so` path for better memory allocator performance. |

### PACE Core

| Variable | Default | Description |
|----------|---------|-------------|
| `PACE_LOG_LEVEL` | `info` | Log level: `debug`, `profile`, `info`, `warning`, `error`, `none`. Set to `profile` for per-op timing. |
| `PACE_FUSED_MLP` | `1` (enabled) | Enables fused MLP path for the TPP backend. Set to `0` to disable. |
| `PACE_USE_AOCL_DLP_RESHAPE` | `1` (enabled) | Applies AOCL-DLP optimized weight reordering after transpose. Set to `0` to disable. Read by both Python and C++ sides. |

### KV Cache & Attention

| Variable | Default | Description |
|----------|---------|-------------|
| `PACE_BMC_NUM_SPLITS` | auto | Number of pre-allocation splits for BMC KV cache. Higher = more incremental allocation. Auto-defaults to `sqrt(max_seq_length)`. |
| `PACE_MAX_CACHE_TOKENS` | `262144` | Maximum total tokens for the paged KV cache pool. Determines the pool size (`total_blocks = max_tokens / block_size`). Increase for workloads with many concurrent long sequences; decrease to save memory. |
| `PACE_MASK_CACHE_MAX_ENTRIES` | `1000` | Maximum LRU entries in the shared causal-mask buffer pool. Caps memory from many distinct mask shapes. |
| `SLAB_BLOCK_SIZE` | auto (from L2) | SlabPool block size. Auto-tuned from L2 cache. Must be multiple of 16, max 256. |
| `SLAB_LAYOUT` | `head_major` | SlabPool memory layout: `head_major` or `block_major`. |
| `SLAB_SCHEDULE` | auto | OMP scheduling for slab attention: `static` or `dynamic`. |
| `SLAB_MAX_SPLITS` | auto | Maximum Split-K splits for slab attention. |

### TPP / libXSMM Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBXSMM_BLOCK_SIZE` | `32` | Weight blocking size for TPP backend. Try `64` for prompt-heavy workloads. |
| `PACE_NCB_BLOCK_SIZE` | `64` | libXSMM NCB (N-dimension column block) size for GEMM. |
| `PACE_DOWN_NCB` | `32` | libXSMM NCB size for MLP down-projection GEMM. |
| `PACE_MLP_BSB` | `0` (auto) | Batch-size block override for fused MLP GEMM. `0` = auto-select based on batch size. |
| `PACE_GEMM_LOOP_SCHEME` | `aCB` | libXSMM GEMM loop iteration order. |
| `IMBPS_BLOCK_SIZE` | `1` | Parameter split count for IMBPS MLP backend. |
