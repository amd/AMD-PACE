# pace-vllm Internals

This document covers the technical details of how `pace-vllm` integrates with vLLM. For installation and basic usage, see the [main README](../README.md).

## Plugin architecture

`pace-vllm` registers PACE as a vLLM CPU platform via the `vllm.platform_plugins` entry point, declared in [`packages/pace_vllm/pyproject.toml`](../pyproject.toml):

```toml
[project.entry-points."vllm.platform_plugins"]
pace = "pace_vllm:register"
```

When vLLM imports this plugin at startup, [`pace_vllm.register()`](../pace_vllm/__init__.py) runs and:

1. **Validates the installed vLLM version** against a hard-coded supported range (see [vLLM version compatibility](#vllm-version-compatibility)). On mismatch, returns `None` — vLLM treats that as a no-op plugin and falls back to its stock CPU pipeline.
2. **Loads the bundled `libpace_cpp.so`** via `torch.ops.load_library`, which fires the `TORCH_LIBRARY_FRAGMENT` static initializers and populates `torch.ops.pace.*` and `torch.classes.pace.*`. The standalone `pace` Python package is **never imported** — pace-vllm is fully self-contained at runtime.
3. **Returns the platform import path** `"pace_vllm.platform.PacePlatform"`, which vLLM uses to swap its CPU platform.

`PacePlatform` then:

- Routes the CPU worker through PACE-specific subclasses (`PaceWorker`, `PaceModelRunner`).
- Swaps the attention backend to `PaceAttentionBackend`.
- Owns KV cache memory via `torch.classes.pace.SlabPool` (one `SlabPool` per attention layer).
- Substitutes PACE OOTs (out-of-tree custom ops) for vLLM's default
  Linear/RMSNorm layers and eligible gated, unquantized BF16 FusedMoE layers.
  Non-gated or non-TPP-packable MoE layers fall back to vLLM's stock CPU
  dispatch.

Under compile mode (`enforce_eager=False`), pace-vllm also installs a post-grad pattern matcher (`FusedMLPPass`) into Inductor's `post_grad_custom_post_pass`. Once Inductor compiles the model graph, every matched MLP block — gated SwiGLU / GeGLU, or ungated `fc1->act->fc2` (with or without bias, across silu / gelu-tanh / gelu-exact / relu) — is rewritten into a single `pace::libxsmm_fused_mlp` call.

## What the source install actually does

`pip install --no-build-isolation -v packages/pace_vllm` (from the repo root) runs:

1. **`python setup.py build_clib`** as a subprocess, which invokes the parent (`amd-pace`) CMake pipeline. This builds `libpace_cpp.so`.
2. **Copies `libpace_cpp.so`** into the pace-vllm wheel's `pace_vllm/lib/` directory.
3. **Snapshots `pace/_register_fake.py`** to `pace_vllm/_fakes_snapshot.py`, so `torch.compile` can resolve fake / meta impls without importing the standalone `pace` package.

## Startup log lines

When the plugin loads cleanly you'll see, in order:

```
Loading plugin pace
pace-vllm: vllm 0.30.0 satisfies supported range '>=0.28.0,<=0.30.0'.
pace-vllm: loaded bundled libpace_cpp.so via torch.ops.load_library.
pace-vllm: native surfaces live (torch.ops.pace.rmsnorm, torch.classes.pace.SlabPool).
pace-vllm: plugin active, PacePlatform registered.
pace-vllm: PacePlatform active (worker_cls=...PaceWorker, was=..., attn_backend=...PaceAttentionBackend, prefix_caching=<true|false>, auto_custom_ops=['rms_norm']).
pace-vllm: PaceWorker.init_device starting.
pace-vllm: registered OOTs (TPP Linear + ParallelLMHead + JIT RMSNorm + FusedMoE).
pace-vllm: installed FusedMLPPass into post_grad_custom_post_pass.
pace-vllm: PaceModelRunner active.
pace-vllm: PaceWorker active (worker=PaceWorker, model_runner=PaceModelRunner).
pace-vllm: bound <N> zero-sized KV placeholders (slab owns all).
pace-vllm: PaceKVCache sized from budget=<N> GiB, layers=<N>, geometries=<N>, num_blocks=<N> per layer (total <N> GiB).
pace-vllm: PaceKVCache allocated (layers=<N>, geometries=<N>, sample block_size=<N> num_kv_heads=<N> head_dim=<N>, total=<N> GiB).
pace-vllm: FusedMLPPass matched <N> MLP sites.
```

The `installed FusedMLPPass...` line only appears under compile mode (`enforce_eager=False`); the `matched <N> MLP sites.` line lands later, once Inductor compiles the model graph. With `enforce_eager=True` both lines are absent and the MLP runs through three separate `pace::libxsmmlinear_plain` calls.

To bypass `FusedMLPPass` entirely without rebuilding (A/B vs the unfused path, or to mitigate a regression on a specific model in seconds), set `PACE_VLLM_FUSED_MLP=0`. The MLP then runs through three separate `pace::libxsmmlinear_plain` calls, identical to the `enforce_eager=True` shape.

During model load, every eligible Linear / ParallelLMHead also logs one `pace-vllm: Linear <prefix> packed into TPP layout (out=<N>, in=<N>, packed_shape=...)` line. Per scheduler step you'll see `pace-vllm: slab lifecycle step: +<n> new, -<n> finished, !<n> preempted (reset).` whenever request lifecycle changes.

## Configuration

| Variable | Source | Behavior |
| --- | --- | --- |
| `VLLM_PLUGINS` | vLLM | unset -> all plugins. `pace` -> only pace-vllm. `""` -> no plugins. |
| `VLLM_CPU_KVCACHE_SPACE` / `--kv-cache-memory-bytes` | vLLM | Explicit CPU KV cache budget. When unset, `PaceWorker.determine_available_memory` delegates to vLLM's `CPUWorker` auto-formula (`numa_total * memory_fraction - process_RSS`) and stashes the result on `cache_config.kv_cache_memory_bytes` so `PaceKVCache.from_kv_cache_config` reads a populated field either way. |
| `PACE_VLLM_SLAB_BLOCK_SIZE` | pace-vllm | Optional integer override that skips the C++ L2 autotuner and forces a specific SlabPool block size (in tokens). Unset/empty leaves the autotuner in charge; non-positive or non-integer raises at startup. |
| `OMP_NUM_THREADS` | OpenMP | CPU thread count for inference. |
| `PACE_VLLM_CUSTOM_OPS` | pace-vllm | Which pace `CustomOp` OOTs auto-fire in compile mode. Accepts `all` (default), `none`, or `rms_norm`. Per-op `+/-` overrides via `compilation_config.custom_ops` still apply. |
| `PACE_VLLM_FUSED_MLP` | pace-vllm | Set to `0` / `false` / `off` / `no` / `disable` / `disabled` to skip `FusedMLPPass` entirely (MLPs stay on three `pace::libxsmmlinear_plain` calls). Unset (default) or any other value keeps the pass enabled. Useful for A/B'ing fused vs unfused without rebuilding. |
| `enforce_eager` (vLLM `LLM` / `--enforce-eager`) | vLLM | `False` (default) lets `FusedMLPPass` rewrite MLP blocks into `pace::libxsmm_fused_mlp`. `True` disables compile entirely; the pass early-returns and the MLP stays on three separate TPP Linears. `FusedMLPPass` is **not** gated by `PACE_VLLM_CUSTOM_OPS`. |

## Prefix caching

When vLLM prefix caching is enabled, pace-vllm installs `PaceSlabConnector` and
uses PACE's SlabPool to retain and restore complete reusable prefix blocks,
while leaving at least one prompt token for recomputation. The connector aligns
vLLM's scheduler block size with the SlabPool block size, requires uniform slab
block sizing, and coordinates cache lifecycle across `uni` or `mp` workers.

Prefix caching is currently incompatible with speculative decoding, async
scheduling, KV offloading, another KV connector, and hybrid Mamba/SSM cache
models. For hybrid models, pace-vllm disables prefix caching so the SSM state
remains correct.

## vLLM version compatibility

`pace_vllm.register()` checks the installed `vllm.__version__` against a hard-coded supported range before loading any native libraries or registering the platform. The range lives in `_PACE_VLLM_SUPPORTED_VLLM_RANGE` at the top of [`pace_vllm/__init__.py`](../pace_vllm/__init__.py); the current value is `>=0.28.0,<=0.30.0`.

pace-vllm supports vLLM versions 0.28.0 through 0.30.0. vLLM 0.30.0 is validated across the full pace-vllm compatibility matrix. Versions before 0.28.0 are not supported.

There is no env-var override — changing the supported range requires a source edit and end-to-end re-verification.

If the installed vLLM is outside the range, `register()` logs a warning and returns `None`, which vLLM treats as a no-op plugin and falls back to its stock CPU pipeline.
