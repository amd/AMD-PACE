# EQUIP in PACE

Bounded KV-cache decode: keep a few **attention sinks** and a fixed **recent**
window so memory does not grow with sequence length.

- **StreamingLLM** slides that window with a shift-and-append update.  
- **EQUIP** does the same keep-set with an **in-place** overwrite of the oldest
  recent slot (faster cache updates; same accuracy as StreamingLLM).

Papers: [EQUIP](https://aclanthology.org/2026.acl-long.1210.pdf) ·
[StreamingLLM](https://arxiv.org/abs/2309.17453)

---

## Before you run

| Need | Where to look |
|------|----------------|
| Install / env, `LLMModel`, KV cache types | [LLM.md](LLM.md) |
| Operator backends (`OperatorConfig`: Norm, QKV, Attention, MLP, …) | [LLM.md](LLM.md) · [PythonOps.md](PythonOps.md) · [Performance Guide — Operators](PerformanceGuide.md#amd-pace-operators) |
| Best host setup for timing runs (NUMA bind, OpenMP, tcmalloc, THP) | [Performance Guide — Standard Practices](PerformanceGuide.md#standard-practices) |

EQUIP needs a **contiguous** cache: `KVCacheType.DYNAMIC` or `KVCacheType.BMC`.
The usual offline default `SLAB_POOL` does **not** support eviction. Attention
must be **JIT** or **NATIVE** (see cache–attention compatibility in
[LLM.md](LLM.md#kv-cache-backends)).

For the most reliable numbers, follow the Performance Guide before
benchmarking: bind CPU/memory with `numactl`, set `OMP_NUM_THREADS` /
`OMP_WAIT_POLICY=active`, preload `tcmalloc`, and use THP `always` when you
can. Details and commands are in
[PerformanceGuide.md](PerformanceGuide.md).

---

## Run generation with EQUIP

Pass `eviction_config` on `LLMModel` with `DYNAMIC` or `BMC`.

```python
from transformers import AutoTokenizer
from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    EvictionConfig,
    RoPEMode,
    UpdateMode,
)

model_path = "NousResearch/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LLMModel(
    model_path,
    kv_cache_type=KVCacheType.DYNAMIC,
    eviction_config=EvictionConfig(
        sink_size=4,
        recent_size=252,          # budget = sink + recent
        rope_mode=RoPEMode.REINDEX,
        update=UpdateMode.INPLACE,  # EQUIP; use SHIFT for StreamingLLM
    ),
)

inputs = tokenizer(["Once upon a time"], return_tensors="pt", padding=True)
out = model.generate(inputs, SamplingConfig(max_new_tokens=128, do_sample=False))
print(tokenizer.decode(out.output_token_ids[0], skip_special_tokens=True))
```

Same flow as `examples/equip_example.py`.

| Setting | What to pick |
|---------|----------------|
| `update=INPLACE` | EQUIP |
| `update=SHIFT` | StreamingLLM-style |
| `rope_mode=REINDEX` | Re-apply RoPE over the kept window |
| `rope_mode=SINK_ONLY` | Refresh sink RoPE only |
| `kv_cache_type` | `DYNAMIC` or `BMC` only |

---

## Benchmark EQUIP vs StreamingLLM vs AR

Compares wall time (and tokens/s) for:

1. **equip** — your EQUIP settings  
2. **streaming** — StreamingLLM (SHIFT)  
3. **ar** — full-cache autoregressive (no eviction)

**1. Edit the config** — start from
`benchmarks/llm/performance/benchmark_equip_config.json`:

- `model_args` — model, dtype, and `llm_operators` (backend per op; same choices
  as `OperatorConfig` — see [Performance Guide](PerformanceGuide.md#amd-pace-operators)
  and [LLM.md](LLM.md))
- `generation_args` — `prompt_len`, `seqs`, `batch_sizes`
- `equip` — sink/recent, rope/update, cache type (or a **list** of EQUIP variants)
- `baselines` — keep `["streaming", "ar"]` for the three-way compare
- `streaming` / `ar` — baseline details (AR defaults to DYNAMIC + NATIVE attention)

**2. Prepare the host** using
[PerformanceGuide.md](PerformanceGuide.md#standard-practices) (NUMA, OpenMP,
tcmalloc, THP), then run from the repo root with your PACE env active:

```bash
# Example shape — adjust core/memory bind to your machine (see Performance Guide)
export OMP_NUM_THREADS=96
export OMP_WAIT_POLICY=active
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD:-}"

numactl --physcpubind=0-95 --membind=0 \
  env python benchmarks/llm/performance/benchmark_equip.py \
    --config benchmarks/llm/performance/benchmark_equip_config.json
```

Optional: `--output path/to/results.json`.

**3. Read results** — printed tables plus JSON under `output_dir` (default
`./benchmark_results`): seconds and tokens/s per method × batch × sequence
length, and EQUIP speedup over each baseline.

For a BMC + NATIVE EQUIP vs streaming vs AR sweep, use
`benchmarks/llm/performance/benchmark_equip_bmc_native.json` the same way.

---

## Note: prompt longer than the budget

When the prompt is longer than `sink_size + recent_size`, prefill builds a
full-prompt KV and compacts to sinks + recent on the first decode step that
needs room. Peak memory follows prompt length; afterward, decode stays within
the budget. The first generated token attends over the full prefill KV; the
rolling eviction window applies from the second generated token onward.
