# STeP in AMD PACE

This document describes STeP (Speculative Tensor Parallelism) in AMD PACE, a lossless, training-free, self-speculative decoding method for serving a model whose weights exceed available VRAM. STeP partitions every dense FFN across the GPU and the CPU and uses the GPU-resident prefix as the draft model.

The work presented here has been accepted to **NeurIPS 2026**:

> Viren Luke Radhakrishnan, Dhruva Kashyap, Pranav K Nayak, Chiranjib Bhattacharyya and Prakash Raghavendra. *In STeP: Speculative Tensor Parallelism for Concurrent Heterogeneous Inference of LLMs*. Conference on Neural Information Processing Systems (NeurIPS), 2026.

## Method

A gated FFN computes `down(act(gate(x)) * up(x))`, a sum over intermediate channels. The sum is partitioned: the first `k` channels are held on the GPU and the remainder on the CPU. Adding the two partial results reconstructs the original FFN.

The partition yields two models from one set of weights:

- **Draft**: the GPU prefix alone, a smaller approximate model that requires no CPU work.
- **Verifier**: the prefix plus the CPU residual, which is the full model.

Repeat until end of sequence
```
Committed context
     ↓
Draft (GPU prefix) → tree of candidate continuations
     ↓
Verifier (GPU prefix + CPU residual) → one forward over the tree
     ↓
Acceptance → longest path the verifier agrees with, plus one bonus token
     ↓
KV cache compaction onto the accepted path
```

Accepted tokens are the full model's tokens. Splitting the FFN changes the order of floating-point additions, so output can differ slightly from an unsplit run of the same weights.

### Supported models

STeP loads checkpoints through transformers, from a hub id or a local path, so it is not limited to the architectures PACE's native backend implements. It supports:

- Dense gated FFNs (`gate_proj`, `up_proj`, `down_proj`) with a SiLU (SwiGLU) or GELU (GeGLU) activation. Mixture-of-experts models are not supported.
- Multimodal checkpoints, with the vision tower left unchanged.
- FP8 checkpoints, which are dequantized as they load. Other quantized checkpoints are not supported.

STeP runs one request at a time. A batch, or a second request while one is running, is refused.

## Install

STeP needs a GPU-enabled version of PyTorch installed first, as described under [GPU](../README.md#gpu), and the `gpu` extra.

## Usage

### Calibrating a checkpoint

Calibration runs once per checkpoint and writes a reordered copy. It runs one decoder layer at a time on one GPU, which needs room for the larger of one layer with its activations and the rest of the model (embeddings, output head and any vision tower), rather than the whole model.

The hidden states passed between layers, `samples × sequence_length × hidden_size` values in bf16, stay on the GPU if they fit and in host memory otherwise. Beyond those, a bf16 checkpoint needs only a few GiB of host memory. A checkpoint in FP8 or another dtype is converted on the host as it loads, which needs memory for the whole model, and an FP8 checkpoint's reordered copy is written in bf16.

`--samples` is the total, split evenly across corpora, so this takes 512 windows from each:

```bash
python tools/step_calibrate.py meta-llama/Llama-3.1-70B-Instruct ./llama-3.1-70b-step \
    --dataset wikitext2 --dataset starcoder --samples 1024
```

The same from Python:

```python
from pace.llm.step import StepCalibrationConfig
from pace.llm.step.calibrate import calibrate

calibrate(
    StepCalibrationConfig(
        model_name_or_path="meta-llama/Llama-3.1-70B-Instruct",
        output_dir="./llama-3.1-70b-step",
        datasets=["wikitext2", "starcoder"],
        num_samples=1024,
    )
)
```

### Generating

```python
import torch
from transformers import TextStreamer
from pace.llm import LLMModel, SamplingConfig, StepGenerationBackendConfig

model = LLMModel(
    "./llama-3.1-70b-step",
    dtype=torch.bfloat16,
    generation_backend_config=StepGenerationBackendConfig(
        vram_budget_gb=80.0,
        max_seq_length=4096,
        cpu_threads=64,
    ),
)

tokenizer = model.get_tokenizer()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain speculative decoding."}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

output = model.generate(
    inputs,
    SamplingConfig(max_new_tokens=128, do_sample=False),
    text_streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
)
print(output.speculative_stats.describe())
```

`LLMModel` takes token ids, so apply the chat template yourself for an instruction-tuned model, as above. Such models produce poor output on a raw prompt.

A streamer receives each round in one `put`: the accepted draft tokens, then the bonus token. [pace_step.py](../examples/pace_step.py) prints accepted draft tokens in green and bonus tokens in grey.

`output.speculative_stats` is a `StepStats` with tokens per step, accepted and proposed draft tokens, decode throughput, prefill time, plain rounds and evictions. With `collect_timings` it also splits decode time between drafting, verification and the CPU residual. The first request in a process includes kernel warm-up in its prefill time.

[pace_step.py](../examples/pace_step.py) exposes the split, tree, speculation and sampling settings on the command line:

```bash
python examples/pace_step.py ./llama-3.1-70b-step --vram-budget-gb 80 \
    --prompt "Explain speculative decoding." --max-new-tokens 128
```

### Serving

Select STeP in the inference server with `--generation_backend_config`:

```bash
pace-server --server_model ./llama-3.1-70b-step \
    --generation_backend_config '{"type": "step", "vram_budget_gb": 80, "cpu_threads": 64}'
```

STeP runs under the default `iterative` scheduler, which serves one request at a time per engine and queues the rest. If `--serve_type` is `continuous_batch` or `continuous_prefill_first`, the router uses `iterative` instead and logs a warning.

`--op_config` has no effect on STeP. Leave `--kv_cache_type` at its default. `--kv_cache_memory_gb` is refused, since STeP sizes its own caches from `vram_budget_gb` and `max_seq_length`.

## Calibration

Calibration decides which channels go to the GPU, using the ModHiFi saliency criterion. The saliency of channel `i` is

```
score[i] = ‖W_down[:, i]‖² · E[Φ_i²]
```

where `Φ` is the gated intermediate activation, averaged over a calibration corpus. Channels are reordered by descending saliency.

Scoring runs one decoder layer at a time. A first pass runs each batch through the model with its decoder layers replaced by stand-ins, recording the first layer's hidden states and the masks and position embeddings the model builds for every layer, which differ between layers in models that mix attention types. Each layer then moves to the GPU in turn, runs every batch from the previous layer's output and has its channels scored. The reordered checkpoint is written one shard at a time, each shard's FFN weights permuted as it is saved.

Reordering channels does not change what the model computes, so a calibrated checkpoint is an ordinary safetensors checkpoint that works anywhere its source does.

Use a corpus that resembles your workload. Pass `--dataset` more than once to combine sources. A source is an alias (`wikitext2`, `starcoder`, `c4`), another hub id, or a JSON/JSONL file of your own data with each row's text under `text`, `content` or `code`.

STeP also runs on an uncalibrated checkpoint, but fewer draft tokens are accepted, which matters most at low ratios.

## Choosing the split

`draft_ratio` sets the GPU's share of each FFN. If it is unset, STeP derives it from a VRAM budget:

```
r = (budget - fixed) / ffn
```

`fixed` is what stays on the GPU whatever the split: attention, embeddings, norms, a KV cache for `max_seq_length` tokens, the activations of a prefill that long, and the matrix multiply libraries' scratch space. `ffn` is the FFN's weights with its share of those activations. A derived split therefore runs sequences up to `max_seq_length` without moving any channels. With speculation on, the KV cache term includes a second cache for the draft.

The budget is the memory free on the device at startup minus 1 GiB, or `vram_budget_gb` if that is smaller.

`tools/step_plan.py` reports the split for a budget without loading weights, using the same calculation as the runtime:

```
$ python tools/step_plan.py meta-llama/Llama-3.1-8B-Instruct --vram-budget-gb 12
model      meta-llama/Llama-3.1-8B-Instruct
budget     12.0 GiB
resident   attention 2.50 GiB, embedding 1.96 GiB, norm 0.000 GiB, kv cache 1.00 GiB, prefill activations 0.52 GiB, ffn 10.50 GiB, workspace 0.07 GiB
fixed cost 5.87 GiB, full model 16.70 GiB
split      8064/14336 channels on GPU (r=0.5625, requested 0.5663), 6272 on CPU, aligned for the packed CPU kernel
residual aligned for the fast CPU kernel: True
```

The host holds the CPU residual, `1 - r` of the FFN weights, plus a few GiB, and needs no more while loading a bf16 checkpoint. A checkpoint in FP8 or another dtype is converted on the host as it loads, which needs memory for the whole model.

### Channel alignment

With the PACE kernels, the split is adjusted so the CPU share is a multiple of 64 channels. An unaligned residual falls back to a slower path. With `residual_backend="torch"` the requested ratio is used exactly. `ChannelSplit.describe()` reports which applies.

### Eviction under memory pressure

A sequence longer than `max_seq_length`, or an explicit `draft_ratio` above what the budget holds, would take VRAM use past the budget. STeP then moves the least salient GPU channels to the CPU and continues, with the output unchanged. An explicit `draft_ratio` whose weights alone exceed the budget is replaced by the ratio derived from the budget. Each eviction makes room for the longer of `max_seq_length` and twice the current length, the doubling capped at the prompt plus `max_new_tokens`.

Evicted channels stay on the CPU for the rest of the request and, by default, for later requests as well. `reset_between_requests=True` returns them to the GPU at the start of every request. The backend's `reset()` does the same on demand.

## Speculation control

At low ratios the draft may be too weak for speculation to pay off. `speculation` controls whether STeP drafts:

- `"on"` always speculates.
- `"off"` decodes one token per full-model forward, which is plain tensor parallelism across the GPU and the CPU.
- `"auto"` speculates until a full window of rounds averages fewer than `min_tokens_per_step` tokens, then stops for the rest of the request.

Each round commits at least the bonus token, so tokens per step is never below 1.0.

Once `"auto"` stops speculating, it stays off for later requests too, until `reset_between_requests` or `reset()` clears it along with any eviction.

### Bounding draft cost at long context

`draft_kv_window` caps how much context the draft attends to, so drafting cost stops growing with sequence length. `draft_kv_sink` always keeps that many leading tokens.

The window is off by default. At moderate lengths drafting time is mostly FFN work rather than attention, so a window lowers acceptance without saving time. Enable it only if profiling shows draft attention dominating step time.

## Sampling

Set sampling per request with `SamplingConfig`, as for any backend. Each committed token is drawn from the full model's distribution, so sampling behaves as it does without speculation.

Greedy decoding, `temperature`, `top_k`, `top_p`, `min_p` and `seed` are supported, as are `return_probs` and `return_logprobs`. `repetition_penalty`, `frequency_penalty` and `min_new_tokens` are not: a request that sets one is refused, and one set by the checkpoint's `generation_config.json` is switched off with a warning. `return_input_logprobs` is also refused.

Settings a request leaves unset come from the checkpoint's `generation_config.json`. To sample, pass `do_sample=True` with a temperature above zero. With `do_sample` off, decoding is greedy at any temperature.

## The CPU residual

Each layer runs its GPU prefix and its CPU residual concurrently. The residual usually finishes last, so it determines how long a decode step takes.

`residual_backend` selects the implementation:

- `"pace"` uses PACE's fused FFN kernel, `libxsmm_fused_mlp`, and requires bf16.
- `"torch"` uses plain torch operations.
- `"auto"` uses the PACE kernel when it can, and torch when the dtype is not bf16 or the activation has no kernel.

The choice affects speed only, not output.

## StepGenerationBackendConfig reference

The model, tokenizer and dtype come from `LLMModel` or the server's model settings. In server JSON the config carries `"type": "step"`.

- `draft_ratio` (Optional[float])
    - **Description**: FFN fraction held on the GPU.
    - **Default**: `None`, deriving it from the VRAM budget.
    - **Note**: With the PACE kernels, that is `residual_backend` of `"pace"`, or `"auto"` with bf16, the ratio is adjusted slightly so the CPU share is a multiple of 64 channels. With `"torch"` it is used exactly.

- `vram_budget_gb` (Optional[float])
    - **Description**: VRAM the runtime may use, at most the memory free on the device at startup minus 1 GiB. The split is planned against it, and eviction keeps usage under it.
    - **Default**: `None`, using that maximum.

- `max_seq_length` (int)
    - **Description**: Sequence length the split is planned for. A larger value leaves less of the budget for GPU channels. Longer sequences are still served, with eviction making room.
    - **Default**: `4096`.

- `tree_width` / `tree_depth` (int)
    - **Description**: Candidates per level, and number of levels.
    - **Default**: `3` / `3`.

- `tree_mode` (str)
    - **Description**: `"narrow"` reduces the width by one per level; `"full"` holds it constant.
    - **Default**: `"narrow"`.

- `gpu_index` (int)
    - **Description**: Device ordinal for the GPU prefix.
    - **Default**: `0`.

- `cpu_threads` (Optional[int])
    - **Description**: Threads for the CPU residual.
    - **Default**: `None`, leaving the torch default of one thread per physical core.
    - **Note**: More threads than physical cores oversubscribes the residual and slows decoding.

- `residual_backend` (str)
    - **Description**: `"auto"`, `"pace"` or `"torch"`. Also determines whether the split is aligned.
    - **Default**: `"auto"`.

- `speculation` (str)
    - **Description**: `"on"`, `"off"` or `"auto"`, as described under [Speculation control](#speculation-control).
    - **Default**: `"auto"`.

- `min_tokens_per_step` (float) / `speculation_window` (int)
    - **Description**: Threshold and window for `"auto"`. Speculation stops once the mean tokens per round over the last `speculation_window` rounds falls below `min_tokens_per_step`. A shorter window decides sooner, on less evidence.
    - **Default**: `1.5` / `4`.

- `draft_kv_window` / `draft_kv_sink` (Optional[int] / int)
    - **Description**: Cap on the context the draft attends to, and leading entries always retained.
    - **Default**: `None` / `256`.

- `collect_timings` (bool)
    - **Description**: Break decode time into draft, verify and CPU residual. Adds a device synchronize per phase.
    - **Default**: `False`.

- `reset_between_requests` (bool)
    - **Description**: Start every request from the planned split with speculation re-enabled, undoing any eviction and `"auto"` fallback left by the previous one.
    - **Default**: `False`.

## Code layout

STeP lives in `pace/llm/step/`. Request handling, sampling, stopping and streaming come from PACE's shared generation code in `pace/llm/generation/`.

Calibration, run once per checkpoint:

| file | contents |
|---|---|
| `hf_modules.py` | `find_gated_mlps`, which locates the text decoder's gated FFN blocks for calibration and the split. |
| `saliency.py` | `SaliencyAccumulator`, the ModHiFi criterion, accumulating per-channel scores over calibration data. |
| `calibrate.py` | `calibrate`, `layerwise_saliency`, `save_reordered`. HuggingFace checkpoint in, permuted HuggingFace checkpoint out, scored one decoder layer at a time and written one shard at a time. |

Runtime:

| file | contents |
|---|---|
| `config.py` | `StepGenerationBackendConfig`, registered as backend type `"step"`, and `StepCalibrationConfig`. |
| `backend.py` | `StepGenerationBackend`: prefill, one speculation round per decode, eviction, the speculation policy and `StepStats`. |
| `budget.py` | Memory accounting, `plan_split`, which the runtime and the planning tool share, and channel alignment. |
| `loading.py` | Checkpoint loading: multimodal unwrapping, FP8 dequantization, parameter counts from a config alone, and moving memory-mapped weights to the GPU without duplicating them on the host. |
| `split.py` | `SplitFFN`, the per-layer partition. Owns the transfers, the co-execution, and eviction and restore. |
| `residual.py` | `ResidualFFN`, the CPU-resident channels on `libxsmm_fused_mlp`, with weight packing and fallbacks. |
| `tree.py` | `DraftTree`, `TreeDrafter`, `TreeVerifier`, the tree attention masks and KV cache compaction. |

Shared with the rest of PACE:

| path | contents |
|---|---|
| `pace/llm/utils.py` | FP8 dequantization on load. |
| `pace/utils/inference_dataset.py` | `CalibrationDataset`, the calibration windows, on `LLMInferenceDataset`'s loading. |

Tools, tests and examples:

| path | contents |
|---|---|
| `tools/step_calibrate.py` | Calibrate a checkpoint. |
| `tools/step_plan.py` | Report the split a VRAM budget implies, without loading weights. |
| `tests/llm_infra/step/` | Unit tests for the modules above, with end-to-end tests on a small model when a GPU is present. |
| `examples/pace_step.py` | Generation with the split, tree, speculation and sampling settings on the command line, streaming accepted tokens in green. |
