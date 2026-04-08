# Large Language Models with PACE

## Contents
1. [Introduction](#introduction)
2. [Roadmap](#roadmap)
3. [Inspiration](#inspiration)
4. [Example Usage](#example-usage)
5. [Infrastructure](#infrastructure)
6. [Models](#models)
7. [Adding a New Model](#adding-a-new-model)
8. [KV Cache Backends](#kv-cache-backends)

## Introduction
Unlike the models before LLMs gained popularity, which was mostly focused on model development, and optimizations, LLMs, due to their nature of being autoregressive, require a lot of infrastructure optimizations as well. Thus we are breaking down the whole process of inferencing from an LLM model into two parts: the [infrastructure](#infrastructure) and the [model](#models).

## Roadmap

### **Infrastructure**:

| Features             | Status |
|----------------------|--------|
| Random Sampling      | ✅      |
| Greedy Sampling      | ✅      |
| Streamer             | ✅      |
| Continuous Batching  | ✅      |
| Speculative Decoding | ✅      |
| Graph Compilation    | ❌      |

### **Models**:

| Models        | FP32/BF16 | Dynamic Quantization | Static Quantization |
|---------------|-----------|----------------------|---------------------|
| OPT           | ✅        | ❌                  | ❌                  |
| LLAMA (<=3.3) | ✅        | ❌                  | ❌                  |
| GPT-J         | ✅        | ❌                  | ❌                  |
| Phi3/4        | ✅        | ❌                  | ❌                  |
| QWEN2/2.5     | ✅        | ❌                  | ❌                  |
| Gemma 3       | ✅        | ❌                  | ❌                  |
| GptOss        | ✅        | ❌                  | ❌                  |


## Inspiration

The inspiration behind creating a new infrastructure Large Language Models is so that the optimizations can be done in both **infrastructure** and the **model** level. The infrastructure should take in a HF models _as is_ and should be able to run inference on it. The infrastructure should also support the ability to run multiple models (model independent), with multiple data types (type independent).

> Some methods has been taken/adapted from [HF Transformers](https://github.com/huggingface/transformers), and [vLLM](https://github.com/vllm-project/vllm).

## Example Usage

Here is how you can load a model and run inference on it:

```python
import torch
import pace
from transformers import AutoTokenizer
from pace.llm import LLMModel, SamplingConfig

model_name = "model-name"
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
inputs_encoded = tokenizer.batch_encode_plus(["Input.."], return_tensors="pt", padding="longest")

pace_model = LLMModel(model_name, dtype=torch_dtype)
sampling_config = SamplingConfig(
    max_new_tokens=35,
    do_sample=False)
pace_output = pace_model.generate(inputs_encoded, sampling_config)
print(tokenizer.decode(pace_output.output_token_ids[0], skip_special_tokens=True))
```

For a more detailed example, please refer to the example given in `examples/pace_llm_basic.py`


## Infrastructure
These are the components of the infrastructure and they are all under `pace/llm`:

### LLMModel
The `LLMModel` class is what is exposed to the user and what the user should use to run inference on any model. The `LLMModel` class accepts a model id (from HF) or path to a model locally. `LLMModel` class is like a frontend to the `Generator` class, on calling `generate` method, it internally calls the `Generator` class to run inference on the model.

`LLMModel` methods:
1. Constructor: Accepts the path to the model path (mandatory), a tokenizer path, and the data type for the model.
2. `generate`: Accepts the input and runs inference on the model. The `generate` method accepts the input, and the sampling criteria.

### Generator
The `Generator` class is responsible for generating the output from the model. The `Generator` class is model independent and can be used to run inference on any model.

The `Generator` class is responsible for:
1. Loading the model, loading the correct weights and configurations for the model (through [`model_utils`](#model_utils))
2. Loading and managing the tokenizer.
3. Preprocessing the input, managing the sampling, and the stopping criteria.
4. Running inference on the model in an auto-regressive manner.
5. Managing KV cache for the model with the help of `KVCacheManager` (from `pace.llm.attention`).

`Generator` methods:
1. Constructor: Accepts the model path, tokenizer path, and the data type for the model.
2. `prepare_for_generate`: Accepts the input and the sampling criteria and prepares the input, the mask and the sampler and the stopping criteria for the model.
3. `generate`: Accepts the input and runs inference on the model. The `generate` method accepts the input, and runs a while loop to generate the output. The loop passes the input through the model, the sampler and finally breaks when the stopping criteria is met.

### Sampler
`Sampler` class is responsible for sampling the next token from the model. The `Sampler` class is model independent and can be used to run inference on any model. The `Sampler` takes in the logits from the model and samples the next token based on the sampling criteria. The Sampling criteria is provided by [`SamplingConfig`](#samplingconfig).

`Sampler` divides the sampling into three parts:
1. Penalties: `repetition_penalty`, `frequency_penalty` — applied to raw logits before any filtering.
2. Preprocessors: `temperature`, `top_k`, `top_p`, `min_p` — shape the logit distribution.
3. Sampling: `greedy` (argmax) or `random` (multinomial from the resulting probabilities).

`Sampler` methods:
1. Constructor: Accepts the sampling criteria.
2. `sample`: Accepts the logits from the model and samples the next token based on the sampling criteria. The sampling criteria can be greedy or random sampling.

### Stopping Criteria
`StoppingCriteria` class is responsible for stopping the generation process based on the stopping criteria. The `StoppingCriteria` class is model independent and can be used to run inference on any model. The `StoppingCriteria` takes in the generated tokens and stops the generation process based on the stopping criteria.

`StoppingCriteria` methods:
1. Constructor: Accepts the sampling config.
2. `stop_now`: Accepts the generated tokens and checks if the stopping criteria is met. The stopping criteria can be based on the number of tokens generated, EOS token or a stop string (more to be added later).

### Configs
Some of the configuration files which helps to configure the generation process.

#### SamplingConfig
`SamplingConfig` contains multiple strategies like `top_k`, `top_p`, `temperature` etc. It is adapted from the HF implementation. Please check `pace/llm/configs.py` for more details.

### model_utils
`model_utils` is a utility class that is responsible for loading the model, the tokenizer, and the configurations for the model. The `model_utils` class is model independent and can be used to load any model.

It is responsible for:
1. Loading the config from the model path, identifying the model type and loading the correct model class.
2. Taking care of casting data types (FP32/BF16 supported for now).
3. Checking if the model weights are properly present in the path, and load the weights into RAM and call the `model.load_weights` method to load the weights into the model properly according to the dictionary. Supports both `.bin` and `.safetensors` formats for weight files.
4. Loading the tokenizer from the tokenizer path if provided else from the model path.

### hf_utils
`hf_utils` module is responsible for resolving the model path by downloading or loading from the cache for the model weights if the model name is provided. It does the same for the tokenizer as well.

## Models
All models will be adapted from the HF repo with inference only ops. One forward pass is done to generate one token. The models will be added in the `models` directory.

### BaseModelForCausalLM
`BaseModelForCausalLM` is an abstract base class for all generator based models. All the models implemented in PACE will inherit from this class. It contains an initializer, a forward pass, and a load weights method, all of which are abstract and need to be implemented by the child classes.

## Features

### Streamers
Streamers are used to stream the output to the stdout, as soon as the output is generated. The streamers are model independent and can be used to stream the output of any model. HuggingFace provides a [`TextStream`](https://huggingface.co/docs/transformers.js/en/api/generation/streamers#generationstreamerstextstreamer) class which is used to stream the output to the stdout. The `TextStream` class is used to stream the output of the model to the stdout.

For an example of how to use the streamer, please refer to the example given in `examples/pace_llm_streamer.py`.

## Adding a New Model

All models live in `pace/llm/models/` and inherit from `BaseModelForCausalLM`. The Llama implementation (`pace/llm/models/llama.py`) is a good reference — it also serves Phi3/4 since they share the same architecture.

### Steps

1. **Create the model file**: `pace/llm/models/<arch>.py`

    Define the following components:
    - **Attention module** — uses `FusedQKVLinear`, `RotaryEmbedding`, `Attention`, `Linear` from `pace.llm.ops`
    - **Decoder layer** — attention + MLP + norms, with residual connections
    - **Model backbone** — embedding, decoder layers, final norm
    - **Top-level `<Arch>ForCausalLM`** — inherits `BaseModelForCausalLM`, adds `lm_head`

2. **Implement `forward`**: Signature is `forward(input_ids, positions, kv_cache) -> ModelOutput`. Only new (unprocessed) tokens are passed — the caller manages `num_computed_tokens`.

3. **Implement `load_weights`**: Maps HuggingFace checkpoint weight names to PACE module parameters. Use `rename_layers` (class attribute) for simple renames and `target_map` for splitting fused projections:
    ```python
    class MyModelForCausalLM(BaseModelForCausalLM):
        # Splits a fused "gate_up_proj" checkpoint weight into separate gate/up
        target_map = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        # Renames to match PACE's MergedMLP sub-module structure
        rename_layers = {
            "up_proj": "up_proj.linear",
            "gate_proj": "gate_proj.linear",
        }
    ```

4. **Register in model list**: Add to `_MODELS` in `pace/llm/models/model_list.py`:
    ```python
    _MODELS = {
        ...
        "<Arch>ForCausalLM": ("<module_name>", "<Arch>ForCausalLM"),
    }
    ```

5. **Accept `OperatorConfig`**: All layers should use the backend from `OperatorConfig` (e.g., `opconfig.qkv_projection`, `opconfig.mlp`, `opconfig.norm`, `opconfig.lm_head`). This allows users to select different backends (NATIVE, JIT, TPP, etc.) per operator type.

### Available PACE Ops

These ops are used in model implementations:

| Op | Import | Purpose |
|----|--------|---------|
| `Linear` | `pace.llm.ops` | General linear projection |
| `FusedQKVLinear` | `pace.llm.ops` | Fused Q/K/V projection with support for MHA and GQA |
| `RMSNorm` | `pace.llm.ops` | RMS normalization |
| `FusedRMSNormResidual` | `pace.llm.ops` | Fused RMSNorm + residual add |
| `RotaryEmbedding` | `pace.llm.ops` | Rotary position embeddings (RoPE) |
| `MergedMLP` | `pace.llm.ops` | Fused gate/up + down projection MLP |
| `Attention` | `pace.llm.attention` | Attention with pluggable backends (JIT, NATIVE, SLAB, PAGED) |

### Patterns

- **Config** comes from HuggingFace via `AutoConfig.from_pretrained(model_path)`.
- **Subclass** when the architecture is very similar to an existing one (e.g., Phi3 reuses the Llama implementation).
- **Weight loading** must handle fused projections — use `target_map` for splitting and `rename_layers` for renaming. For Q/K/V fusion, collect individual projection weights and call `fused_layer.load_from_unfused(tensors)`.

## KV Cache Backends

PACE supports multiple KV cache types, defined in `KVCacheType` (`pace/llm/attention/base.py`). The cache type determines how key/value tensors are stored and which attention backend is used.

| Cache Type | Description | Compatible Attention Backends |
|------------|-------------|-------------------------------|
| `DYNAMIC` | Simple contiguous buffer, dynamically sized per sequence. Good for offline inference with small batches. | JIT, NATIVE |
| `BMC` | Block-Major Contiguous cache. Splits the cache into blocks controlled by `PACE_BMC_NUM_SPLITS`. Better memory utilization for longer sequences. | JIT, NATIVE |
| `SLAB_POOL` | Pool-based slab allocator for production serving. Pre-allocates a fixed memory pool with configurable block sizes. See [SlabAttention.md](SlabAttention.md) for details. | SLAB |
| `PAGED` | Paged attention with block-level memory management. Pool size controlled by `PACE_MAX_CACHE_TOKENS` (default 262144). | PAGED |

### Cache-Attention Compatibility

`OperatorConfig.finalize(cache_type=...)` enforces compatibility. If the user-specified attention backend is incompatible with the cache type, it is overridden with a warning:

| Cache Type  | Default Attention | Allowed Attention Backends |
|-------------|-------------------|----------------------------|
| `DYNAMIC`   | JIT               | JIT, NATIVE                |
| `BMC`       | JIT               | JIT, NATIVE                |
| `SLAB_POOL` | SLAB              | SLAB                       |
| `PAGED`     | PAGED             | PAGED                      |
