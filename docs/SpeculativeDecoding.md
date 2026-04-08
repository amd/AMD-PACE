# Speculative Decoding in AMD PACE

This document provides an overview of the Speculative Decoding feature in AMD PACE, which is designed to enhance the performance of Large Language Models (LLMs) by allowing them to generate text more efficiently.

Currently, AMD PACE supports speculative decoding for LLMs, specifically through the PARallel Draft Model Adaptation (PARD) technique.


## PARD

PARallel Draft Model Adaptation (PARD) is a speculative decoding technique that accelerates text generation by using a smaller, faster draft model to predict multiple tokens ahead, which are then verified by the main target model. This approach can significantly reduce the number of forward passes required for text generation. You can know more about PARD in the [PARD paper](https://arxiv.org/pdf/2504.18583).

## PARD Implementation in AMD PACE

AMD PACE implements PARD speculative decoding with the following key components:

### Key Components

- **Target Model**: The main model that produces the final output
- **Draft Model**: A smaller, faster model used for speculation
- **PARD Token**: Special token used for parallel speculation
- **Acceptance Algorithm**: Logic to determine which speculated tokens to accept

### Architecture

Repeat until end of sequence
```
Input Prompt
     ↓
Draft Model → Speculate N tokens → PARD tokens
     ↓
Target Model → Verify all tokens in parallel
     ↓
Acceptance Algorithm → Keep valid tokens + 1 correction
     ↓
Output Tokens
```

## Configuration

PARD is configured using the `PardSpecDecodeConfig` class:

```python
from pace.llm.configs import PardSpecDecodeConfig

pard_config = PardSpecDecodeConfig(
    model_name_or_path="path/to/draft/model",
    pard_token=None,  # Auto-detected from model config
    num_speculative_tokens=12  # Number of tokens to speculate
)
```

### PardSpecDecodeConfig Parameters

#### Required Parameters

- `model_name_or_path` (str)
    - **Description**: Path to the draft model or model name from HuggingFace Hub
    - **Example**: `"amd/PARD-Qwen2.5-0.5B"` or `"/path/to/local/model"`
    - **Requirements**: Must be a valid model path or HuggingFace model name

### Optional Parameters

- `pard_token` (Optional[torch.Tensor])
    - **Description**: Special token used for parallel speculation (used at the time of training the draft model)
    - **Default**: `None` (auto-detected from model config)
    - **Usage**: Usually auto-detected from the draft model's configuration

- `num_speculative_tokens` (int)
    - **Description**: Number of tokens to speculate ahead
    - **Default**: `12`
    - **Range**: Positive integer (typically 1-32)

## Usage Example

```python
import torch
from transformers import AutoTokenizer
from pace.llm import LLMModel, PardSpecDecodeConfig, SamplingConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"

# Configure PARD
pard_config = PardSpecDecodeConfig(
    model_name_or_path="amd/PARD-Qwen2.5-0.5B",
    num_speculative_tokens=12
)

# Initialize model with PARD
model = LLMModel(
    model_name_or_path=model_name,
    dtype=torch.bfloat16,
    spec_config=pard_config
)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Configure sampling
sampling_config = SamplingConfig(
    max_new_tokens=100,
    temperature=0,
    return_text=True
)

# Generate with speculative decoding
response = model.generate(inputs, sampling_config=sampling_config)
```
For a more detailed example, refer to the [PARD example](../examples/pace_llm_pard.py).

## Note
1. AMD PACE with PARD Speculative Decoding is only enabled for Greedy Decoding .
1. Speculative Decoding in AMD PACE is optimized for inter token latency. Multiple requests can be served concurrently with multiple instances, and you can find such an example in [Multi-Instance with AMD PACE](../examples/multi_instance_sd_pace.py).
