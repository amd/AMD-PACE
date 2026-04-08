# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ModelInfo:
    """Model configuration for correctness testing.

    Attributes:
        architecture: HF model architecture class name (e.g. 'LlamaForCausalLM').
        default_model: Default HuggingFace model identifier for this architecture.
        dtype: Data type for model weights (default: torch.bfloat16).
    """

    architecture: str
    default_model: str
    dtype: torch.dtype = torch.bfloat16


MODELS = {
    "OPTForCausalLM": ModelInfo("OPTForCausalLM", "facebook/opt-125m"),
    "GPTJForCausalLM": ModelInfo("GPTJForCausalLM", "EleutherAI/gpt-j-6b"),
    "LlamaForCausalLM": ModelInfo("LlamaForCausalLM", "NousResearch/Llama-3.2-1B"),
    "Qwen2ForCausalLM": ModelInfo("Qwen2ForCausalLM", "Qwen/Qwen2.5-0.5B"),
    "Phi3ForCausalLM": ModelInfo("Phi3ForCausalLM", "microsoft/Phi-4-mini-instruct"),
    "Gemma3ForCausalLM": ModelInfo("Gemma3ForCausalLM", "unsloth/gemma-3-270m"),
    "GptOssForCausalLM": ModelInfo("GptOssForCausalLM", "openai/gpt-oss-20b"),
}


def get_all_model_ids() -> List[str]:
    """Return the default model ID for every registered architecture."""
    return [info.default_model for info in MODELS.values()]
