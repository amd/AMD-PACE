# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import importlib
from typing import Dict, Tuple, Type

from pace.llm.models.base_model import BaseModel

_DECODER_MODELS = {
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "GPTJForCausalLM": ("gptj", "GPTJForCausalLM"),
    "Qwen2ForCausalLM": ("qwen", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": ("qwen", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma3ForConditionalGeneration": ("gemma3", "Gemma3ForConditionalGeneration"),
    "GptOssForCausalLM": ("gpt_oss", "GptOssForCausalLM"),
    "Llama4ForCausalLM": ("llama4", "Llama4ForCausalLM"),
    "Llama4ForConditionalGeneration": ("llama4", "Llama4ForCausalLM"),
    "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),
    "Gemma4ForConditionalGeneration": ("gemma4", "Gemma4ForConditionalGeneration"),
    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
    "Qwen3_5MoeForConditionalGeneration": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
}

_ENCODER_MODELS = {
    "BertModel": ("bert", "BertModel"),
    "BertForSequenceClassification": ("bert", "BertForSequenceClassification"),
    "BertForTokenClassification": ("bert", "BertForTokenClassification"),
    "ModernBertModel": ("modernbert", "ModernBertModel"),
    "ModernBertForSequenceClassification": (
        "modernbert",
        "ModernBertForSequenceClassification",
    ),
    "ModernBertForTokenClassification": (
        "modernbert",
        "ModernBertForTokenClassification",
    ),
    # MaskedLM checkpoints load as the encoder body; no fill-mask API.
    "ModernBertForMaskedLM": ("modernbert", "ModernBertModel"),
}


def models_supported():
    """Decoder architectures loadable by the native generation backend."""
    return _DECODER_MODELS.keys()


def encoder_models_supported():
    """Encoder architectures loadable by ``EmbeddingModel``."""
    return _ENCODER_MODELS.keys()


def _load_class(table: Dict[str, Tuple[str, str]], model_arch: str):
    module_name, model_cls_name = table[model_arch]
    module = importlib.import_module(f"pace.llm.models.{module_name}")
    return getattr(module, model_cls_name, None)


def get_model_class(model_arch) -> Type[BaseModel]:
    """Decoder class for ``model_arch``. Encoder names are rejected here."""
    if model_arch in _DECODER_MODELS:
        return _load_class(_DECODER_MODELS, model_arch)
    if model_arch in _ENCODER_MODELS:
        raise ModuleNotFoundError(
            f"{model_arch} is an encoder model; load it with EmbeddingModel, "
            "not LLMModel."
        )
    raise ModuleNotFoundError(f"The model arch {model_arch} is not supported yet.")


def get_encoder_model_class(model_arch) -> Type[BaseModel]:
    """Encoder class for ``model_arch``. Decoder names are rejected here."""
    if model_arch in _ENCODER_MODELS:
        return _load_class(_ENCODER_MODELS, model_arch)
    if model_arch in _DECODER_MODELS:
        raise ModuleNotFoundError(
            f"{model_arch} is a decoder model; load it with LLMModel, "
            "not EmbeddingModel."
        )
    raise ModuleNotFoundError(f"The model arch {model_arch} is not supported yet.")
