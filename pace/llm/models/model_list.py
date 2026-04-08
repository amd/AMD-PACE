# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import importlib
from typing import Type

from pace.llm.models.base_model import BaseModelForCausalLM

"""
The dictionary of model architectures supported.
The key is the model architecture name, and the value is a tuple of the module name and the class name.
"""
_MODELS = {
    # Architecture -> (module, class).
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "GPTJForCausalLM": ("gptj", "GPTJForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma3ForConditionalGeneration": ("gemma3", "Gemma3ForConditionalGeneration"),
    "GptOssForCausalLM": ("gpt_oss", "GptOssForCausalLM"),
}


def models_supported():
    """
    Get the list of model architectures supported.
    """
    return _MODELS.keys()


def _get_model(model_arch):
    """
    Load the model class from the module. This is a helper function for lazy loading.

    Args:
        model_arch: The model architecture name.

    Returns:
        The model class.
    """
    module_name, model_cls_name = _MODELS[model_arch]
    module = importlib.import_module(f"pace.llm.models.{module_name}")
    return getattr(module, model_cls_name, None)


def get_model_class(model_arch) -> Type[BaseModelForCausalLM]:
    """
    Get the model class for the given model architecture.
    If the model architecture is not supported, raise a ModuleNotFoundError

    Args:
        model_arch: The model architecture name.

    Returns:
        The model class.

    Raises:
        ModuleNotFoundError: If the model architecture is not supported.
    """
    if model_arch in models_supported():
        return _get_model(model_arch)
    raise ModuleNotFoundError(f"The model arch {model_arch} is not supported yet.")
