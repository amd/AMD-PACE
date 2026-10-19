# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from pace.llm.attention.specs import resolve_text_config
from pace.llm.utils import dequantization_config, drop_quantization_config
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_INFO

_ACTIVATION_KEYS = ("hidden_act", "hidden_activation")


def is_multimodal(config) -> bool:
    """
    Whether a config wraps a language model alongside a vision tower.

    Args:
        config: An HF config.

    Returns:
        bool: True for a multimodal wrapper.
    """
    return hasattr(config, "vision_config")


def activation_of(config) -> str:
    """
    Read the FFN activation name from a text config.

    Llama, Mistral and Qwen name this field `hidden_act`; Gemma names it `hidden_activation`.

    Args:
        config: A text config, as returned by `resolve_text_config`.

    Returns:
        str: The activation name.
    """
    found = next(
        (
            name
            for key in _ACTIVATION_KEYS
            if (name := getattr(config, key, None)) is not None
        ),
        None,
    )
    PACE_LLM_ASSERT(
        found is not None,
        f"config names no FFN activation under any of {_ACTIVATION_KEYS}",
    )
    return found


def model_class(config):
    """
    The auto class a checkpoint loads through.

    Multimodal checkpoints are not registered under `AutoModelForCausalLM`, so they load through `AutoModelForImageTextToText`.

    Args:
        config: An HF config.

    Returns:
        The transformers auto class.
    """
    if is_multimodal(config):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def count_parameters(config) -> int:
    """
    Count a checkpoint's parameters from its config, without allocating any weights.

    The model is built on the meta device through the class `load_model` loads it with, and without its quantization record, so the count is what the runtime loads: a vision tower and tied embeddings included, FP8 weights at their dequantized size.

    Args:
        config: The checkpoint's HF config. It is not modified.

    Returns:
        int: Parameter count.
    """
    config = copy.deepcopy(config)
    drop_quantization_config(config, resolve_text_config(config))
    with torch.device("meta"):
        model = model_class(config).from_config(config)
    return sum(parameter.numel() for parameter in model.parameters())


def load_model(model_name_or_path: str, dtype: torch.dtype):
    """
    Load a checkpoint's language model on the host, dequantizing and unwrapping as needed.

    Multimodal checkpoints keep their vision tower, unused. FP8 checkpoints are dequantized by transformers as they load, through the `quantization_config` from `dequantization_config`, since quantized weights cannot be channel-split. transformers' record of that conversion is dropped, so the model saves with dense weights.

    Args:
        model_name_or_path: HF hub id or local directory.
        dtype: Load dtype.

    Returns:
        The loaded model.
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    settings = {"dtype": dtype, "low_cpu_mem_usage": True}

    quantization = dequantization_config(config, dtype)
    if quantization is not None:
        settings["quantization_config"] = quantization

    if is_multimodal(config):
        PACE_LLM_INFO(
            "multimodal checkpoint: splitting the text decoder only, the vision tower is unmodified"
        )
    model = model_class(config).from_pretrained(model_name_or_path, **settings)

    if quantization is not None:
        drop_quantization_config(model.config, resolve_text_config(model.config))
        _drop_fp8_conversion(model)
    return model


def _drop_fp8_conversion(model: nn.Module) -> None:
    """
    Drop the FP8 dequantization transformers recorded at load.

    `save_pretrained` reverses every recorded conversion, which would quantize the dense weights back to FP8.

    Args:
        model: Model loaded from an FP8 checkpoint with dequantization on.
    """
    kept = []
    for conversion in getattr(model, "_weight_conversions", None) or []:
        operations = getattr(conversion, "operations", [])
        if any(isinstance(operation, Fp8Dequantize) for operation in operations):
            PACE_LLM_ASSERT(
                len(operations) == 1,
                "this checkpoint's FP8 weights are also merged or split as they load, which STeP cannot save back",
            )
            continue
        kept.append(conversion)
    model._weight_conversions = kept
    model.hf_quantizer = None


def move_to_device(module: nn.Module, device: torch.device) -> nn.Module:
    """
    Move a module to a GPU, copying each host parameter through a clone.

    A copy to the GPU straight from a memory-mapped checkpoint can leave a private host copy of every page until the file is unmapped. Cloning first holds one extra tensor at a time. Buffers are small and move through `module.to`.

    Args:
        module: Module whose parameters may be views of a memory-mapped checkpoint.
        device: Destination device.

    Returns:
        The module, on `device`.
    """
    if device.type != "cpu":
        for parameter in module.parameters():
            if parameter.device.type == "cpu":
                parameter.data = parameter.data.clone().to(device)
    return module.to(device)


@contextmanager
def on_device(module: nn.Module, device: torch.device):
    """
    Run a module on a GPU for the duration of the block, then give it back its host weights.

    The original host parameters are put back rather than copied back from the device, so memory-mapped weights stay mapped and are never duplicated on the host. Changes made to them on the device are discarded. Buffers come back through `module.to`.

    Args:
        module: Module whose parameters are on the host.
        device: Device to run it on.

    Yields:
        The module, on `device`.
    """
    host = [(parameter, parameter.data) for parameter in module.parameters()]
    try:
        yield move_to_device(module, device)
    finally:
        for parameter, data in host:
            parameter.data = data
        module.to("cpu")
