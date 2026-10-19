# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch
from transformers import FineGrainedFP8Config

from pace.utils.logging import PACE_LLM_INFO, PACE_LLM_WARNING


def quantization_method(config) -> str:
    """
    Name the quantization scheme a checkpoint was saved in.

    Args:
        config: An HF config.

    Returns:
        str: Lowercased method name, empty when the checkpoint is unquantized.
    """
    quantization = getattr(config, "quantization_config", None)
    if quantization is None:
        return ""
    if isinstance(quantization, dict):
        return str(quantization.get("quant_method", "")).lower()
    return str(getattr(quantization, "quant_method", "")).lower()


def dequantization_config(config, dtype: torch.dtype) -> Optional[FineGrainedFP8Config]:
    """
    The `quantization_config` that makes `from_pretrained` load a quantized checkpoint as dense weights.

    Only FP8 has a dequantizing load path. Other schemes are reported and loaded as saved.

    Args:
        config: The checkpoint's HF config.
        dtype: Dtype the weights are dequantized to.

    Returns:
        A config to pass to `from_pretrained` as `quantization_config`, or None when the checkpoint is unquantized or has no dequantizing path.
    """
    method = quantization_method(config)
    if not method:
        return None
    if "fp8" in method:
        PACE_LLM_INFO(f"checkpoint is {method}; dequantizing to {dtype} on load")
        return FineGrainedFP8Config(dequantize=True)
    PACE_LLM_WARNING(
        f"checkpoint is {method}, for which no dequantization path is implemented; its weights load as saved"
    )
    return None


def drop_quantization_config(*configs) -> None:
    """
    Clear the quantization record from the configs of a dequantized model.

    Without this the saved checkpoint still advertises FP8 in its config, and loaders will attempt to requantize weights that are already dense.

    Args:
        configs: The model's config, plus any nested config holding its own copy, such as a multimodal wrapper's text config.
    """
    for config in {id(config): config for config in configs}.values():
        if getattr(config, "quantization_config", None) is not None:
            config.quantization_config = None
        if hasattr(config, "_pre_quantization_dtype"):
            delattr(config, "_pre_quantization_dtype")
