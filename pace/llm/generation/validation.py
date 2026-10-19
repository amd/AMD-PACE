# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os

import torch
from transformers.utils import CONFIG_NAME


def validate_model_inputs(model_path, tokenizer_path, dtype) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    config_path = os.path.join(model_path, CONFIG_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"model path does not contain {CONFIG_NAME}: {model_path}"
        )
    if tokenizer_path is not None and not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer_path}")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"generation dtype must be a torch.dtype, got {type(dtype)}")
