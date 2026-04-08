# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from typing import List

SUPPORTED_MODEL_LIST: List[dict] = [
    {"id": "meta-llama/Llama-3.1-8B", "dtypes": ["bfloat16", "float32"]},
    {"id": "meta-llama/Llama-3.2-3B", "dtypes": ["bfloat16", "float32"]},
    {"id": "EleutherAI/gpt-j-6b", "dtypes": ["bfloat16", "float32"]},
    {"id": "Qwen/Qwen2-7B-Instruct", "dtypes": ["bfloat16", "float32"]},
    {"id": "facebook/opt-6.7b", "dtypes": ["bfloat16", "float32"]},
    {"id": "microsoft/phi-4", "dtypes": ["bfloat16", "float32"]},
]
