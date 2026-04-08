# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from .enum import OperatorType, FusedOperatorType, BackendType, DataType
from .linear import Linear, RepeatedKVLinear, FusedQKVLinear
from .normalization import (
    LayerNorm,
    RMSNorm,
    Gemma3RMSNorm,
    FusedRMSNormResidual,
    FusedGemma3RMSNormResidual,
    FusedLayerNormResidual,
)
from .rotary_embedding import RotaryEmbedding
from .activations import SoftMax, Sigmoid, Activation
from .fused_linear import (
    FusedLinearGelu,
    FusedLinearMul,
    FusedLinearRelu,
    FusedLinearSiLU,
)
from .mlp import MergedMLP
from pace.llm.attention.paged.ops import (
    PagedAttentionMetadata,
    paged_attention_reshape_and_cache,
    paged_attention_with_kv_cache,
    get_paged_attention_scheduler_metadata,
    get_optimal_attention_isa,
)

# Required to register backends
from . import backends  # noqa: F401

__all__ = [
    "OperatorType",
    "BackendType",
    "FusedOperatorType",
    "DataType",
    "Linear",
    "RepeatedKVLinear",
    "LayerNorm",
    "RMSNorm",
    "Gemma3RMSNorm",
    "FusedRMSNormResidual",
    "FusedGemma3RMSNormResidual",
    "FusedLayerNormResidual",
    "RotaryEmbedding",
    "SoftMax",
    "Sigmoid",
    "Activation",
    "FusedLinearGelu",
    "FusedLinearMul",
    "FusedLinearRelu",
    "FusedLinearSiLU",
    "MergedMLP",
    "FusedQKVLinear",
    "PagedAttentionMetadata",
    "paged_attention_reshape_and_cache",
    "paged_attention_with_kv_cache",
    "get_paged_attention_scheduler_metadata",
    "get_optimal_attention_isa",
]
