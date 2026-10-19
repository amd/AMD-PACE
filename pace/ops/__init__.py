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
    L2Norm,
    Gemma3RMSNorm,
    RMSNormGated,
    FusedRMSNormResidual,
    FusedGemma3RMSNormResidual,
    FusedLayerNormResidual,
)
from .rotary_embedding import RotaryEmbedding
from .gated_deltanet import (
    GatedDeltaAttention,
    GatedDeltaRecurrence,
    CausalConv1dUpdate,
)
from .activations import SoftMax, Sigmoid, Activation
from .fused_linear import (
    FusedLinearGelu,
    FusedLinearMul,
    FusedLinearRelu,
    FusedLinearSiLU,
)
from .mlp import MergedMLP
from .moe import MoEExperts
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
    "L2Norm",
    "Gemma3RMSNorm",
    "RMSNormGated",
    "FusedRMSNormResidual",
    "FusedGemma3RMSNormResidual",
    "FusedLayerNormResidual",
    "RotaryEmbedding",
    "GatedDeltaAttention",
    "GatedDeltaRecurrence",
    "CausalConv1dUpdate",
    "SoftMax",
    "Sigmoid",
    "Activation",
    "FusedLinearGelu",
    "FusedLinearMul",
    "FusedLinearRelu",
    "FusedLinearSiLU",
    "MergedMLP",
    "MoEExperts",
    "FusedQKVLinear",
    "PagedAttentionMetadata",
    "paged_attention_reshape_and_cache",
    "paged_attention_with_kv_cache",
    "get_paged_attention_scheduler_metadata",
    "get_optimal_attention_isa",
]
