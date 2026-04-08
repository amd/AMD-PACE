# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

try:
    from enum import auto, StrEnum
except ImportError:
    # Backport for Python versions < 3.11
    # This is needed for compatibility with older Python versions
    # where StrEnum is not available in the enum module.
    from backports.strenum import StrEnum
    from enum import auto


# These are imported here so that they are available in the pace.llm namespace
# and can be used in the llm module without importing them separately
# Some of them are wrapped for LLM use cases
from pace.ops import (  # noqa: F401
    OperatorType,
    BackendType,
    DataType,
    Linear,
    RepeatedKVLinear,
    LayerNorm,
    RMSNorm,
    FusedRMSNormResidual,
    FusedGemma3RMSNormResidual,
    FusedLayerNormResidual,
    RotaryEmbedding,
    SoftMax,
    Sigmoid,
    Activation,
    MergedMLP,
    FusedQKVLinear,
)


class LLMOperatorType(StrEnum):
    """
    LLMOperatorType is an enumeration class that contains the different operator types used in the LLM.
    """

    QKVProjection = auto()
    RoPE = auto()
    Attention = auto()
    OutProjection = auto()
    MLP = auto()
    Norm = auto()
    LMHead = auto()


# Alias for BackendType so that the user can use it as LLMBackendType
# and can be imported from pace.llm without importing BackendType separately
LLMBackendType = BackendType
