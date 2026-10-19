# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Sentence/sequence pooling for encoder embedding models.

Encoders (BERT/bge, ModernBERT/mmBERT) emit per-token hidden states
``[B, S, H]``; an embedding requires reducing the token dimension to a single
``[B, H]`` vector. This module implements the common Sentence-Transformers
pooling strategies plus optional L2 normalization (bge models normalize by
default and compare via cosine similarity).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import torch
import torch.nn.functional as F


class PoolingType(str, Enum):
    """Supported token-pooling reductions."""

    CLS = "cls"
    MEAN = "mean"
    LAST = "last"

    @staticmethod
    def resolve(value: Union[str, "PoolingType"]) -> "PoolingType":
        if isinstance(value, PoolingType):
            return value
        return PoolingType(str(value).lower())


def pool(
    last_hidden_state: torch.Tensor,
    pooling_type: Union[str, PoolingType] = PoolingType.CLS,
    attention_mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Reduce ``[B, S, H]`` token states to ``[B, H]`` sentence embeddings.

    Args:
        last_hidden_state: Encoder output, shape ``[B, S, H]``.
        pooling_type: ``cls`` (first token), ``mean`` (masked average over
            tokens), or ``last`` (last non-pad token).
        attention_mask: ``[B, S]`` (1 = real, 0 = pad). Required for correct
            ``mean`` / ``last`` pooling with padding; ignored by ``cls``.
        normalize: L2-normalize the pooled vector (bge / cosine-sim default).

    Returns:
        ``[B, H]`` pooled embeddings.
    """
    pooling_type = PoolingType.resolve(pooling_type)

    if pooling_type == PoolingType.CLS:
        pooled = last_hidden_state[:, 0]
    elif pooling_type == PoolingType.MEAN:
        pooled = _mean_pool(last_hidden_state, attention_mask)
    elif pooling_type == PoolingType.LAST:
        pooled = _last_pool(last_hidden_state, attention_mask)
    else:  # pragma: no cover - resolve() already validated
        raise ValueError(f"Unsupported pooling type: {pooling_type}")

    if normalize:
        # Compute the norm in fp32 for stability, then cast back so a bf16
        # encoder still returns bf16 embeddings.
        pooled = F.normalize(pooled.float(), p=2, dim=-1).to(last_hidden_state.dtype)
    return pooled


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)  # [B, S, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _last_pool(
    last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    bsz = last_hidden_state.shape[0]
    if attention_mask is None:
        return last_hidden_state[:, -1]
    # Index of the last real token per sequence (handles right-padding).
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    return last_hidden_state[torch.arange(bsz), lengths]
