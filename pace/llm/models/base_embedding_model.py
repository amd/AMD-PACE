# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Base class for encoder (embedding / classification) models.

Unlike :class:`BaseModelForCausalLM`, encoder models attend bidirectionally,
carry no KV cache, and run a single forward pass over the full sequence.
Their forward signature is ``(input_ids, attention_mask)`` and they return an
:class:`EmbeddingOutput` (or :class:`ClassifierOutput` for classifiers) rather
than autoregressive logits.

Weight-loading plumbing (``rename_fused_params`` / ``preprocess_weights``) is
inherited from :class:`BaseModel`, the base both families share.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch

from pace.llm.models.base_model import BaseModel


class BaseModelForEmbedding(BaseModel):
    """Abstract base class for encoder-only embedding/classification models.

    Everything other than the shape of a forward pass -- config handling,
    ``rename_fused_params``, ``preprocess_weights`` -- comes from
    :class:`BaseModel`, so ``model_utils.load_weights`` works unchanged across
    both families.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Single bidirectional forward pass over the full sequence.

        Args:
            input_ids: Token ids, shape ``[batch_size, seq_len]``.
            attention_mask: ``[batch_size, seq_len]`` (1 = real, 0 = pad).

        Returns:
            EmbeddingOutput or ClassifierOutput.
        """
        ...
