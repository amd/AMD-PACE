# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional, List
from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """
    Output of the model. Contains the logits and key-value caches.
    """

    logits: torch.Tensor


@dataclass
class EmbeddingOutput:
    """Output of an encoder embedding model.

    Args:
        embeddings: Pooled sentence embeddings, shape [batch_size, hidden_size]
            (optionally L2-normalized).
        last_hidden_state: Per-token encoder hidden states,
            shape [batch_size, seq_len, hidden_size]. Retained so callers can
            apply a custom pooling strategy.
    """

    embeddings: torch.Tensor
    last_hidden_state: Optional[torch.Tensor] = None


@dataclass
class ClassifierOutput:
    """Output of an encoder sequence-classification model.

    Args:
        logits: Class logits, shape [batch_size, num_labels].
        probabilities: Optional softmax (multi-class) / sigmoid (multi-label)
            probabilities, same shape as logits.
        predicted_class_ids: Optional argmax class id per sequence,
            shape [batch_size].
    """

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predicted_class_ids: Optional[torch.Tensor] = None


@dataclass
class TokenClassifierOutput:
    """Output of an encoder token-classification model (NER / PII).

    Unlike :class:`ClassifierOutput` (one label per sequence), this carries a
    label distribution for *every token* in the sequence.

    Args:
        logits: Per-token class logits, shape [batch_size, seq_len, num_labels].
        probabilities: Optional softmax probabilities, same shape as logits.
        predicted_class_ids: Optional argmax class id per token,
            shape [batch_size, seq_len].
    """

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predicted_class_ids: Optional[torch.Tensor] = None


@dataclass
class SamplerOutput:
    """
    Output of the sampler. Contains the sampled tokens, and
    optionally their probabilities and log probabilities.
    """

    next_tokens: torch.Tensor
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class SpeculativeStats:
    """
    Statistics for speculative decoding. Contains the number of speculative
    tokens, the number of speculative tokens that were kept, and the number
    of speculative tokens that were discarded.
    """

    total_speculated_tokens: int
    mean_accepted_tokens: float


@dataclass
class GeneratorOutput:
    """
    Output of the generator. Contains the token ids of the generated text,
    and optionally the decoded text, input log probabilities, and
    output probabilities and log probabilities
    """

    output_token_ids: torch.Tensor
    decoded_text: Optional[List[str]] = None
    input_logprobs: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    speculative_stats: Optional[SpeculativeStats] = None
