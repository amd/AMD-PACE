# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Front-facing API for encoder models (embeddings + classification).

``EmbeddingModel`` is to encoder models what ``LLMModel`` is to decoder
models: it loads weights + tokenizer once and exposes a single-forward
``encode()`` (sentence embeddings) and ``classify()`` (sequence-classification
logits). There is no KV cache, sampler, or decode loop -- encoders attend
bidirectionally and emit all token states in one pass.

Example:
    >>> from pace.llm import EmbeddingModel
    >>> model = EmbeddingModel("BAAI/bge-small-en-v1.5", pooling="cls")
    >>> emb = model.encode(["hello world", "how are you"])  # [2, hidden]
"""

import os
from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, PretrainedConfig, PreTrainedTokenizerBase

from pace.llm.attention import AttentionBackendType
from pace.llm.configs import OperatorConfig
from pace.llm.models.model_utils import get_tokenizer, init_model
from pace.llm.ops import LLMBackendType, LLMOperatorType
from pace.llm.outputs import (
    ClassifierOutput,
    EmbeddingOutput,
    TokenClassifierOutput,
)
from pace.llm.pooling import PoolingType, pool
from pace.utils.logging import PACE_LLM_WARNING

TextInput = Union[str, List[str], BatchEncoding, dict]


def default_encoder_opconfig() -> OperatorConfig:
    """Recommended operator backends for encoder (BERT / ModernBERT) models.

    Routes the QKV / output / MLP linears through the TPP (libXSMM) bf16 GEMM
    kernels and LayerNorm through the JIT bf16 kernel; attention uses the
    contiguous JIT backend (in bidirectional mode for encoders).

    JIT Linear and JIT attention run in float32 as well as bf16. TPP GEMM and
    JIT LayerNorm are bfloat16-only; in float32 those two ops fall back to
    NATIVE through the registry. The per-layer classification / pooler heads
    in the model definitions stay on NATIVE because their output dims
    (num_labels) are not TPP-packable.
    """
    op = OperatorConfig()
    op[LLMOperatorType.QKVProjection] = LLMBackendType.TPP
    op[LLMOperatorType.OutProjection] = LLMBackendType.TPP
    op[LLMOperatorType.MLP] = LLMBackendType.TPP
    op[LLMOperatorType.Norm] = LLMBackendType.JIT
    op[LLMOperatorType.Attention] = AttentionBackendType.JIT
    return op


def finalize_encoder_opconfig(
    opconfig: Optional[OperatorConfig],
) -> OperatorConfig:
    """Finalize an encoder ``OperatorConfig``, filling unset ops with encoder defaults.

    ``OperatorConfig.finalize()`` fills missing keys with *decoder* defaults
    (Slab attention among them). A partially filled config would otherwise
    pick those up. Encoder defaults are applied first so a caller can override
    just the ops they care about.
    """
    if opconfig is not None and getattr(opconfig, "_finalized", False):
        return opconfig
    if opconfig is None:
        opconfig = default_encoder_opconfig()
    else:
        for key, value in default_encoder_opconfig().items():
            opconfig.setdefault(key, value)
    # No cache_type: encoders are cacheless, so the cache/attention
    # compatibility check is skipped.
    return opconfig.finalize()


class EmbeddingModel(object):
    """Wraps an encoder model and exposes ``encode`` / ``classify``.

    Args:
        model_name_or_path: Path or identifier of the encoder checkpoint
            (e.g. a BERT/bge or ModernBERT/mmBERT model).
        tokenizer_name_or_path: Optional tokenizer override.
        dtype: Model dtype (default bf16).
        opconfig: Operator-backend mapping. Finalized WITHOUT a cache type
            (encoders have no KV cache, so cache-type validation is bypassed).
            Defaults to :func:`default_encoder_opconfig` (TPP linear + JIT
            LayerNorm + JIT bidirectional attention). Unset ops on a partial
            config are filled with those encoder defaults, not decoder ones.
            JIT Linear and JIT attention run in float32; TPP GEMM and JIT
            LayerNorm need bfloat16. Pass an all-NATIVE OperatorConfig to run
            purely in PyTorch.
        pooling: Default token pooling for ``encode`` (cls / mean / last).
        normalize: Default L2-normalization for ``encode`` (bge default True).
        max_length: Optional tokenizer truncation length.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        opconfig: Optional[OperatorConfig] = None,
        pooling: Union[str, PoolingType] = PoolingType.CLS,
        normalize: bool = True,
        max_length: Optional[int] = None,
        disable_tqdm: bool = False,
    ):
        opconfig = finalize_encoder_opconfig(opconfig)

        if dtype != torch.bfloat16:
            PACE_LLM_WARNING(
                f"EmbeddingModel loaded in {dtype}; TPP GEMM and JIT LayerNorm "
                "are bfloat16-only and fall back to NATIVE. JIT Linear and JIT "
                "attention still run in float32. Pass dtype=torch.bfloat16 to "
                "use the TPP kernels."
            )

        self.model = init_model(
            model_name_or_path,
            dtype,
            opconfig,
            disable_tqdm=disable_tqdm,
            encoder=True,
        )
        self.tokenizer = get_tokenizer(model_name_or_path, tokenizer_name_or_path)
        self.config: PretrainedConfig = self.model.get_config()
        self.pooling = PoolingType.resolve(pooling)
        self.normalize = normalize
        self.max_length = max_length

    def _tokenize(self, texts: TextInput) -> BatchEncoding:
        if isinstance(texts, (BatchEncoding, dict)):
            return texts
        if isinstance(texts, str):
            texts = [texts]
        kwargs = dict(padding=True, truncation=True, return_tensors="pt")
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        return self.tokenizer(texts, **kwargs)

    def _run(self, texts: TextInput):
        enc = self._tokenize(texts)
        return (
            self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                token_type_ids=enc.get("token_type_ids"),
            ),
            enc,
        )

    @torch.inference_mode()
    def encode(
        self,
        texts: TextInput,
        pooling: Optional[Union[str, PoolingType]] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return pooled sentence embeddings, shape ``[batch_size, hidden]``."""
        output, enc = self._run(texts)
        if not isinstance(output, EmbeddingOutput):
            raise TypeError(
                "encode() requires an embedding model; loaded model returned "
                f"{type(output).__name__}. Use classify() for classifiers."
            )
        return pool(
            output.last_hidden_state,
            pooling if pooling is not None else self.pooling,
            attention_mask=enc.get("attention_mask"),
            normalize=self.normalize if normalize is None else normalize,
        )

    @torch.inference_mode()
    def classify(
        self, texts: TextInput, return_probabilities: bool = True
    ) -> ClassifierOutput:
        """Return classification logits (and optional probabilities / argmax)."""
        output, _ = self._run(texts)
        if not isinstance(output, ClassifierOutput):
            raise TypeError(
                "classify() requires a *ForSequenceClassification model; loaded "
                f"model returned {type(output).__name__}. Use encode() instead."
            )
        logits = output.logits
        probabilities = None
        predicted = None
        if return_probabilities:
            probabilities = self._probabilities(logits)
        if logits.shape[-1] > 1:
            predicted = logits.argmax(dim=-1)
        return ClassifierOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class_ids=predicted,
        )

    @torch.inference_mode()
    def classify_tokens(
        self, texts: TextInput, return_probabilities: bool = True
    ) -> TokenClassifierOutput:
        """Return per-token classification logits (NER / PII).

        Requires a ``*ForTokenClassification`` model. ``logits`` has shape
        ``[batch, seq_len, num_labels]``; ``predicted_class_ids`` is the argmax
        per token, shape ``[batch, seq_len]``.
        """
        output, _ = self._run(texts)
        if not isinstance(output, TokenClassifierOutput):
            raise TypeError(
                "classify_tokens() requires a *ForTokenClassification model; "
                f"loaded model returned {type(output).__name__}. Use classify() "
                "for sequence classification or encode() for embeddings."
            )
        logits = output.logits
        probabilities = None
        predicted = None
        if return_probabilities:
            probabilities = torch.softmax(logits.float(), dim=-1)
        if logits.shape[-1] > 1:
            predicted = logits.argmax(dim=-1)
        return TokenClassifierOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class_ids=predicted,
        )

    def _probabilities(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        problem_type = getattr(self.config, "problem_type", None)
        num_labels = getattr(self.config, "num_labels", logits.shape[-1])
        logits_f = logits.float()
        if problem_type == "multi_label_classification":
            return torch.sigmoid(logits_f)
        if problem_type == "regression" or num_labels == 1:
            return None
        return torch.softmax(logits_f, dim=-1)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def get_config(self) -> PretrainedConfig:
        return self.config

    def __repr__(self):
        return self.model.__repr__()
