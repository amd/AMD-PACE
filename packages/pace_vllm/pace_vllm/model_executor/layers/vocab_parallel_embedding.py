# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Pace OOTs for `ParallelLMHead` and `VocabParallelEmbedding`.

Two LM-head shapes show up in vLLM CPU models:
  - Dedicated `ParallelLMHead` (Llama / Gemma / GPT-OSS / ...).
  - `lm_head = self.model.embed_tokens` aliased onto the input embedding
    (Qwen2/3, OLMo, GLM4, OPT, Starcoder2, ...). On these models
    `lm_head` is a `VocabParallelEmbedding`, not a `ParallelLMHead`.

Both subclasses share `PaceUnquantizedEmbeddingMethod`: pre-pack at
`process_weights_after_loading`, route `apply()` through libxsmm when
the pack succeeds, fall through to vLLM's stock CPU GEMM otherwise.
The original weight is preserved either way so the gather path used
by tied input embeddings keeps working.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from pace_vllm.model_executor.layers.utils import tpp_prepack

logger = init_logger("pace_vllm.model_executor.layers.vocab_parallel_embedding")


class PaceUnquantizedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Eager-pack LM-head matmul through `torch.ops.pace.libxsmmlinear_plain`,
    preserving `layer.weight` for tied-embedding models."""

    supports_pre_processed_weights = False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        packed = tpp_prepack(layer.weight.data)
        if packed is None:
            layer._pace_use_tpp = False
            logger.info(
                "pace-vllm: LM head shape %s dtype %s not TPP-packable; "
                "using stock CPU dispatch.",
                tuple(layer.weight.shape),
                layer.weight.dtype,
            )
            # Stock dispatch needs cpu_linear set up; only spend the work
            # in the fallback branch.
            super().process_weights_after_loading(layer)
            return

        layer._pace_packed_w = packed
        layer._pace_use_tpp = True
        logger.info(
            "pace-vllm: LM head packed into TPP layout (kept original "
            "weight for tied-embedding compatibility; packed_shape=%s).",
            tuple(packed.shape),
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(layer, "_pace_use_tpp", False):
            return super().apply(layer, x, bias)
        # libXSMM op expects 3D; pass 3D through, flatten any other rank.
        orig_shape = x.shape[:-1]
        x3d = x if x.dim() == 3 else x.reshape(-1, x.shape[-1]).unsqueeze(0)
        out = torch.ops.pace.libxsmmlinear_plain(x3d, layer._pace_packed_w, bias)
        return out.view(*orig_shape, -1)


@ParallelLMHead.register_oot
class PaceParallelLMHead(ParallelLMHead):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            self.quant_method = PaceUnquantizedEmbeddingMethod()


@VocabParallelEmbedding.register_oot
class PaceVocabParallelEmbedding(VocabParallelEmbedding):
    """OOT covering the `lm_head = self.model.embed_tokens` aliasing
    pattern (Qwen2/3, OLMo, GLM4, OPT, Starcoder2). Uses the same
    eager-pack method as `PaceParallelLMHead` -- input-embedding-only
    layers pay one prepack reshape's worth of memory + work, in
    exchange for a single code path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            self.quant_method = PaceUnquantizedEmbeddingMethod()


__all__ = [
    "PaceUnquantizedEmbeddingMethod",
    "PaceParallelLMHead",
    "PaceVocabParallelEmbedding",
]
