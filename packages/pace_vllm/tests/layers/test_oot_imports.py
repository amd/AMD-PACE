# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests that the pace OOT subclass modules are importable and that the
PluggableLayer / CustomOp `register_oot` substitutions actually take effect on
the in-tree vLLM classes (so model construction picks up the pace forwards)."""

from __future__ import annotations

import unittest

import torch

import pace_vllm
from pace_vllm.v1.worker.cpu_worker import _register_pace_oots

# OOT registration mutates global PluggableLayer / CustomOp state; load the
# native lib + run register_oot once for the whole test module.
pace_vllm._load_pace_native()
_register_pace_oots()


class TestLinearOOTs(unittest.TestCase):
    """Each pace Linear subclass is a `register_oot` of its in-tree vLLM
    counterpart, so `register_oot` records the pace class as the substitute."""

    def test_qkv_parallel_linear_substitution(self) -> None:
        from pace_vllm.model_executor.layers.linear import (
            PaceQKVParallelLinear,
            PaceUnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear,
            UnquantizedLinearMethod,
        )

        self.assertTrue(issubclass(PaceQKVParallelLinear, QKVParallelLinear))
        self.assertTrue(
            issubclass(PaceUnquantizedLinearMethod, UnquantizedLinearMethod)
        )
        self.assertFalse(PaceUnquantizedLinearMethod.supports_pre_processed_weights)

    def test_all_five_linear_subclasses_are_registered(self) -> None:
        from pace_vllm.model_executor.layers.linear import (
            PaceColumnParallelLinear,
            PaceMergedColumnParallelLinear,
            PaceQKVParallelLinear,
            PaceReplicatedLinear,
            PaceRowParallelLinear,
        )
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            ReplicatedLinear,
            RowParallelLinear,
        )

        pairs = [
            (PaceQKVParallelLinear, QKVParallelLinear),
            (PaceMergedColumnParallelLinear, MergedColumnParallelLinear),
            (PaceRowParallelLinear, RowParallelLinear),
            (PaceColumnParallelLinear, ColumnParallelLinear),
            (PaceReplicatedLinear, ReplicatedLinear),
        ]
        for pace_cls, vllm_cls in pairs:
            with self.subTest(pace_cls=pace_cls.__name__):
                self.assertTrue(issubclass(pace_cls, vllm_cls))


class TestParallelLMHeadOOT(unittest.TestCase):
    def test_parallel_lm_head_substitution(self) -> None:
        from pace_vllm.model_executor.layers.vocab_parallel_embedding import (
            PaceParallelLMHead,
            PaceUnquantizedEmbeddingMethod,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            UnquantizedEmbeddingMethod,
        )

        self.assertTrue(issubclass(PaceParallelLMHead, ParallelLMHead))
        self.assertTrue(
            issubclass(PaceUnquantizedEmbeddingMethod, UnquantizedEmbeddingMethod)
        )
        self.assertFalse(PaceUnquantizedEmbeddingMethod.supports_pre_processed_weights)

    def test_vocab_parallel_embedding_substitution(self) -> None:
        # Aliased lm_head = embed_tokens (Qwen2/3, OLMo, GLM4, OPT, ...).
        # Same eager-pack method as PaceParallelLMHead -- one code path
        # for both LM-head shapes.
        from pace_vllm.model_executor.layers.vocab_parallel_embedding import (
            PaceUnquantizedEmbeddingMethod,
            PaceVocabParallelEmbedding,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        self.assertTrue(issubclass(PaceVocabParallelEmbedding, VocabParallelEmbedding))
        self.assertTrue(
            issubclass(PaceUnquantizedEmbeddingMethod, UnquantizedEmbeddingMethod)
        )


class TestPaceUnquantizedEmbeddingMethodFallback(unittest.TestCase):
    """Regression: OPT-125m loads with fp16 weights -> tpp_prepack returns
    None -> process_weights_after_loading must set up `layer.cpu_linear`
    via super so subsequent apply() calls have the stock CPU GEMM path.
    Previously raised `AttributeError: 'no attribute cpu_linear'`."""

    @staticmethod
    def _stub_layer(weight: torch.Tensor):
        # Tensor.data returns the underlying tensor, so a plain Tensor
        # works the same as nn.Parameter for both
        # `layer.weight.data` (tpp_prepack) and `layer.weight.shape /
        # .dtype` (log args).
        from types import SimpleNamespace

        return SimpleNamespace(weight=weight)

    def test_pack_failure_invokes_super_pwal(self) -> None:
        from unittest.mock import patch

        from pace_vllm.model_executor.layers.vocab_parallel_embedding import (
            PaceUnquantizedEmbeddingMethod,
        )

        # fp16 weight -> tpp_prepack returns None.
        weight = torch.randn(128, 192, dtype=torch.float16)
        layer = self._stub_layer(weight)
        method = PaceUnquantizedEmbeddingMethod()

        with patch.object(
            PaceUnquantizedEmbeddingMethod.__bases__[0],
            "process_weights_after_loading",
        ) as mock_pwal:
            method.process_weights_after_loading(layer)

        mock_pwal.assert_called_once_with(layer)
        self.assertFalse(layer._pace_use_tpp)

    def test_apply_falls_back_to_super_when_not_packed(self) -> None:
        from unittest.mock import patch

        from pace_vllm.model_executor.layers.vocab_parallel_embedding import (
            PaceUnquantizedEmbeddingMethod,
        )

        weight = torch.randn(128, 192, dtype=torch.float16)
        layer = self._stub_layer(weight)
        layer._pace_use_tpp = False  # what process_weights_after_loading would set

        sentinel_out = torch.empty(7, 128, dtype=torch.float16)
        with patch.object(
            PaceUnquantizedEmbeddingMethod.__bases__[0],
            "apply",
            return_value=sentinel_out,
        ) as mock_apply:
            x = torch.randn(7, 192, dtype=torch.float16)
            out = PaceUnquantizedEmbeddingMethod().apply(layer, x)

        mock_apply.assert_called_once()
        self.assertIs(out, sentinel_out)


class TestRMSNormOOTs(unittest.TestCase):
    def test_rmsnorm_substitution(self) -> None:
        from pace_vllm.model_executor.layers.layernorm import (
            PaceGemmaRMSNorm,
            PaceRMSNorm,
        )
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

        self.assertTrue(issubclass(PaceRMSNorm, RMSNorm))
        self.assertTrue(issubclass(PaceGemmaRMSNorm, GemmaRMSNorm))


class TestRegisterPaceOOTsIdempotent(unittest.TestCase):
    """Calling `_register_pace_oots()` multiple times must be a no-op."""

    def test_idempotent(self) -> None:
        # Already called at module import; calling again should not raise.
        _register_pace_oots()
        _register_pace_oots()


if __name__ == "__main__":
    unittest.main()
