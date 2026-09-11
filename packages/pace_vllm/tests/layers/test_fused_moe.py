# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for PaceUnquantizedFusedMoEMethod."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import pace_vllm.model_executor.layers.fused_moe as pace_fused_moe
from pace_vllm.model_executor.layers.fused_moe import (
    PaceUnquantizedFusedMoEMethod,
    _resolve_activation_str,
)
from pace_vllm.model_executor.layers.utils import tpp_prepack_moe


def _make_method() -> PaceUnquantizedFusedMoEMethod:
    """Create an instance without vLLM's backend-dependent initialization."""
    return object.__new__(PaceUnquantizedFusedMoEMethod)


def _make_layer(**overrides) -> SimpleNamespace:
    attrs = dict(
        _pace_use_fused_moe=True,
        scoring_func="softmax",
        use_grouped_topk=False,
        renormalize=True,
        activation="silu",
        global_num_experts=8,
        num_experts=8,
        top_k=2,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestSelectExpertsImport(unittest.TestCase):
    def test_select_experts_is_callable(self) -> None:
        self.assertTrue(callable(pace_fused_moe.select_experts))


class TestPaceFusedMoeRegistration(unittest.TestCase):
    def test_oot_registration(self) -> None:
        from vllm.model_executor.custom_op import op_registry_oot
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        self.assertIs(
            op_registry_oot.get("UnquantizedFusedMoEMethod"),
            PaceUnquantizedFusedMoEMethod,
        )
        self.assertTrue(
            issubclass(PaceUnquantizedFusedMoEMethod, UnquantizedFusedMoEMethod)
        )


class TestApplyMonolithicGuards(unittest.TestCase):
    def _x_and_logits(self):
        x = torch.zeros(4, 16, dtype=torch.bfloat16)
        router_logits = torch.zeros(4, 8, dtype=torch.float32)
        return x, router_logits

    def test_raises_on_unknown_activation(self) -> None:
        method = _make_method()
        layer = _make_layer(activation="custom_glu")
        x, rl = self._x_and_logits()
        with self.assertRaises(NotImplementedError):
            method.apply_monolithic(layer, x, rl)

    def test_raises_on_non_default_gptoss_alpha(self) -> None:
        method = _make_method()
        layer = _make_layer(activation="swigluoai", gemm1_alpha=1.0)
        x, rl = self._x_and_logits()
        with self.assertRaises(NotImplementedError):
            method.apply_monolithic(layer, x, rl)

    def test_falls_through_when_pace_off(self) -> None:
        method = _make_method()
        layer = _make_layer(_pace_use_fused_moe=False)
        x, rl = self._x_and_logits()
        sentinel = torch.empty(0)
        with patch.object(
            pace_fused_moe.UnquantizedFusedMoEMethod,
            "apply_monolithic",
            return_value=sentinel,
        ) as mock_super:
            out = method.apply_monolithic(layer, x, rl)
        mock_super.assert_called_once_with(layer, x, rl, None)
        self.assertIs(out, sentinel)


class TestRouterDecision(unittest.TestCase):
    def test_softmax_binds_pace_cpp_router(self) -> None:
        method = _make_method()
        layer = _make_layer(scoring_func="softmax", renormalize=True)
        self.assertEqual(method._decide_router(layer), "pace::moe_route_softmax")

    def test_sigmoid_defers_to_vllm(self) -> None:
        method = _make_method()
        layer = _make_layer(scoring_func="sigmoid")
        self.assertIsNone(method._decide_router(layer))

    def test_grouped_topk_defers_to_vllm(self) -> None:
        method = _make_method()
        layer = _make_layer(use_grouped_topk=True)
        self.assertIsNone(method._decide_router(layer))

    def test_no_renormalize_defers_to_vllm(self) -> None:
        method = _make_method()
        layer = _make_layer(renormalize=False)
        self.assertIsNone(method._decide_router(layer))

    def test_custom_routing_defers_to_vllm(self) -> None:
        method = _make_method()
        layer = _make_layer(custom_routing_function=lambda **kw: None)
        self.assertIsNone(method._decide_router(layer))

    def test_routed_scaling_factor_defers_to_vllm(self) -> None:
        method = _make_method()
        layer = _make_layer(routed_scaling_factor=2.5)
        self.assertIsNone(method._decide_router(layer))


class TestProcessWeights(unittest.TestCase):
    def test_falls_back_for_unpackable_shape(self) -> None:
        method = _make_method()
        method.moe = SimpleNamespace(is_act_and_mul=True)
        w13 = torch.zeros(8, 200, 128, dtype=torch.bfloat16)
        w2 = torch.zeros(8, 128, 100, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w13_weight=SimpleNamespace(data=w13),
            w2_weight=SimpleNamespace(data=w2),
            prefix="model.layers.0.mlp.experts",
        )
        with patch.object(
            pace_fused_moe.UnquantizedFusedMoEMethod,
            "process_weights_after_loading",
        ) as mock_super:
            method.process_weights_after_loading(layer)
        mock_super.assert_called_once_with(layer)
        self.assertFalse(layer._pace_use_fused_moe)

    def test_gptoss_glu_uses_interleaved_split(self) -> None:
        method = _make_method()
        method.moe = SimpleNamespace(is_act_and_mul=True)
        e, inter, hidden = 2, 64, 128
        w13 = torch.randn(e, 2 * inter, hidden, dtype=torch.bfloat16)
        w2 = torch.randn(e, hidden, inter, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w13_weight=SimpleNamespace(data=w13),
            w2_weight=SimpleNamespace(data=w2),
            activation="swigluoai",
            prefix="x",
        )
        method.process_weights_after_loading(layer)
        self.assertTrue(layer._pace_use_fused_moe)
        self.assertTrue(
            torch.equal(
                layer._pace_wt_gate, tpp_prepack_moe(w13[:, 0::2, :].contiguous())
            )
        )
        self.assertTrue(
            torch.equal(
                layer._pace_wt_up, tpp_prepack_moe(w13[:, 1::2, :].contiguous())
            )
        )

    def test_silu_uses_contiguous_split(self) -> None:
        method = _make_method()
        method.moe = SimpleNamespace(is_act_and_mul=True)
        e, inter, hidden = 2, 64, 128
        w13 = torch.randn(e, 2 * inter, hidden, dtype=torch.bfloat16)
        w2 = torch.randn(e, hidden, inter, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w13_weight=SimpleNamespace(data=w13),
            w2_weight=SimpleNamespace(data=w2),
            activation="silu",
            prefix="x",
        )
        method.process_weights_after_loading(layer)
        self.assertTrue(layer._pace_use_fused_moe)
        self.assertTrue(
            torch.equal(
                layer._pace_wt_gate, tpp_prepack_moe(w13[:, :inter, :].contiguous())
            )
        )
        self.assertTrue(
            torch.equal(
                layer._pace_wt_up, tpp_prepack_moe(w13[:, inter:, :].contiguous())
            )
        )

    def test_unknown_activation_raises_loud(self) -> None:
        method = _make_method()
        method.moe = SimpleNamespace(is_act_and_mul=True)
        e, inter, hidden = 2, 64, 128
        layer = SimpleNamespace(
            w13_weight=SimpleNamespace(
                data=torch.randn(e, 2 * inter, hidden, dtype=torch.bfloat16)
            ),
            w2_weight=SimpleNamespace(
                data=torch.randn(e, hidden, inter, dtype=torch.bfloat16)
            ),
            activation="mystery_glu",
            prefix="x",
        )
        with self.assertRaises(NotImplementedError):
            method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "_pace_use_fused_moe"))

    def test_non_gated_moe_falls_back(self) -> None:
        method = _make_method()
        method.moe = SimpleNamespace(is_act_and_mul=False)
        layer = SimpleNamespace(prefix="model.layers.0.mlp.experts")
        with patch.object(
            pace_fused_moe.UnquantizedFusedMoEMethod,
            "process_weights_after_loading",
        ) as mock_super:
            method.process_weights_after_loading(layer)
        mock_super.assert_called_once_with(layer)
        self.assertFalse(layer._pace_use_fused_moe)


class TestGateUpLayout(unittest.TestCase):
    def test_gptoss_glu_is_interleaved(self) -> None:
        self.assertTrue(
            PaceUnquantizedFusedMoEMethod._gate_up_interleaved("gptoss_glu")
        )

    def test_contiguous_activations(self) -> None:
        for act in ("silu", "gelu", "relu"):
            with self.subTest(act=act):
                self.assertFalse(
                    PaceUnquantizedFusedMoEMethod._gate_up_interleaved(act)
                )

    def test_unknown_activation_raises(self) -> None:
        for act in ("mystery_glu", "swiglustep", ""):
            with self.subTest(act=act):
                with self.assertRaises(NotImplementedError):
                    PaceUnquantizedFusedMoEMethod._gate_up_interleaved(act)


class TestResolveActivation(unittest.TestCase):
    def test_swigluoai_maps_to_gptoss_glu(self) -> None:
        layer = SimpleNamespace(activation="swigluoai")
        self.assertEqual(_resolve_activation_str(layer), "gptoss_glu")

    def test_swigluoai_enum_maps_to_gptoss_glu(self) -> None:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        layer = SimpleNamespace(activation=MoEActivation.SWIGLUOAI)
        self.assertEqual(_resolve_activation_str(layer), "gptoss_glu")

    def test_passthrough_activation(self) -> None:
        layer = SimpleNamespace(activation="silu")
        self.assertEqual(_resolve_activation_str(layer), "silu")


if __name__ == "__main__":
    unittest.main()
