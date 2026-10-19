# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

from pace.llm.configs import OperatorConfig
from pace.ops.enum import BackendType, DataType
from pace.ops.linear import Linear
from pace.ops.moe import MoEConfig, MoEExperts
from pace.ops.moe_routing import MoERouter, RoutingType


class TestMoERouter(TestCase):
    """The MoERouter operator computes each scheme via its NATIVE backend."""

    def _route(self, logits, topk, routing=RoutingType.SOFTMAX, **kw):
        return MoERouter(
            topk=topk, routing=routing, backend_impl=BackendType.NATIVE, **kw
        )(logits)

    def test_softmax_matches_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 32) * 4.0
        ids, w = self._route(logits, 4, RoutingType.SOFTMAX)
        tv, ti = torch.topk(logits.float(), 4, dim=-1)
        self.assertEqual(ids, ti.to(torch.int64))
        self.assertEqual(w, torch.softmax(tv, dim=-1).float())

    def test_sigmoid_weights(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 16) * 4.0
        ids, w = self._route(logits, 2, RoutingType.SIGMOID, renormalize=False)
        tv = torch.gather(logits.float(), 1, ids)
        self.assertEqual(w, torch.sigmoid(tv).float())

    def test_grouped_sigmoid_selects_within_groups(self):
        # Group 1 (experts 4-7) dominates; with topk_group=1 every pick must
        # fall in it, proving selection is group-constrained (not a no-op).
        logits = torch.full((6, 16), -10.0)
        logits[:, 4:8] = 10.0
        ids, w = self._route(
            logits, 3, RoutingType.GROUPED_SIGMOID, num_expert_group=4, topk_group=1
        )
        self.assertEqual(ids.shape, (6, 3))
        self.assertTrue(bool(((ids >= 4) & (ids < 8)).all()))

    def test_router_backend_falls_back_to_native(self):
        router = MoERouter(topk=2, backend_impl=BackendType.TPP)
        self.assertIsNotNone(router.backend)
        ids, w = router(torch.randn(4, 8) * 4.0)
        self.assertEqual(ids.shape, (4, 2))

    def test_softmax_without_renormalization_matches_reference(self):
        torch.manual_seed(1)
        logits = torch.randn(16, 32) * 4.0
        ids, weights = self._route(logits, 4, RoutingType.SOFTMAX, renormalize=False)
        probabilities = torch.softmax(logits.float(), dim=-1)
        expected_weights, expected_ids = torch.topk(probabilities, 4, dim=-1)
        self.assertEqual(ids, expected_ids.to(torch.int64))
        self.assertEqual(weights, expected_weights, atol=1e-6, rtol=1e-6)

    def test_softmax_renormalize_flag_changes_weights(self):
        torch.manual_seed(2)
        logits = torch.randn(16, 32) * 4.0
        normalized_ids, normalized = self._route(
            logits, 4, RoutingType.SOFTMAX, renormalize=True
        )
        unnormalized_ids, unnormalized = self._route(
            logits, 4, RoutingType.SOFTMAX, renormalize=False
        )
        self.assertEqual(normalized_ids, unnormalized_ids)
        self.assertFalse(torch.allclose(normalized, unnormalized))
        self.assertEqual(normalized.sum(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6)
        self.assertTrue(bool((unnormalized.sum(dim=-1) < 1.0).all()))

    def test_tpp_softmax_without_renormalization_matches_reference(self):
        torch.manual_seed(3)
        logits = torch.randn(16, 32) * 4.0
        router = MoERouter(
            topk=4,
            routing=RoutingType.SOFTMAX,
            renormalize=False,
            backend_impl=BackendType.TPP,
        )
        ids, weights = router(logits)
        probabilities = torch.softmax(logits.float(), dim=-1)
        expected_weights, expected_ids = torch.topk(probabilities, 4, dim=-1)
        self.assertEqual(ids, expected_ids.to(torch.int64))
        self.assertEqual(weights, expected_weights, atol=1e-6, rtol=1e-6)

    def test_grouped_sigmoid_rejects_bad_group_config(self):
        logits = torch.randn(4, 15)
        with self.assertRaises(ValueError):
            self._route(logits, 2, RoutingType.GROUPED_SIGMOID, num_expert_group=4)
        with self.assertRaises(ValueError):
            self._route(
                logits[:, :16],
                2,
                RoutingType.GROUPED_SIGMOID,
                num_expert_group=4,
                topk_group=5,
            )


def _reference_moe(
    x, router, gate_w, up_w, down_w, topk, activation, gate, gb=None, ub=None, db=None
):
    """Independent per-token reference for the NATIVE fallback."""
    pw = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}.get(activation)
    tv, ti = torch.topk(router.float(), topk, dim=-1)
    rw = torch.softmax(tv, dim=-1)
    xf = x.float()
    out = torch.zeros_like(xf)
    for t in range(x.size(0)):
        for k in range(topk):
            e = ti[t, k].item()
            up = xf[t] @ up_w[e].float().t()
            if ub is not None:
                up = up + ub[e].float()
            if gate:
                g = xf[t] @ gate_w[e].float().t()
                if gb is not None:
                    g = g + gb[e].float()
                if activation == "gptoss_glu":
                    g = g.clamp(max=7.0)
                    u = up.clamp(min=-7.0, max=7.0)
                    inter = (u + 1) * (g * torch.sigmoid(g * 1.702))
                else:
                    inter = pw(g) * up
            else:
                inter = pw(up)
            o = inter @ down_w[e].float().t()
            if db is not None:
                o = o + db[e].float()
            out[t] += rw[t, k].item() * o
    return out


class TestMoEExperts(TestCase):
    def _build(
        self,
        H=64,
        I=128,
        E=8,
        topk=2,
        activation="silu",
        gate=True,
        bias=False,
        num_shared=0,
        shared_inter=0,
        routed_scaling=1.0,
        shared_gate=False,
    ):
        cfg = MoEConfig(
            hidden_size=H,
            intermediate_size=I,
            num_experts=E,
            topk=topk,
            activation=activation,
            gate=gate,
            bias=bias,
            num_shared_experts=num_shared,
            shared_intermediate_size=shared_inter,
            routed_scaling_factor=routed_scaling,
            shared_expert_gate=shared_gate,
        )
        moe = MoEExperts(cfg, backend_impl=BackendType.NATIVE)
        # Real models run bf16; cast the (fp32-by-default) module to match the
        # bf16 expert weights, as model.to(bfloat16) does in practice.
        moe = moe.to(torch.bfloat16)
        # The router Linear starts as torch.empty(); initialize it deterministically
        # so routing is stable (uninitialized memory may hold NaN/Inf and make these
        # tests flaky). Seed separately to leave the expert-weight RNG stream below
        # unchanged.
        torch.manual_seed(1)
        moe.router.weight.data.copy_(torch.randn_like(moe.router.weight) * 0.1)
        if moe.router.bias is not None:
            moe.router.bias.data.zero_()
        if moe.shared_gate is not None:
            moe.shared_gate.weight.data.copy_(
                torch.randn_like(moe.shared_gate.weight) * 0.1
            )
        torch.manual_seed(0)
        gate_w = [torch.randn(I, H, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
        up_w = [torch.randn(I, H, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
        down_w = [torch.randn(H, I, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
        gate_b = up_b = down_b = None
        if bias:
            gate_b = [torch.randn(I, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
            up_b = [torch.randn(I, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
            down_b = [torch.randn(H, dtype=torch.bfloat16) * 0.1 for _ in range(E)]
        for e in range(E):
            if gate:
                moe.gate_projs[e].weight.data.copy_(gate_w[e])
            moe.up_projs[e].weight.data.copy_(up_w[e])
            moe.down_projs[e].weight.data.copy_(down_w[e])
            if bias:
                if gate:
                    moe.gate_projs[e].bias.data.copy_(gate_b[e])
                moe.up_projs[e].bias.data.copy_(up_b[e])
                moe.down_projs[e].bias.data.copy_(down_b[e])
        return moe, gate_w, up_w, down_w, gate_b, up_b, down_b

    def test_router_and_experts_are_pace_ops(self):
        moe = self._build(E=4)[0]
        self.assertIsInstance(moe.router, Linear)
        self.assertIsInstance(moe.router_op, MoERouter)
        self.assertIsInstance(moe.up_projs[0], Linear)
        self.assertIsInstance(moe.down_projs[0], Linear)

    def test_native_fallback_matches_reference(self):
        cases = [
            ("silu", True),
            ("gelu", True),
            ("relu", False),
            ("gptoss_glu", True),
        ]
        for activation, gate in cases:
            with self.subTest(activation=activation, gate=gate):
                moe, gw, uw, dw, *_ = self._build(activation=activation, gate=gate)
                x = torch.randn(16, 64, dtype=torch.bfloat16) * 0.1
                router = moe.router(x)
                out = moe(x)
                ref = _reference_moe(x, router, gw, uw, dw, moe.topk, activation, gate)
                self.assertEqual(out.float(), ref, atol=2e-2, rtol=2e-2)

    def test_bias_path_matches_reference(self):
        moe, gw, uw, dw, gb, ub, db = self._build(
            activation="silu", gate=True, bias=True
        )
        x = torch.randn(16, 64, dtype=torch.bfloat16) * 0.1
        out = moe(x)
        ref = _reference_moe(
            x, moe.router(x), gw, uw, dw, moe.topk, "silu", True, gb, ub, db
        )
        self.assertEqual(out.float(), ref, atol=2e-2, rtol=2e-2)

    def test_shared_experts_added_to_routed(self):
        moe, gw, uw, dw, *_ = self._build(E=4, num_shared=2)
        torch.manual_seed(1)
        for p in moe.shared_mlps.parameters():
            p.data.normal_(0, 0.1)
        x = torch.randn(12, 64, dtype=torch.bfloat16) * 0.1
        out = moe(x)
        routed = _reference_moe(x, moe.router(x), gw, uw, dw, moe.topk, "silu", True)
        shared = sum(m(x).float() for m in moe.shared_mlps)
        self.assertEqual(out.float(), routed + shared, atol=2e-2, rtol=2e-2)

    def test_shared_experts_can_be_sigmoid_gated(self):
        moe, gw, uw, dw, *_ = self._build(E=4, num_shared=1, shared_gate=True)
        torch.manual_seed(1)
        for p in moe.shared_mlps.parameters():
            p.data.normal_(0, 0.1)
        x = torch.randn(12, 64, dtype=torch.bfloat16) * 0.1
        out = moe(x)
        routed = _reference_moe(x, moe.router(x), gw, uw, dw, moe.topk, "silu", True)
        shared = sum(m(x).float() for m in moe.shared_mlps)
        gate = torch.sigmoid(moe.shared_gate(x).float())
        self.assertEqual(out.float(), routed + gate * shared, atol=2e-2, rtol=2e-2)

    def test_gated_shared_experts_fold_with_the_gate_as_routing_weight(self):
        moe = self._build(E=4, num_shared=1, shared_gate=True)[0]
        self.assertTrue(moe.foldable_shared)

        captured = {}

        class _Recorder:
            def execute(self, output, hidden, topk_ids, topk_weights, *args):
                captured["ids"] = topk_ids
                captured["weights"] = topk_weights
                return output

        moe.backend = _Recorder()
        moe._folded_shared = True
        x = torch.randn(12, 64, dtype=torch.bfloat16) * 0.1

        moe(x)

        shared_ids = captured["ids"][:, -1]
        shared_w = captured["weights"][:, -1:]
        expected = torch.sigmoid(moe.shared_gate(x)).to(shared_w.dtype)
        self.assertEqual(shared_ids, torch.full_like(shared_ids, moe.num_experts))
        self.assertEqual(shared_w, expected)

    def test_shared_gate_is_built_only_when_shared_experts_are_gated(self):
        self.assertIsNone(self._build(E=4)[0].shared_gate)
        self.assertIsNone(self._build(E=4, num_shared=1)[0].shared_gate)

        gated = self._build(E=4, num_shared=1, shared_gate=True)[0]
        self.assertEqual(tuple(gated.shared_gate.weight.shape), (1, gated.hidden_size))

    def test_shared_experts_can_size_independently_of_routed_experts(self):
        moe, gw, uw, dw, *_ = self._build(E=4, num_shared=1, shared_inter=32)
        torch.manual_seed(1)
        for p in moe.shared_mlps.parameters():
            p.data.normal_(0, 0.1)

        self.assertEqual(moe.shared_mlps[0].down_proj.weight.shape[1], 32)
        self.assertFalse(moe.foldable_shared)

        x = torch.randn(12, 64, dtype=torch.bfloat16) * 0.1
        out = moe(x)
        routed = _reference_moe(x, moe.router(x), gw, uw, dw, moe.topk, "silu", True)
        shared = sum(m(x).float() for m in moe.shared_mlps)
        self.assertEqual(out.float(), routed + shared, atol=2e-2, rtol=2e-2)

    def test_routed_scaling_factor_scales_routed_not_shared(self):
        # routed_scaling_factor multiplies the routed output; shared experts are
        # added unscaled (DeepSeek convention).
        factor = 2.5
        moe, gw, uw, dw, *_ = self._build(E=4, num_shared=1, routed_scaling=factor)
        torch.manual_seed(1)
        for p in moe.shared_mlps.parameters():
            p.data.normal_(0, 0.1)
        x = torch.randn(12, 64, dtype=torch.bfloat16) * 0.1
        out = moe(x)
        routed = _reference_moe(x, moe.router(x), gw, uw, dw, moe.topk, "silu", True)
        shared = sum(m(x).float() for m in moe.shared_mlps)
        self.assertEqual(out.float(), factor * routed + shared, atol=3e-2, rtol=3e-2)

    def test_gptoss_glu_requires_gate(self):
        with self.assertRaises(ValueError):
            self._build(activation="gptoss_glu", gate=False)

    def test_unsupported_activation_raises(self):
        with self.assertRaises(ValueError):
            self._build(activation="tanh")

    def test_load_expert_weights_interleave_split(self):
        # gate_up_weight is interleaved [2*I, H]: even rows -> gate, odd -> up.
        moe = self._build(E=4)[0]
        H, I = moe.hidden_size, moe.intermediate_size
        gate_up = torch.randn(2 * I, H, dtype=torch.bfloat16) * 0.1
        moe.load_expert_weights(2, gate_up_weight=gate_up)
        self.assertEqual(moe.gate_projs[2].weight.data, gate_up[::2, :])
        self.assertEqual(moe.up_projs[2].weight.data, gate_up[1::2, :])


class TestMoEExpertsTPP(TestCase):
    """The TPP fused path matches the NATIVE reference."""

    def _build_pair(
        self,
        H=128,
        I=256,
        E=8,
        topk=2,
        activation="silu",
        gate=True,
        bias=False,
        num_shared=0,
        routed_scaling=1.0,
        tpp_backend=BackendType.TPP,
    ):
        cfg = MoEConfig(
            hidden_size=H,
            intermediate_size=I,
            num_experts=E,
            topk=topk,
            activation=activation,
            gate=gate,
            bias=bias,
            num_shared_experts=num_shared,
            routed_scaling_factor=routed_scaling,
        )
        native = MoEExperts(
            cfg, backend_impl=BackendType.NATIVE, dtype=DataType.BFLOAT16
        )
        tpp = MoEExperts(cfg, backend_impl=tpp_backend, dtype=DataType.BFLOAT16)
        native = native.to(torch.bfloat16)
        tpp = tpp.to(torch.bfloat16)
        torch.manual_seed(0)
        for p in native.parameters():
            p.data.normal_(0, 0.1)
        tpp.load_state_dict(native.state_dict())
        tpp.backend.preprocess(tpp)
        return native, tpp

    def test_tpp_matches_native(self):
        # "gelu_tanh" (Gemma4's MoE activation) has no dedicated TPP kernel; it
        # aliases to the one GELU, so parity with the exact tanh form holds only
        # to bf16 tolerance.
        for activation, gate in [
            ("silu", True),
            ("gelu", True),
            ("gelu_tanh", True),
            ("relu", False),
            ("gptoss_glu", True),
        ]:
            with self.subTest(activation=activation, gate=gate):
                native, tpp = self._build_pair(activation=activation, gate=gate)
                x = torch.randn(16, 128, dtype=torch.bfloat16) * 0.1
                self.assertEqual(
                    tpp(x).float(), native(x).float(), atol=3e-2, rtol=3e-2
                )

    def test_tpp_routed_scaling(self):
        native, tpp = self._build_pair(routed_scaling=2.5)
        x = torch.randn(16, 128, dtype=torch.bfloat16) * 0.1
        self.assertEqual(tpp(x).float(), native(x).float(), atol=3e-2, rtol=3e-2)

    def test_tpp_folds_shared_experts(self):
        native, tpp = self._build_pair(num_shared=2, routed_scaling=2.5)
        self.assertTrue(tpp._folded_shared)
        x = torch.randn(16, 128, dtype=torch.bfloat16) * 0.1
        self.assertEqual(tpp(x).float(), native(x).float(), atol=3e-2, rtol=3e-2)

    def test_default_operator_config_runs_fused_tpp(self):
        config = OperatorConfig()
        config.finalize()
        native, tpp = self._build_pair(tpp_backend=config.mlp)
        self.assertEqual(type(tpp.backend).__name__, "TPPFusedMoE")
        x = torch.randn(16, 128, dtype=torch.bfloat16) * 0.1
        self.assertEqual(tpp(x).float(), native(x).float(), atol=3e-2, rtol=3e-2)
