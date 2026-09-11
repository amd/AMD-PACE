# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the MoE engine + router ops (torch.ops.pace.libxsmm_moe_engine,
torch.ops.pace.moe_route_softmax)."""

import torch
import torch.nn.functional as F
import pace  # noqa: F401
from pace.ops.backends.tpp import _pack_weight_for_block_size
from torch.testing._internal.common_utils import TestCase

BLOCK_SIZE = 32
SCALE = 0.01
BF16_ATOL = 1e-3
BF16_RTOL = 1.6e-2


def _pack(w):
    return _pack_weight_for_block_size(w, BLOCK_SIZE)


def _gptoss_glu(gate, up, alpha=1.702, limit=7.0):
    """GptOss activation: gate.clamp(7) * sigmoid(gate*alpha), (up.clamp(±7)+1) * glu."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def _reference_engine(
    hidden,
    topk_indices,
    routing_weights,
    wg_2d_list,
    wu_2d_list,
    wd_2d_list,
    gb_list,
    ub_list,
    db_list,
    activation,
    num_experts,
    gated,
):
    """Reference for the routing-agnostic engine: gather/FFN/scatter given a
    precomputed (topk_indices, routing_weights) contract."""
    is_gptoss = activation == "gptoss_glu"
    act_fn = (
        {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}.get(activation)
        if not is_gptoss
        else None
    )
    topk = topk_indices.size(1)

    output = torch.zeros_like(hidden)
    for e in range(num_experts):
        token_mask = (topk_indices == e).any(dim=-1)
        if not token_mask.any():
            continue
        x = hidden[token_mask]

        if is_gptoss:
            gate_out = F.linear(x, wg_2d_list[e], gb_list[e] if gb_list else None)
            up_out = F.linear(x, wu_2d_list[e], ub_list[e] if ub_list else None)
            inter = _gptoss_glu(gate_out, up_out)
        elif gated:
            gate_out = F.linear(x, wg_2d_list[e], gb_list[e] if gb_list else None)
            up_out = F.linear(x, wu_2d_list[e], ub_list[e] if ub_list else None)
            inter = act_fn(gate_out) * up_out
        else:
            up_out = F.linear(x, wu_2d_list[e], ub_list[e] if ub_list else None)
            inter = act_fn(up_out)

        expert_out = F.linear(inter, wd_2d_list[e], db_list[e] if db_list else None)

        for k_idx in range(topk):
            k_mask = topk_indices[token_mask, k_idx] == e
            if not k_mask.any():
                continue
            w = routing_weights[token_mask, k_idx][k_mask].unsqueeze(-1)
            orig_idx = torch.where(token_mask)[0][k_mask]
            output[orig_idx] += expert_out[k_mask] * w.to(expert_out.dtype)

    return output


class TestFusedMoE(TestCase):
    def _make_inputs(
        self,
        num_tokens,
        hidden_size,
        inter_size,
        num_experts,
        topk,
        gated=True,
        bias=False,
    ):
        torch.manual_seed(42)
        hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16) * SCALE
        router = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16) * SCALE

        wg_2d, wu_2d, wd_2d = [], [], []
        wg_packed, wu_packed, wd_packed = [], [], []
        gb, ub, db = [], [], []

        for _ in range(num_experts):
            if gated:
                g = torch.randn(inter_size, hidden_size, dtype=torch.bfloat16) * SCALE
                wg_2d.append(g)
                wg_packed.append(_pack(g))
            u = torch.randn(inter_size, hidden_size, dtype=torch.bfloat16) * SCALE
            d = torch.randn(hidden_size, inter_size, dtype=torch.bfloat16) * SCALE
            wu_2d.append(u)
            wd_2d.append(d)
            wu_packed.append(_pack(u))
            wd_packed.append(_pack(d))
            if bias:
                if gated:
                    gb.append(torch.randn(inter_size, dtype=torch.bfloat16) * SCALE)
                ub.append(torch.randn(inter_size, dtype=torch.bfloat16) * SCALE)
                db.append(torch.randn(hidden_size, dtype=torch.bfloat16) * SCALE)

        wt_gate = torch.stack(wg_packed) if gated else None
        wt_up = torch.stack(wu_packed)
        wt_down = torch.stack(wd_packed)
        gate_bias_t = torch.stack(gb) if bias and gated else None
        up_bias_t = torch.stack(ub) if bias else None
        down_bias_t = torch.stack(db) if bias else None

        return (
            hidden,
            router,
            wt_gate,
            wt_up,
            wt_down,
            gate_bias_t,
            up_bias_t,
            down_bias_t,
            wg_2d,
            wu_2d,
            wd_2d,
            gb if bias else None,
            ub if bias else None,
            db if bias else None,
        )

    def _run_test(
        self,
        num_tokens,
        hidden,
        inter,
        experts,
        topk,
        activation="silu",
        gated=True,
        bias=False,
    ):
        (
            h,
            r,
            wg_p,
            wu_p,
            wd_p,
            gb_p,
            ub_p,
            db_p,
            wg_2d,
            wu_2d,
            wd_2d,
            gb_l,
            ub_l,
            db_l,
        ) = self._make_inputs(num_tokens, hidden, inter, experts, topk, gated, bias)

        topk_v, topk_i = torch.topk(r.float(), topk, dim=-1, largest=True, sorted=True)
        ids = topk_i.to(torch.int64)
        weights = torch.softmax(topk_v, dim=-1).to(torch.float32)

        output = torch.zeros(h.size(0), h.size(1), dtype=torch.bfloat16)
        torch.ops.pace.libxsmm_moe_engine(
            output,
            h,
            ids,
            weights,
            wg_p,
            wu_p,
            wd_p,
            gb_p,
            ub_p,
            db_p,
            activation,
            experts,
        )
        ref = _reference_engine(
            h,
            ids,
            weights,
            wg_2d,
            wu_2d,
            wd_2d,
            gb_l,
            ub_l,
            db_l,
            activation,
            experts,
            gated,
        )
        self.assertEqual(output.shape, ref.shape)
        torch.testing.assert_close(
            output.float(), ref.float(), atol=BF16_ATOL, rtol=BF16_RTOL
        )

    def test_silu_gated(self):
        self._run_test(16, 128, 256, 4, 2, "silu")

    def test_gelu_gated(self):
        self._run_test(8, 128, 256, 4, 1, "gelu")

    def test_relu_gated(self):
        self._run_test(8, 128, 256, 4, 2, "relu")

    def test_single_token(self):
        self._run_test(1, 128, 256, 4, 1, "silu")

    def test_topk_1(self):
        self._run_test(32, 128, 256, 8, 1, "silu")

    def test_topk_2(self):
        self._run_test(32, 128, 256, 8, 2, "silu")

    def test_gptoss_glu(self):
        self._run_test(16, 128, 256, 4, 2, "gptoss_glu")

    def test_gptoss_glu_single_token(self):
        self._run_test(1, 128, 256, 4, 1, "gptoss_glu")

    def test_gptoss_glu_16experts(self):
        self._run_test(32, 128, 256, 16, 2, "gptoss_glu")

    def test_output_shape(self):
        h, r, wg_p, wu_p, wd_p, gb_p, ub_p, db_p, *_ = self._make_inputs(
            32, 128, 256, 8, 2
        )
        ids, weights = torch.ops.pace.moe_route_softmax(r, 2)
        output = torch.zeros(h.size(0), h.size(1), dtype=torch.bfloat16)
        torch.ops.pace.libxsmm_moe_engine(
            output, h, ids, weights, wg_p, wu_p, wd_p, gb_p, ub_p, db_p, "silu", 8
        )
        self.assertEqual(output.shape, h.shape)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_silu_gated_with_bias(self):
        self._run_test(16, 128, 256, 4, 2, "silu", gated=True, bias=True)

    def test_gelu_gated_with_bias(self):
        self._run_test(8, 128, 256, 4, 1, "gelu", gated=True, bias=True)

    def test_gptoss_glu_with_bias(self):
        self._run_test(16, 128, 256, 4, 2, "gptoss_glu", gated=True, bias=True)

    def test_silu_with_bias_topk1(self):
        self._run_test(32, 128, 256, 8, 1, "silu", gated=True, bias=True)

    def test_silu_non_gated(self):
        self._run_test(16, 128, 256, 4, 2, "silu", gated=False)

    def test_gelu_non_gated(self):
        self._run_test(8, 128, 256, 4, 1, "gelu", gated=False)

    def test_relu_non_gated(self):
        self._run_test(8, 128, 256, 4, 2, "relu", gated=False)

    def test_silu_non_gated_with_bias(self):
        self._run_test(16, 128, 256, 4, 2, "silu", gated=False, bias=True)

    # 512 tokens over 4 experts forces the expert-sequential path.
    def test_seq_path_silu(self):
        self._run_test(512, 128, 256, 4, 2, "silu")

    def test_seq_path_gelu(self):
        self._run_test(512, 128, 256, 4, 1, "gelu")

    def test_seq_path_gptoss_glu(self):
        self._run_test(512, 128, 256, 4, 2, "gptoss_glu")

    def test_seq_path_with_bias(self):
        self._run_test(512, 128, 256, 4, 2, "silu", gated=True, bias=True)

    def test_seq_path_non_gated(self):
        self._run_test(512, 128, 256, 4, 2, "silu", gated=False)

    def test_router_matches_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for topk in (1, 2, 4):
                with self.subTest(dtype=dtype, topk=topk):
                    logits = (torch.randn(24, 32) * 4.0).to(dtype)
                    ids, w = torch.ops.pace.moe_route_softmax(logits, topk)
                    self.assertEqual(ids.shape, (24, topk))
                    self.assertEqual(w.shape, (24, topk))
                    tv, ti = torch.topk(logits.float(), topk, dim=-1)
                    ref_w = torch.softmax(tv, dim=-1)
                    got = {
                        (t, int(ids[t, k])): float(w[t, k])
                        for t in range(24)
                        for k in range(topk)
                    }
                    exp = {
                        (t, int(ti[t, k])): float(ref_w[t, k])
                        for t in range(24)
                        for k in range(topk)
                    }
                    self.assertEqual(set(got), set(exp))
                    for key in got:
                        self.assertAlmostEqual(got[key], exp[key], delta=2e-2)

    def test_engine_rejects_out_of_range_ids(self):
        h, r, wg_p, wu_p, wd_p, gb_p, ub_p, db_p, *_ = self._make_inputs(
            8, 128, 256, 4, 2
        )
        bad_ids = torch.full((8, 2), 99, dtype=torch.int64)
        weights = torch.ones(8, 2, dtype=torch.float32)
        output = torch.zeros(8, 128, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            torch.ops.pace.libxsmm_moe_engine(
                output,
                h,
                bad_ids,
                weights,
                wg_p,
                wu_p,
                wd_p,
                gb_p,
                ub_p,
                db_p,
                "silu",
                4,
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
