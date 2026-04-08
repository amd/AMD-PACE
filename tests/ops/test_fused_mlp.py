# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the fused MLP kernel (libxsmm_fused_mlp).

Usage:
    pip install --no-build-isolation -v --force .
    python -m pytest tests/ops/test_fused_mlp.py -v -s
"""

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
from hypothesis import given, settings
import hypothesis.strategies as st

import pace  # noqa: F401
from pace.ops.backends.tpp import _pack_weight_for_block_size

BLOCK_SIZE = 32
SCALE = 0.01
BF16_ATOL = 1e-3
BF16_RTOL = 1.6e-2


def _pack(w):
    return _pack_weight_for_block_size(w, BLOCK_SIZE)


def _ref_gated_mlp(src, wg, wu, wd, bg, bu, bd, act_fn):
    gate = F.linear(src, wg, bg)
    up = F.linear(src, wu, bu)
    inter = act_fn(gate) * up
    return F.linear(inter, wd, bd)


def _ref_plain_mlp(src, wu, wd, bu, bd, act_fn):
    up = F.linear(src, wu, bu)
    inter = act_fn(up)
    return F.linear(inter, wd, bd)


ACT_FNS = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}


class TestFusedMLPKernel(TestCase):
    """Direct torch.ops.pace.libxsmm_fused_mlp correctness tests."""

    @settings(deadline=None, max_examples=10)
    @given(
        dims=st.sampled_from(
            [
                {"K": 2048, "N": 8192},
                {"K": 4096, "N": 14336},
            ]
        ),
        M=st.sampled_from([1, 32, 128]),
        activation=st.sampled_from(["silu", "gelu", "relu"]),
    )
    def test_gated_no_bias(self, dims, M, activation):
        K, N = dims["K"], dims["N"]
        torch.manual_seed(42)
        src = torch.randn(1, M, K, dtype=torch.bfloat16) * SCALE
        wg = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wu = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wd = torch.randn(K, N, dtype=torch.bfloat16) * SCALE

        ref = _ref_gated_mlp(src, wg, wu, wd, None, None, None, ACT_FNS[activation])
        out = torch.ops.pace.libxsmm_fused_mlp(
            src, _pack(wg), _pack(wu), _pack(wd), None, None, None, activation
        )
        self.assertEqual(out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    @settings(deadline=None, max_examples=10)
    @given(
        M=st.sampled_from([1, 64]),
        gate_bias=st.booleans(),
        up_bias=st.booleans(),
        down_bias=st.booleans(),
    )
    def test_silu_bias_combos(self, M, gate_bias, up_bias, down_bias):
        K, N = 4096, 14336
        torch.manual_seed(42)
        src = torch.randn(1, M, K, dtype=torch.bfloat16) * SCALE
        wg = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wu = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wd = torch.randn(K, N, dtype=torch.bfloat16) * SCALE
        bg = torch.randn(N, dtype=torch.bfloat16) * SCALE if gate_bias else None
        bu = torch.randn(N, dtype=torch.bfloat16) * SCALE if up_bias else None
        bd = torch.randn(K, dtype=torch.bfloat16) * SCALE if down_bias else None

        ref = _ref_gated_mlp(src, wg, wu, wd, bg, bu, bd, F.silu)
        out = torch.ops.pace.libxsmm_fused_mlp(
            src, _pack(wg), _pack(wu), _pack(wd), bg, bu, bd, "silu"
        )
        self.assertEqual(out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    @settings(deadline=None, max_examples=10)
    @given(
        M=st.sampled_from([1, 32]),
        activation=st.sampled_from(["silu", "gelu", "relu"]),
    )
    def test_plain_no_gate(self, M, activation):
        K, N = 2048, 8192
        torch.manual_seed(42)
        src = torch.randn(1, M, K, dtype=torch.bfloat16) * SCALE
        wu = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wd = torch.randn(K, N, dtype=torch.bfloat16) * SCALE

        ref = _ref_plain_mlp(src, wu, wd, None, None, ACT_FNS[activation])
        out = torch.ops.pace.libxsmm_fused_mlp(
            src, None, _pack(wu), _pack(wd), None, None, None, activation
        )
        self.assertEqual(out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_2d_input(self):
        torch.manual_seed(42)
        K, N, M = 4096, 14336, 8
        src = torch.randn(M, K, dtype=torch.bfloat16) * SCALE
        wg = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wu = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wd = torch.randn(K, N, dtype=torch.bfloat16) * SCALE

        ref = _ref_gated_mlp(
            src.unsqueeze(0), wg, wu, wd, None, None, None, F.silu
        ).squeeze(0)
        out = torch.ops.pace.libxsmm_fused_mlp(
            src, _pack(wg), _pack(wu), _pack(wd), None, None, None, "silu"
        )
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_invalid_dtype(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.float32)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "only supports bfloat16"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, None, None, "silu"
            )

    def test_invalid_activation(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.bfloat16)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, None, None, "tanh"
            )

    def test_invalid_input_dim(self):
        K, N = 2048, 8192
        src = torch.randn(K, dtype=torch.bfloat16)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "expected input to be 2D or 3D"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, None, None, "silu"
            )

    def test_invalid_weight_dim(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.bfloat16)
        wu_2d = torch.randn(N, K, dtype=torch.bfloat16)
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "expected wt_up to be 5D"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu_2d, wd, None, None, None, "silu"
            )

    def test_invalid_bias_dtype(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.bfloat16)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        bad_bias = torch.randn(N, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "expected up_bias to be bfloat16"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, bad_bias, None, "silu"
            )

    def test_invalid_bias_dim(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.bfloat16)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        bad_bias = torch.randn(1, N, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "expected down_bias to be 1D"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, None, bad_bias, "silu"
            )

    def test_invalid_bias_size(self):
        K, N = 2048, 8192
        src = torch.randn(1, 8, K, dtype=torch.bfloat16)
        wu = _pack(torch.randn(N, K, dtype=torch.bfloat16))
        wd = _pack(torch.randn(K, N, dtype=torch.bfloat16))
        wrong_size_bias = torch.randn(N + 1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "expected up_bias size"):
            torch.ops.pace.libxsmm_fused_mlp(
                src, None, wu, wd, None, wrong_size_bias, None, "silu"
            )


class TestTPPFusedMLPBackend(TestCase):
    """Tests through the MergedMLP + TPPFusedMLP backend pipeline."""

    def _make_mlp(self, K, N, activation, gate, bias, preprocess=True):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            from pace.ops.mlp import MergedMLP
            from pace.ops.enum import BackendType

            mlp = MergedMLP(
                in_features=K,
                out_features=N,
                bias=bias,
                activation=activation,
                gate=gate,
                backend_impl=BackendType.TPP,
            )
            if preprocess:
                mlp.backend.preprocess(mlp)
            mlp.eval()
            return mlp
        finally:
            torch.set_default_dtype(prev_dtype)

    @settings(deadline=None, max_examples=10)
    @given(
        activation=st.sampled_from(["silu", "gelu", "relu"]),
        gate=st.booleans(),
        bias=st.booleans(),
    )
    def test_output_shape_and_dtype(self, activation, gate, bias):
        K, N = 4096, 14336
        mlp = self._make_mlp(K, N, activation, gate=gate, bias=bias)
        x = torch.randn(1, 8, K, dtype=torch.bfloat16) * SCALE
        out = mlp(x)
        self.assertEqual(out.shape, (1, 8, K))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_backend_type(self):
        mlp = self._make_mlp(4096, 14336, "silu", gate=True, bias=False)
        self.assertEqual(type(mlp.backend).__name__, "TPPFusedMLP")

    def test_correctness_through_backend(self):
        torch.manual_seed(42)
        K, N, M = 4096, 14336, 32
        src = torch.randn(1, M, K, dtype=torch.bfloat16) * SCALE
        wg_2d = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wu_2d = torch.randn(N, K, dtype=torch.bfloat16) * SCALE
        wd_2d = torch.randn(K, N, dtype=torch.bfloat16) * SCALE

        ref = _ref_gated_mlp(src, wg_2d, wu_2d, wd_2d, None, None, None, F.silu)

        mlp = self._make_mlp(K, N, "silu", gate=True, bias=False, preprocess=False)
        with torch.no_grad():
            mlp.gate_proj.linear.weight.copy_(wg_2d)
            mlp.up_proj.linear.weight.copy_(wu_2d)
            mlp.down_proj.weight.copy_(wd_2d)
        mlp.backend.preprocess(mlp)
        out = mlp(src)
        self.assertEqual(out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)
