# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import pace  # noqa: F401

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

from pace.ops.enum import BackendType, DataType, OperatorType
from pace.ops.registry import backend_registry
from pace.ops.normalization import LayerNorm, RMSNorm


class TestNormOps(TestCase):
    """Direct C++ norm op tests (torch.ops.pace.*)."""

    def test_fused_add_rmsnorm(self):
        dim = 128
        x = torch.randn(16, dim, dtype=torch.bfloat16)
        residual = torch.randn(16, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.bfloat16)

        normed, residual_out = torch.ops.pace.fused_add_rmsnorm(
            x, residual, weight, 1e-6
        )

        residual_ref = x + residual
        normed_ref = F.rms_norm(residual_ref, normalized_shape=(dim,), weight=weight)

        self.assertEqual(residual_out, residual_ref, atol=2e-2, rtol=2e-2)
        self.assertEqual(normed, normed_ref, atol=2e-2, rtol=2e-2)

    def test_rmsnorm(self):
        dim = 128
        x = torch.randn(16, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.bfloat16)

        y = torch.ops.pace.rmsnorm(x, weight, 1e-6)
        y_ref = F.rms_norm(x, normalized_shape=(dim,), weight=weight)

        self.assertEqual(y, y_ref, atol=2e-2, rtol=2e-2)

    def test_fused_add_layernorm(self):
        dim = 128
        x = torch.randn(16, dim, dtype=torch.bfloat16)
        residual = torch.randn(16, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.bfloat16)
        bias = torch.randn(dim, dtype=torch.bfloat16)

        normed, residual_out = torch.ops.pace.fused_add_layernorm(
            x, residual, weight, bias, 1e-5
        )

        residual_ref = x + residual
        normed_ref = F.layer_norm(
            residual_ref, normalized_shape=(dim,), weight=weight, bias=bias
        )

        self.assertEqual(residual_out, residual_ref, atol=2e-2, rtol=2e-2)
        self.assertEqual(normed, normed_ref, atol=2e-2, rtol=2e-2)

    def test_layernorm(self):
        dim = 128
        x = torch.randn(16, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.bfloat16)
        bias = torch.randn(dim, dtype=torch.bfloat16)

        y = torch.ops.pace.layernorm(x, weight, bias, 1e-5)
        y_ref = F.layer_norm(x, normalized_shape=(dim,), weight=weight, bias=bias)

        self.assertEqual(y, y_ref, atol=2e-2, rtol=2e-2)


class TestJITNormBackends(TestCase):
    """JIT norm backends vs NATIVE reference."""

    def setUp(self):
        jit_rmsnorm = backend_registry.get(
            OperatorType.RMSNORM, BackendType.JIT, DataType.BFLOAT16
        )
        jit_layernorm = backend_registry.get(
            OperatorType.LAYERNORM, BackendType.JIT, DataType.BFLOAT16
        )
        if jit_rmsnorm is None or jit_layernorm is None:
            self.skipTest("JIT norm backends not registered")

    def test_jit_rmsnorm(self):
        for rows, dim in [(1, 64), (8, 128), (32, 256)]:
            with self.subTest(rows=rows, dim=dim):
                native = RMSNorm(
                    dim, backend_impl=BackendType.NATIVE, dtype=DataType.BFLOAT16
                )
                jit = RMSNorm(
                    dim, backend_impl=BackendType.JIT, dtype=DataType.BFLOAT16
                )
                jit.weight.copy_(native.weight)

                x = torch.randn(rows, dim, dtype=torch.bfloat16)
                y_native = native(x)
                y_jit = jit(x)
                self.assertEqual(y_jit, y_native, atol=2e-2, rtol=2e-2)

    def test_jit_layernorm(self):
        for rows, dim in [(1, 64), (8, 128), (32, 256)]:
            with self.subTest(rows=rows, dim=dim):
                native = LayerNorm(
                    dim, backend_impl=BackendType.NATIVE, dtype=DataType.BFLOAT16
                )
                jit = LayerNorm(
                    dim, backend_impl=BackendType.JIT, dtype=DataType.BFLOAT16
                )
                jit.weight.copy_(native.weight)
                jit.bias.copy_(native.bias)

                x = torch.randn(rows, dim, dtype=torch.bfloat16)
                y_native = native(x)
                y_jit = jit(x)
                self.assertEqual(y_jit, y_native, atol=2e-2, rtol=2e-2)
