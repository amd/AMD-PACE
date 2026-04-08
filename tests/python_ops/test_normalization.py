# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401 -- ensures C++ ops and backend registrations are loaded

from pace.ops.enum import OperatorType, FusedOperatorType
from pace.ops.registry import backend_registry
from pace.ops.normalization import (
    LayerNorm,
    RMSNorm,
    FusedRMSNormResidual,
    FusedLayerNormResidual,
)


class TestNormalization(TestCase):

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LAYERNORM))
    )
    def test_layernorm(self, backend):

        normalized_shape = (64,)
        layernorm = LayerNorm(
            normalized_shape,
            elementwise_affine=True,
            backend_impl=backend[0],
            dtype=backend[1],
        )
        self.assertEqual(layernorm.weight.shape, (64,))
        self.assertEqual(layernorm.bias.shape, (64,))

        weight = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        layernorm.weight.copy_(weight)
        layernorm.bias.copy_(bias)

        x = torch.randn(1, 64, dtype=backend[1].to_torch_dtype())
        y = layernorm(x)
        y_torch = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)
        self.assertEqual(y, y_torch)

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.RMSNORM))
    )
    def test_rmsnorm(self, backend):

        normalized_shape = 64
        rmsnorm = RMSNorm(normalized_shape, backend_impl=backend[0], dtype=backend[1])
        self.assertEqual(rmsnorm.weight.shape, (64,))

        weight = torch.randn(normalized_shape, dtype=backend[1].to_torch_dtype())
        rmsnorm.weight.copy_(weight)

        x = torch.randn(1, 64, dtype=backend[1].to_torch_dtype())
        y = rmsnorm(x)
        y_torch = F.rms_norm(x, normalized_shape=(normalized_shape,), weight=weight)
        self.assertEqual(y, y_torch)

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(
                FusedOperatorType.FUSED_RMSNORM_RESIDUAL
            )
        )
    )
    def test_fused_rmsnorm_residual(self, backend):
        """Fused Add+RMSNorm across all registered backends and shapes."""
        for rows, dim in [(1, 64), (2, 64), (4, 128), (8, 64), (32, 256)]:
            with self.subTest(rows=rows, dim=dim, backend=backend):
                fused = FusedRMSNormResidual(
                    dim, backend_impl=backend[0], dtype=backend[1]
                )
                weight = torch.randn(dim, dtype=backend[1].to_torch_dtype())
                fused.weight.copy_(weight)

                x = torch.randn(rows, dim, dtype=backend[1].to_torch_dtype())
                residual = torch.randn(rows, dim, dtype=backend[1].to_torch_dtype())

                normed, residual_out = fused(x, residual)

                residual_ref = x + residual
                normed_ref = F.rms_norm(
                    residual_ref, normalized_shape=(dim,), weight=weight
                )

                self.assertEqual(residual_out, residual_ref, atol=2e-2, rtol=2e-2)
                self.assertEqual(normed, normed_ref, atol=2e-2, rtol=2e-2)

    @given(
        st.sampled_from(
            backend_registry.get_available_backends(
                FusedOperatorType.FUSED_LAYERNORM_RESIDUAL
            )
        )
    )
    def test_fused_layernorm_residual(self, backend):
        """Fused Add+LayerNorm across all registered backends and shapes."""
        for rows, dim in [(1, 64), (2, 64), (4, 128), (8, 64), (32, 256)]:
            with self.subTest(rows=rows, dim=dim, backend=backend):
                fused = FusedLayerNormResidual(
                    dim, backend_impl=backend[0], dtype=backend[1]
                )
                weight = torch.randn((dim,), dtype=backend[1].to_torch_dtype())
                bias = torch.randn((dim,), dtype=backend[1].to_torch_dtype())
                fused.weight.copy_(weight)
                fused.bias.copy_(bias)

                x = torch.randn(rows, dim, dtype=backend[1].to_torch_dtype())
                residual = torch.randn(rows, dim, dtype=backend[1].to_torch_dtype())

                normed, residual_out = fused(x, residual)

                residual_ref = x + residual
                normed_ref = F.layer_norm(
                    residual_ref, normalized_shape=(dim,), weight=weight, bias=bias
                )

                self.assertEqual(residual_out, residual_ref, atol=2e-2, rtol=2e-2)
                self.assertEqual(normed, normed_ref, atol=2e-2, rtol=2e-2)
