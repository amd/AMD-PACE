# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch
import torch.nn.functional as F

from pace.ops.enum import OperatorType
from pace.ops.registry import backend_registry
from pace.ops.linear import Linear, RepeatedKVLinear, FusedQKVLinear


class TestLinear(TestCase):

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_linear(self, backend):
        linear = Linear(128, 64, backend_impl=backend[0], dtype=backend[1])
        self.assertEqual(linear.weight.shape, (64, 128))
        self.assertEqual(linear.bias.shape, (64,))

        # Initialize weights and bias
        weight = torch.randn(64, 128, dtype=backend[1].to_torch_dtype())
        bias = torch.randn(64, dtype=backend[1].to_torch_dtype())
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

        random_input = torch.randn(5, 128, dtype=backend[1].to_torch_dtype())
        ref_out = F.linear(random_input, weight, bias)

        linear.backend.preprocess(linear)
        output = linear(random_input)
        self.assertEqual(output.shape, (5, 64))
        self.assertEqual(output, ref_out)

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_repeated_kv_linear(self, backend):
        num_key_value_heads = 8
        kv_linear = RepeatedKVLinear(
            64,
            128,
            num_key_value_heads=num_key_value_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        kv_linear.weight.load_weights(
            kv_linear.weight,
            torch.randn(
                128 // num_key_value_heads, 64, dtype=backend[1].to_torch_dtype()
            ),
        )
        self.assertEqual(kv_linear.weight.shape, (128, 64))

        kv_linear.bias.load_weights(
            kv_linear.bias,
            torch.randn(128 // num_key_value_heads, dtype=backend[1].to_torch_dtype()),
        )
        self.assertEqual(kv_linear.bias.shape, (128,))

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_fused_qkv_linear_mha(self, backend):
        """Test FusedQKVLinear.load_from_unfused with MHA (num_heads == num_kv_heads)."""
        hidden_size = 128
        num_heads = 8
        dtype = backend[1].to_torch_dtype()

        fused = FusedQKVLinear(
            in_features=hidden_size,
            out_features=3 * hidden_size,
            bias=True,
            num_key_value_heads=num_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        q_w = torch.randn(hidden_size, hidden_size, dtype=dtype)
        k_w = torch.randn(hidden_size, hidden_size, dtype=dtype)
        v_w = torch.randn(hidden_size, hidden_size, dtype=dtype)
        q_b = torch.randn(hidden_size, dtype=dtype)
        k_b = torch.randn(hidden_size, dtype=dtype)
        v_b = torch.randn(hidden_size, dtype=dtype)

        fused.load_from_unfused(
            {
                "weight": {"q": q_w, "k": k_w, "v": v_w},
                "bias": {"q": q_b, "k": k_b, "v": v_b},
            }
        )

        self.assertEqual(fused.weight.shape, (3 * hidden_size, hidden_size))
        self.assertEqual(fused.bias.shape, (3 * hidden_size,))

        expected_w = torch.cat([q_w, k_w, v_w], dim=0)
        expected_b = torch.cat([q_b, k_b, v_b], dim=0)
        self.assertTrue(torch.equal(fused.weight, expected_w))
        self.assertTrue(torch.equal(fused.bias, expected_b))

        random_input = torch.randn(5, hidden_size, dtype=dtype)
        ref_out = F.linear(random_input, expected_w, expected_b)
        fused.backend.preprocess(fused)
        output = fused(random_input)
        self.assertEqual(output, ref_out)

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_fused_qkv_linear_gqa(self, backend):
        """Test FusedQKVLinear.load_from_unfused with GQA (num_heads > num_kv_heads)."""
        hidden_size = 128
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        out_features = q_size + 2 * kv_size
        dtype = backend[1].to_torch_dtype()

        fused = FusedQKVLinear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True,
            num_key_value_heads=num_kv_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        q_w = torch.randn(q_size, hidden_size, dtype=dtype)
        k_w = torch.randn(kv_size, hidden_size, dtype=dtype)
        v_w = torch.randn(kv_size, hidden_size, dtype=dtype)
        q_b = torch.randn(q_size, dtype=dtype)
        k_b = torch.randn(kv_size, dtype=dtype)
        v_b = torch.randn(kv_size, dtype=dtype)

        fused.load_from_unfused(
            {
                "weight": {"q": q_w, "k": k_w, "v": v_w},
                "bias": {"q": q_b, "k": k_b, "v": v_b},
            }
        )

        self.assertEqual(fused.weight.shape, (out_features, hidden_size))
        self.assertEqual(fused.bias.shape, (out_features,))

        expected_w = torch.cat([q_w, k_w, v_w], dim=0)
        expected_b = torch.cat([q_b, k_b, v_b], dim=0)
        self.assertTrue(torch.equal(fused.weight, expected_w))
        self.assertTrue(torch.equal(fused.bias, expected_b))

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_fused_qkv_linear_no_bias(self, backend):
        """Test FusedQKVLinear.load_from_unfused without bias."""
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        out_features = q_size + 2 * kv_size
        dtype = backend[1].to_torch_dtype()

        fused = FusedQKVLinear(
            in_features=hidden_size,
            out_features=out_features,
            bias=False,
            num_key_value_heads=num_kv_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        q_w = torch.randn(q_size, hidden_size, dtype=dtype)
        k_w = torch.randn(kv_size, hidden_size, dtype=dtype)
        v_w = torch.randn(kv_size, hidden_size, dtype=dtype)

        fused.load_from_unfused(
            {
                "weight": {"q": q_w, "k": k_w, "v": v_w},
            }
        )

        self.assertEqual(fused.weight.shape, (out_features, hidden_size))
        self.assertIsNone(fused.bias)

        expected_w = torch.cat([q_w, k_w, v_w], dim=0)
        self.assertTrue(torch.equal(fused.weight, expected_w))

    @given(
        st.sampled_from(backend_registry.get_available_backends(OperatorType.LINEAR))
    )
    def test_fused_qkv_linear_bias_zero_fill(self, backend):
        """Test that missing bias entries are zero-filled when layer has bias=True."""
        hidden_size = 128
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        out_features = q_size + 2 * kv_size
        dtype = backend[1].to_torch_dtype()

        fused = FusedQKVLinear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True,
            num_key_value_heads=num_kv_heads,
            backend_impl=backend[0],
            dtype=backend[1],
        )

        q_w = torch.randn(q_size, hidden_size, dtype=dtype)
        k_w = torch.randn(kv_size, hidden_size, dtype=dtype)
        v_w = torch.randn(kv_size, hidden_size, dtype=dtype)

        fused.load_from_unfused(
            {
                "weight": {"q": q_w, "k": k_w, "v": v_w},
            }
        )

        expected_b = torch.zeros(out_features, dtype=dtype)
        self.assertTrue(torch.equal(fused.bias, expected_b))
