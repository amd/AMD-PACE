# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
# In /amd_pace/tests run using python -m unittest -v ops/test_AOCLDLPLinear.py

import os
import warnings

import torch
import pace  # noqa: F401
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
from hypothesis import given, settings
import hypothesis.strategies as st


def reshape_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Reshape weights to align with PACE operations. Only 2D weights are supported.

    Environment Variable:
        PACE_USE_AOCL_DLP_RESHAPE: Default (unset): enabled. "1": enabled.
                                   "0": disabled. Any other value: warn and enable.
    """
    reshaped_weights = torch.transpose(weights, 0, 1).contiguous()

    # Default 1 (enabled). "1" enabled, "0" disabled. Other: warn and enable.
    _raw = os.getenv("PACE_USE_AOCL_DLP_RESHAPE", "1").strip()
    if _raw == "0":
        use_dlp_reshape = False
    elif _raw == "1":
        use_dlp_reshape = True
    else:
        if _raw:
            warnings.warn(
                f'PACE_USE_AOCL_DLP_RESHAPE has invalid value "{_raw}"; '
                'expected "0" or "1". Defaulting to enabled (1).',
                stacklevel=2,
            )
        use_dlp_reshape = True

    # Only call AOCL-DLP reshape if enabled and dtype is bfloat16
    # For other dtypes, let the linear operation validate and raise the error
    if use_dlp_reshape and weights.dtype == torch.bfloat16:
        reshaped_weights = torch.ops.pace.aocl_dlp_reshape_weights(reshaped_weights)

    return reshaped_weights


# For higher BS(input = (BS = 64, 64, 4096) and above ) 1e-3 is not providing accurate results
default_tolerance = 1e-2

# Common settings for hypothesis-based tests, adjusting input shapes and conditions.
# Note: Only 2D weights are supported - 5D weight functionality has been disabled
common_hypothesis = settings(deadline=None, max_examples=10)(
    given(
        input_shape=st.sampled_from([(128, 64, 128)]),
        weight_shape=st.sampled_from([(128, 128)]),
        dtype=st.sampled_from([torch.bfloat16]),
        use_bias=st.booleans(),
    )
)


class TestLinear(TestCase):
    """Test cases for various AOCL-DLP linear operators with hypothesis framework."""

    @settings(deadline=None)
    @common_hypothesis
    def test_aocl_dlp_linear_plain(self, input_shape, weight_shape, dtype, use_bias):
        """Test plain linear operation against standard PyTorch."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.linear(inputs, weights, bias)
        reshaped_weights = reshape_weights(weights)
        pace_out = torch.ops.pace.aocl_dlp_linear_plain(inputs, reshaped_weights, bias)
        self.assertEqual(
            std_out, pace_out, atol=default_tolerance, rtol=default_tolerance
        )

    @settings(deadline=None)
    @common_hypothesis
    def test_aocl_dlp_linear_silu(self, input_shape, weight_shape, dtype, use_bias):
        """Test SiLU activation after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.silu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights)
        pace_out = torch.ops.pace.aocl_dlp_linear_silu(inputs, reshaped_weights, bias)
        self.assertEqual(
            std_out, pace_out, atol=default_tolerance, rtol=default_tolerance
        )

    @settings(deadline=None)
    @common_hypothesis
    def test_aocl_dlp_linear_gelu(self, input_shape, weight_shape, dtype, use_bias):
        """Test GELU activation function after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.gelu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights)
        pace_out = torch.ops.pace.aocl_dlp_linear_gelu(inputs, reshaped_weights, bias)
        self.assertEqual(
            std_out, pace_out, atol=default_tolerance, rtol=default_tolerance
        )

    @settings(deadline=None)
    @common_hypothesis
    def test_aocl_dlp_linear_relu(self, input_shape, weight_shape, dtype, use_bias):
        """Test ReLU activation function after linear operation."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None

        std_out = F.relu(F.linear(inputs, weights, bias))
        reshaped_weights = reshape_weights(weights)
        pace_out = torch.ops.pace.aocl_dlp_linear_relu(inputs, reshaped_weights, bias)
        self.assertEqual(
            std_out, pace_out, atol=default_tolerance, rtol=default_tolerance
        )

    @settings(deadline=None)
    @common_hypothesis
    def test_aocl_dlp_linear_mul(self, input_shape, weight_shape, dtype, use_bias):
        """Test multiplication after linear operation with a secondary input tensor."""
        inputs = torch.rand(input_shape, dtype=dtype)
        weights = torch.rand(weight_shape, dtype=dtype)
        bias = torch.rand(weight_shape[0], dtype=dtype) if use_bias else None
        mul_input_shape = (input_shape[0], input_shape[1], weight_shape[0])
        mul_input = torch.rand(mul_input_shape, dtype=dtype)

        std_out = mul_input * F.linear(inputs, weights, bias)
        reshaped_weights = reshape_weights(weights)
        pace_out = torch.ops.pace.aocl_dlp_linear_mul(
            inputs, mul_input, reshaped_weights, bias
        )
        # Observed Mismatched elements: (1 / 1048576) for 1e-2 for higher BS, So setting threshold to 1e-1
        self.assertEqual(std_out, pace_out, atol=1e-1, rtol=1e-1)

    # Invalid Test Cases
    @given(
        op=st.sampled_from(
            [
                (torch.ops.pace.aocl_dlp_linear_plain, "dlp_linear_plain", False),
                (torch.ops.pace.aocl_dlp_linear_silu, "dlp_linear_silu", False),
                (torch.ops.pace.aocl_dlp_linear_gelu, "dlp_linear_gelu", False),
                (torch.ops.pace.aocl_dlp_linear_relu, "dlp_linear_relu", False),
                (torch.ops.pace.aocl_dlp_linear_mul, "dlp_linear_mul", True),
            ]
        ),
        dtype=st.sampled_from([torch.float32, torch.float64]),
        size=st.sampled_from([(128, 4096)]),
    )
    def test_aocl_dlp_linear_invalid_dtype(self, op, dtype, size):
        """Test handling of invalid dtype inputs for AOCL-DLP linear operators."""
        op_fn, op_name, is_mul = op
        dtype_str = {torch.float32: "Float", torch.float64: "Double"}[dtype]

        # Test when input has an invalid dtype compared to weights
        inputs = torch.randn(*size, dtype=dtype)
        weights = torch.randn(*size, dtype=torch.bfloat16)
        bias = torch.randn(size[0], dtype=torch.bfloat16)
        if is_mul:
            mul_input = torch.randn(1, size[1], dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: {dtype_str}, weight: BFloat16",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    reshape_weights(weights),
                    bias,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: {dtype_str}, weight: BFloat16",
            ):
                op_fn(inputs.unsqueeze(1), reshape_weights(weights), bias)

        # Test when weight has an invalid dtype compared to inputs
        inputs = torch.randn(*size, dtype=torch.bfloat16)
        weights = torch.randn(*size, dtype=dtype)
        if is_mul:
            mul_input = torch.randn(1, size[1], dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: BFloat16, weight: {dtype_str}.",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    reshape_weights(weights),
                    bias,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} got mismatched types, input: BFloat16, weight: {dtype_str}.",
            ):
                op_fn(inputs.unsqueeze(1), reshape_weights(weights), bias)

    @given(
        op=st.sampled_from(
            [
                (torch.ops.pace.aocl_dlp_linear_plain, "dlp_linear_plain", False),
                (torch.ops.pace.aocl_dlp_linear_silu, "dlp_linear_silu", False),
                (torch.ops.pace.aocl_dlp_linear_gelu, "dlp_linear_gelu", False),
                (torch.ops.pace.aocl_dlp_linear_relu, "dlp_linear_relu", False),
                (torch.ops.pace.aocl_dlp_linear_mul, "dlp_linear_mul", True),
            ]
        ),
        invalid_size=st.sampled_from([(128, 4096)]),
    )
    def test_aocl_dlp_linear_invalid_input_dim(self, op, invalid_size):
        """Test handling of invalid input dimensions for operators, expecting 3D inputs."""
        op_fn, op_name, is_mul = op

        # Generate invalid input dimensions
        inputs = torch.randn(*invalid_size, dtype=torch.bfloat16)

        if is_mul:
            # Generate a secondary input tensor for multiplication
            mul_input = torch.randn(*invalid_size, dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError, f"pace::{op_name} expected input to be 3D"
            ):
                op_fn(inputs, mul_input, reshape_weights(inputs), None)
        else:
            with self.assertRaisesRegex(
                RuntimeError, f"pace::{op_name} expected input to be 3D"
            ):
                op_fn(inputs, reshape_weights(inputs), None)

        # Test invalid weight dimensions - only 2D weights are accepted
        weights = torch.randn(*invalid_size, dtype=torch.bfloat16)
        if is_mul:
            mul_input = torch.randn(1, invalid_size[1], dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                RuntimeError,
                f"pace::{op_name} expected weight to be 2D, but got",
            ):
                op_fn(
                    inputs.unsqueeze(1),
                    mul_input.unsqueeze(1),
                    weights.unsqueeze(1),
                    None,
                )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                rf"pace::{op_name} expected weight to be (?:one of )?2D, but got",
            ):
                op_fn(inputs.unsqueeze(1), weights.unsqueeze(1), None)


if __name__ == "__main__":
    import unittest

    unittest.main()
