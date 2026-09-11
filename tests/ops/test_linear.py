# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
from typing import Tuple

import pace  # noqa: F401
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase


def prepare_linear_input(
    ndim: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for linear tests

    Args:
        ndim: Number of dimensions for input tensor
        dtype: Data type of input tensor

    Returns:
        inputs: Input tensor
        weights: Weight tensor
        bias: Bias tensor
    """

    min_power = 1
    max_power = 10

    input_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(ndim,))
    ).tolist()
    weight_size = (
        2 ** torch.randint(low=min_power, high=max_power, size=(1,))
    ).tolist()
    weight_size.append(input_size[-1])

    inputs = torch.rand(input_size, dtype=dtype)
    weights = torch.rand(weight_size, dtype=dtype)
    bias = torch.rand(weight_size[0], dtype=dtype)

    return inputs, weights, bias


class TestLinear(TestCase):
    """
    Test cases for linear linear operators
    """

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_nd(self, ndim, dtype):
        inputs, weights, bias = prepare_linear_input(ndim, dtype)
        std_out = torch.nn.functional.linear(inputs, weights, bias)
        pace_out = torch.ops.pace.linear(inputs, weights, bias)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_nd_without_bias(self, ndim, dtype):
        inputs, weights, _ = prepare_linear_input(ndim, dtype)
        std_out = torch.nn.functional.linear(inputs, weights)
        pace_out = torch.ops.pace.linear(inputs, weights, None)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_with_relu(self, ndim, dtype):
        inputs, weights, bias = prepare_linear_input(ndim, dtype)
        std_out = torch.relu(torch.nn.functional.linear(inputs, weights, bias))
        pace_out = torch.ops.pace.linear_relu(inputs, weights, bias)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    @given(
        ndim=st.integers(1, 3), dtype=st.sampled_from([torch.float32, torch.bfloat16])
    )
    def test_linear_with_relu_without_bias(self, ndim, dtype):
        inputs, weights, _ = prepare_linear_input(ndim, dtype)
        std_out = torch.relu(torch.nn.functional.linear(inputs, weights))
        pace_out = torch.ops.pace.linear_relu(inputs, weights, None)
        self.assertEqual(std_out, pace_out, atol=1e-3, rtol=1e-3)

    def test_linear_invalid_input(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs, weights.to(torch.float32), bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs.to(torch.float32), weights, bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear(inputs, weights, bias.to(torch.float32))

    def test_linear_invalid_input_shape(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got incompatible weight and bias"):
            torch.ops.pace.linear(inputs, weights, bias[1:])

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear(inputs[1:], weights, bias)

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear(inputs[1:], weights[1:], bias[1:])

    def test_linear_invalid_input_dim(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got dims less than 1"):
            torch.ops.pace.linear(inputs[0], weights, bias)

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for weight"
        ):
            torch.ops.pace.linear(inputs, weights[0], bias)

        with self.assertRaisesRegex(RuntimeError, "only supports 1D tensors for bias"):
            torch.ops.pace.linear(inputs, weights, bias.unsqueeze(0))

    def test_linear_invalid_input_dtype(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes float and bfloat16"
        ):
            torch.ops.pace.linear(
                inputs.to(torch.int32), weights.to(torch.int32), bias.to(torch.int32)
            )

    def test_linear_invalid_input_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs, weights.to(torch.float32), bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs.to(torch.float32), weights, bias)

        with self.assertRaisesRegex(RuntimeError, "mismatched types"):
            torch.ops.pace.linear_relu(inputs, weights, bias.to(torch.float32))

    def test_linear_invalid_input_shape_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got incompatible weight and bias"):
            torch.ops.pace.linear_relu(inputs, weights, bias[1:])

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear_relu(inputs[1:], weights, bias)

        with self.assertRaisesRegex(RuntimeError, "got incompatible input and weight"):
            torch.ops.pace.linear_relu(inputs[1:], weights[1:], bias[1:])

    def test_linear_invalid_input_dim_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "got dims less than 1"):
            torch.ops.pace.linear_relu(inputs[0], weights, bias)

        with self.assertRaisesRegex(
            RuntimeError, "only supports 2D tensors for weight"
        ):
            torch.ops.pace.linear_relu(inputs, weights[0], bias)

        with self.assertRaisesRegex(RuntimeError, "only supports 1D tensors for bias"):
            torch.ops.pace.linear_relu(inputs, weights, bias.unsqueeze(0))

    def test_linear_invalid_input_dtype_relu(self):
        inputs, weights, bias = prepare_linear_input(1, torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "only support the dtypes float and bfloat16"
        ):
            torch.ops.pace.linear_relu(
                inputs.to(torch.int32), weights.to(torch.int32), bias.to(torch.int32)
            )
