# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch

from pace.ops.enum import OperatorType
from pace.ops.registry import backend_registry
from pace.ops.activations import Activation, SoftMax, SiLU, GeLU, ReLU, Tanh, Sigmoid


class TestActivations(TestCase):

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.RELU)),
    )
    def test_activation_relu(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = Activation("relu", backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.relu(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.SOFTMAX)),
    )
    def test_softmax(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = SoftMax(dim=0, backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(
            output_tensor,
            torch.softmax(input_tensor, dim=0, dtype=backend[1].to_torch_dtype()),
        )

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.SILU)),
    )
    def test_silu(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = SiLU(backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.nn.functional.silu(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.GELU)),
    )
    def test_gelu(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = GeLU(backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.nn.functional.gelu(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.TANH)),
    )
    def test_relu(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = ReLU(backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.nn.functional.relu(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.TANH)),
    )
    def test_tanh(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = Tanh(backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.tanh(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())

    @given(
        st.lists(
            st.integers(min_value=1, max_value=16 * 1024), min_size=1, max_size=10
        ),
        st.sampled_from(backend_registry.get_available_backends(OperatorType.SIGMOID)),
    )
    def test_sigmoid(self, input_data, backend):
        input_tensor = torch.tensor(input_data, dtype=backend[1].to_torch_dtype())
        activation = Sigmoid(backend_impl=backend[0], dtype=backend[1])
        output_tensor = activation(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)
        self.assertEqual(output_tensor, torch.sigmoid(input_tensor))
        self.assertEqual(output_tensor.dtype, backend[1].to_torch_dtype())
