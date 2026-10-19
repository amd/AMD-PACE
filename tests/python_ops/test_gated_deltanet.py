# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Shape / size / state sanity tests for the GatedDeltaAttention ops.

These exercise the frontend container and its NATIVE backends, checking that
the recurrence and causal-conv ops accept the expected input shapes, produce
correctly-shaped outputs and refreshed cache states, dispatch prefill vs decode
internally, and handle grouped-query attention (num_v_heads > num_k_heads).
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch

from pace.ops.enum import OperatorType, BackendType
from pace.ops.registry import backend_registry
from pace.ops.gated_deltanet import (
    GatedDeltaAttention,
    GatedDeltaRecurrence,
    CausalConv1dUpdate,
)


def _recurrence_backends():
    return backend_registry.get_available_backends(
        OperatorType.GATED_DELTANET_RECURRENCE
    )


def _conv_backends():
    return backend_registry.get_available_backends(OperatorType.GATED_DELTANET_CONV1D)


class TestGatedDeltaAttention(TestCase):

    # ---- container wiring -------------------------------------------------

    def test_container_exposes_ops(self):
        gda = GatedDeltaAttention(backend_impl=BackendType.NATIVE)
        self.assertIsInstance(gda.recurrence_op, GatedDeltaRecurrence)
        self.assertIsInstance(gda.conv1d, CausalConv1dUpdate)
        self.assertTrue(callable(gda.recurrence))
        self.assertTrue(callable(gda.conv1d_update))

    def test_ops_registered_for_native(self):
        self.assertIn(
            BackendType.NATIVE,
            {b for b, _ in _recurrence_backends()},
        )
        self.assertIn(
            BackendType.NATIVE,
            {b for b, _ in _conv_backends()},
        )

    # ---- recurrence: shapes / state / dispatch ---------------------------

    @settings(deadline=None)
    @given(
        num_k_heads=st.sampled_from([1, 2, 4]),
        num_groups=st.sampled_from([1, 2]),
        head_k_dim=st.sampled_from([32, 64]),
        head_v_dim=st.sampled_from([32, 64]),
        backend=st.sampled_from(_recurrence_backends()),
    )
    def test_recurrence_prefill_shapes(
        self, num_k_heads, num_groups, head_k_dim, head_v_dim, backend
    ):
        dtype = backend[1].to_torch_dtype()
        gda = GatedDeltaAttention(backend_impl=backend[0], dtype=backend[1])

        batch, seq_len = 2, 5
        num_v_heads = num_k_heads * num_groups
        query = torch.randn(batch, seq_len, num_k_heads, head_k_dim, dtype=dtype)
        key = torch.randn(batch, seq_len, num_k_heads, head_k_dim, dtype=dtype)
        value = torch.randn(batch, seq_len, num_v_heads, head_v_dim, dtype=dtype)
        g = torch.randn(batch, seq_len, num_v_heads, dtype=dtype)
        beta = torch.rand(batch, seq_len, num_v_heads, dtype=dtype)

        # Prefill: no prior state -> chunked scan path.
        out, state = gda.recurrence(query, key, value, g, beta, None, num_groups)

        self.assertEqual(
            out.shape, torch.Size([batch, seq_len, num_v_heads, head_v_dim])
        )
        self.assertEqual(
            state.shape,
            torch.Size([batch, num_v_heads, head_k_dim, head_v_dim]),
        )
        self.assertEqual(out.dtype, dtype)
        self.assertFalse(torch.isnan(out).any())

    @settings(deadline=None)
    @given(
        num_k_heads=st.sampled_from([1, 2, 4]),
        num_groups=st.sampled_from([1, 2]),
        head_dim=st.sampled_from([32, 64]),
        backend=st.sampled_from(_recurrence_backends()),
    )
    def test_recurrence_decode_shapes_and_state_update(
        self, num_k_heads, num_groups, head_dim, backend
    ):
        dtype = backend[1].to_torch_dtype()
        gda = GatedDeltaAttention(backend_impl=backend[0], dtype=backend[1])

        batch = 2
        num_v_heads = num_k_heads * num_groups
        state = torch.zeros(batch, num_v_heads, head_dim, head_dim, dtype=torch.float32)

        def step():
            query = torch.randn(batch, 1, num_k_heads, head_dim, dtype=dtype)
            key = torch.randn(batch, 1, num_k_heads, head_dim, dtype=dtype)
            value = torch.randn(batch, 1, num_v_heads, head_dim, dtype=dtype)
            g = torch.randn(batch, 1, num_v_heads, dtype=dtype)
            beta = torch.rand(batch, 1, num_v_heads, dtype=dtype)
            return query, key, value, g, beta

        # Decode: state is not None and seq_len == 1 -> single-step recurrence.
        out, new_state = gda.recurrence(*step(), state, num_groups)

        self.assertEqual(out.shape, torch.Size([batch, 1, num_v_heads, head_dim]))
        self.assertEqual(new_state.shape, state.shape)
        self.assertEqual(out.dtype, dtype)
        # The recurrent state must evolve away from the zero initial state.
        self.assertFalse(
            torch.allclose(new_state.float(), torch.zeros_like(new_state.float()))
        )

    # ---- causal conv1d: shapes / state / dispatch ------------------------

    @settings(deadline=None)
    @given(
        channels=st.sampled_from([8, 16, 32]),
        kernel_size=st.sampled_from([2, 3, 4]),
        seq_len=st.sampled_from([1, 3, 7]),
        backend=st.sampled_from(_conv_backends()),
    )
    def test_conv1d_output_and_state_shapes(
        self, channels, kernel_size, seq_len, backend
    ):
        dtype = backend[1].to_torch_dtype()
        gda = GatedDeltaAttention(backend_impl=backend[0], dtype=backend[1])

        batch = 2
        weight = torch.randn(channels, kernel_size, dtype=dtype)

        # Prefill (no cache): output width == seq_len, state == kernel_size - 1.
        x = torch.randn(batch, channels, seq_len, dtype=dtype)
        out, conv_state = gda.conv1d_update(x, None, weight)

        self.assertEqual(out.shape, torch.Size([batch, channels, seq_len]))
        self.assertEqual(
            conv_state.shape, torch.Size([batch, channels, kernel_size - 1])
        )
        self.assertEqual(out.dtype, dtype)

    @settings(deadline=None)
    @given(
        channels=st.sampled_from([8, 16, 32]),
        kernel_size=st.sampled_from([2, 3, 4]),
        backend=st.sampled_from(_conv_backends()),
    )
    def test_conv1d_decode_updates_state_in_place(self, channels, kernel_size, backend):
        dtype = backend[1].to_torch_dtype()
        gda = GatedDeltaAttention(backend_impl=backend[0], dtype=backend[1])

        batch = 2
        weight = torch.randn(channels, kernel_size, dtype=dtype)
        conv_state = torch.randn(batch, channels, kernel_size - 1, dtype=dtype)
        x = torch.randn(batch, channels, 1, dtype=dtype)

        # Decode (cache present, seq_len == 1): single-token output, in-place
        # state update returning the same tensor object.
        out, returned_state = gda.conv1d_update(x, conv_state, weight)

        self.assertEqual(out.shape, torch.Size([batch, channels, 1]))
        self.assertEqual(
            returned_state.shape, torch.Size([batch, channels, kernel_size - 1])
        )
        self.assertIs(returned_state, conv_state)

    @settings(deadline=None)
    @given(
        channels=st.sampled_from([8, 16]),
        kernel_size=st.sampled_from([3, 4]),
        seq_len=st.sampled_from([2, 4]),
        backend=st.sampled_from(_conv_backends()),
    )
    def test_conv1d_continuation_prefill_width(
        self, channels, kernel_size, seq_len, backend
    ):
        dtype = backend[1].to_torch_dtype()
        gda = GatedDeltaAttention(backend_impl=backend[0], dtype=backend[1])

        batch = 2
        weight = torch.randn(channels, kernel_size, dtype=dtype)
        conv_state = torch.randn(batch, channels, kernel_size - 1, dtype=dtype)
        x = torch.randn(batch, channels, seq_len, dtype=dtype)

        # Continuation prefill (cache present, seq_len > 1): output must keep the
        # original seq_len width despite the prepended cache.
        out, new_state = gda.conv1d_update(x, conv_state, weight)

        self.assertEqual(out.shape, torch.Size([batch, channels, seq_len]))
        self.assertEqual(
            new_state.shape, torch.Size([batch, channels, kernel_size - 1])
        )
