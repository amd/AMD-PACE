# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m unittest -v test_gqa.py

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401


def prepare_gqa_input(
    B: int,
    S: int,
    H: int,
    L: int,
    num_q_heads: int = 1024,
    num_kv_heads: int = 128,
    dtype: torch.dtype = torch.float32,
):
    """
    Prepare inputs for GQA tests

    Args:
        B, S, H, L, num_q_heads, num_kv_heads
        dtype: Data type of input tensor

    Returns:
        Query, Key, Value, Attention Mask tensors
    """

    shape_q = (B, num_q_heads, S, H)
    shape_kv = (B, num_kv_heads, L, H)
    shape_mask = (B, 1, S, L)

    input_Q = torch.randn(*shape_q).to(dtype)
    input_K = torch.randn(*shape_kv).to(dtype)
    input_V = torch.randn(*shape_kv).to(dtype)
    input_mask = torch.randn(*shape_mask).to(dtype)

    return input_Q, input_K, input_V, input_mask


def Torch_direct_GQA(
    input_Q, input_K, input_V, input_mask=None, dtype: torch.dtype = torch.float32
):
    torch_gqa_output = torch.nn.functional.scaled_dot_product_attention(
        input_Q, input_K, input_V, input_mask, enable_gqa=True
    )

    return torch_gqa_output


class TestGQA(TestCase):
    """
    Test cases for pace GQA attention op
    """

    @settings(
        deadline=None,
        max_examples=20,
    )
    @given(
        batch=st.sampled_from([1, 64, 128]),
        head_dim=st.sampled_from([1, 96, 128]),
        seq_len=st.sampled_from([1, 128, 512]),
        KV_len=st.sampled_from([1, 128, 512]),
        num_q_heads=st.sampled_from([1024]),
        num_kv_heads=st.sampled_from([128]),
        input_dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    def test_gqa(
        self, batch, seq_len, head_dim, KV_len, num_q_heads, num_kv_heads, input_dtype
    ):

        input_Q, input_K, input_V, input_mask = prepare_gqa_input(
            batch, seq_len, head_dim, KV_len, num_q_heads, num_kv_heads, input_dtype
        )
        pace_gqa_output = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, input_mask
        )

        # Reference torch direct API
        Torch_gqa_Direct_output = Torch_direct_GQA(
            input_Q, input_K, input_V, input_mask, input_dtype
        )

        threshold = 1e-5
        if input_dtype == torch.bfloat16:
            # GQA output threshold is higher for bf16 due to error accumulation over multiple ops
            threshold = 1e-1

        # Comparing PACE outputs wrt Torch direct API
        self.assertEqual(
            Torch_gqa_Direct_output,
            pace_gqa_output,
            atol=threshold,
            rtol=threshold,
        )

    def test_gqa_invalid_dtypes(
        self,
    ):

        B = 64
        S = 512
        H = 128
        L = 512
        num_q_heads = 512
        num_kv_heads = 128

        with self.assertRaisesRegex(
            RuntimeError,
            "pace::GQA attention supports only the dtypes Float and BF16 types for input",
        ):
            input_Q, input_K, input_V, input_mask = prepare_gqa_input(
                B, S, H, L, num_q_heads, num_kv_heads, torch.int8
            )

            torch.ops.pace.grouped_query_attention(
                input_Q, input_K, input_V, input_mask
            )

    def test_gqa_input_shape(self):

        B = 64
        S = 512
        H = 128
        L = 512
        num_q_heads = 512
        num_kv_heads = 128

        proj_shape_Q = (B * num_q_heads, -1, H)
        proj_shape_KV = (B * num_kv_heads, -1, L)
        proj_shape_mask = (B, S, L)

        dtype = torch.float32

        input_Q, input_K, input_V, input_mask = prepare_gqa_input(
            B, S, H, L, num_q_heads, num_kv_heads, dtype
        )

        with self.assertRaisesRegex(RuntimeError, "GQA requires 4D inputs"):
            torch.ops.pace.grouped_query_attention(
                input_Q.view(*proj_shape_Q), input_K, input_V, input_mask
            )

        with self.assertRaisesRegex(RuntimeError, "GQA requires 4D inputs"):
            torch.ops.pace.grouped_query_attention(
                input_Q, input_K.view(*proj_shape_KV), input_V, input_mask
            )

        with self.assertRaisesRegex(RuntimeError, "GQA requires 4D inputs"):
            torch.ops.pace.grouped_query_attention(
                input_Q, input_K, input_V.view(*proj_shape_KV), input_mask
            )

        with self.assertRaisesRegex(RuntimeError, "GQA requires 4D attention mask"):
            torch.ops.pace.grouped_query_attention(
                input_Q, input_K, input_V, input_mask.view(*proj_shape_mask)
            )

    def test_gqa_invalid_head_dims(self):

        B = 64
        S = 512
        H = 128
        L = 512
        num_q_heads = 500
        num_kv_heads = 128

        dtype = torch.float32

        input_Q, input_K, input_V, input_mask = prepare_gqa_input(
            B, S, H, L, num_q_heads, num_kv_heads, dtype
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "pace::GQA requires that the number of Q heads be divisible by the number of KV heads",
        ):
            torch.ops.pace.grouped_query_attention(
                input_Q, input_K, input_V, input_mask
            )
