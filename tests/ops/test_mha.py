# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m unittest -v test_mha.py

import math
import random

import torch
from torch import nn
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401


def prepare_mha_input(B: int, N: int, S: int, H: int, L: int, dtype: torch.dtype):
    """
    Prepare inputs for MHA tests

    Args:
        B, N, S, H, L
        dtype: Data type of input tensor

    Returns:
        Query, Key, Value, Attention Mask tensors
    """

    shape_q = (B, N, S, H)
    shape_kv = (B, N, L, H)
    shape_mask = (B, 1, 1, L)

    input_Q = torch.randn(*shape_q).to(dtype)
    input_K = torch.randn(*shape_kv).to(dtype)
    input_V = torch.randn(*shape_kv).to(dtype)
    input_mask = torch.randn(*shape_mask).to(dtype)

    return input_Q, input_K, input_V, input_mask


def Torch_ops(
    input_Q, input_K, input_V, input_mask=None, dtype: torch.dtype = torch.float32
):

    attn_weights = torch.matmul(input_Q, input_K.transpose(2, 3)) / math.sqrt(
        input_Q.shape[3]
    )
    # mask add
    if input_mask is not None:
        attn_weights = attn_weights + input_mask
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        dtype
    )
    attn_output = torch.matmul(attn_weights, input_V)

    return attn_output


def Torch_direct_MHA(
    input_Q, input_K, input_V, input_mask=None, dtype: torch.dtype = torch.float32
):

    torch_sdpa_output = torch.nn.functional.scaled_dot_product_attention(
        input_Q, input_K, input_V, input_mask
    )

    return torch_sdpa_output


class TestMHA(TestCase):
    """
    Test cases for pace attention op
    """

    @settings(
        deadline=None,
        max_examples=20,
    )
    @given(
        batch=st.sampled_from([1, 32, 96]),
        num_heads=st.sampled_from([1, 16, 32]),
        head_dim=st.sampled_from([1, 64, 128]),
        seq_len=st.sampled_from([1, 128, 512]),
        KV_len=st.sampled_from([1, 512, 2048]),
        use_KQ=st.booleans(),
        input_dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    def test_mha(
        self, batch, num_heads, seq_len, head_dim, KV_len, use_KQ, input_dtype
    ):

        input_Q, input_K, input_V, input_mask = prepare_mha_input(
            batch, num_heads, seq_len, head_dim, KV_len, input_dtype
        )
        pace_mha_output = torch.ops.pace.multi_head_attention(
            input_Q, input_K, input_V, input_mask, use_KQ
        )

        # Reference torch ops
        Torch_mha_Ops_output = Torch_ops(
            input_Q, input_K, input_V, input_mask, input_dtype
        )

        # Reference torch direct API
        Torch_mha_Direct_output = Torch_direct_MHA(
            input_Q, input_K, input_V, input_mask, input_dtype
        )

        threshold = 1e-5
        if input_dtype == torch.bfloat16:
            # SDPA output threshold is higher for bf16 due to error accumulation over multiple ops
            threshold = 1e-1

        # Comparing SDPA outputs wrt Torch ops and Torch direct SDPA API
        self.assertEqual(
            Torch_mha_Ops_output, pace_mha_output, atol=threshold, rtol=threshold
        )
        self.assertEqual(
            Torch_mha_Direct_output,
            pace_mha_output,
            atol=threshold,
            rtol=threshold,
        )

    def test_mha_invalid_dtypes(
        self,
    ):

        B = 64
        N = 32
        S = 512
        H = 128
        L = 512

        use_KQ = 0

        with self.assertRaisesRegex(
            RuntimeError,
            "MHA attention only support the dtypes Float and BF16 types for input",
        ):

            input_Q, input_K, input_V, input_mask = prepare_mha_input(
                B, N, S, H, L, torch.int8
            )

            torch.ops.pace.multi_head_attention(
                input_Q, input_K, input_V, input_mask, use_KQ
            )

    def test_mha_invalid_KQ_value(
        self,
    ):

        B = 64
        N = 32
        S = 512
        H = 128
        L = 512

        use_KQ = random.randint(-100, 100)

        with self.assertRaisesRegex(
            RuntimeError, "MHA attention requires use_KQ to be 0 or 1"
        ):

            input_Q, input_K, input_V, input_mask = prepare_mha_input(
                B, N, S, H, L, torch.float32
            )

            torch.ops.pace.multi_head_attention(
                input_Q, input_K, input_V, input_mask, use_KQ
            )

    def test_mha_input_shape(self):

        B = 64
        N = 32
        S = 512
        H = 128
        L = 512

        proj_shape = (B * N, -1, H)
        proj_shape_mask = (B, L)

        dtype = torch.float32

        use_KQ = 0

        input_Q, input_K, input_V, input_mask = prepare_mha_input(B, N, S, H, L, dtype)

        with self.assertRaisesRegex(RuntimeError, "MHA attention requires 4D inputs"):
            torch.ops.pace.multi_head_attention(
                input_Q.view(*proj_shape), input_K, input_V, input_mask, use_KQ
            )

        with self.assertRaisesRegex(RuntimeError, "MHA attention requires 4D inputs"):
            torch.ops.pace.multi_head_attention(
                input_Q, input_K.view(*proj_shape), input_V, input_mask, use_KQ
            )

        with self.assertRaisesRegex(RuntimeError, "MHA attention requires 4D inputs"):
            torch.ops.pace.multi_head_attention(
                input_Q, input_K, input_V.view(*proj_shape), input_mask, use_KQ
            )

        with self.assertRaisesRegex(
            RuntimeError, "MHA attention requires 4D attention mask"
        ):
            torch.ops.pace.multi_head_attention(
                input_Q, input_K, input_V, input_mask.view(*proj_shape_mask), use_KQ
            )
