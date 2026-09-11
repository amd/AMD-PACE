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
    per_head_mask: bool = False,
):
    """
    Prepare inputs for GQA tests

    Args:
        B, S, H, L, num_q_heads, num_kv_heads
        dtype: Data type of input tensor
        per_head_mask: If True, create mask with shape [B, num_q_heads, S, L]
                       (per-head sink mask). Otherwise [B, 1, S, L] (broadcast).

    Returns:
        Query, Key, Value, Attention Mask tensors
    """

    shape_q = (B, num_q_heads, S, H)
    shape_kv = (B, num_kv_heads, L, H)
    if per_head_mask:
        shape_mask = (B, num_q_heads, S, L)
    else:
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
        max_examples=10,
    )
    @given(
        batch=st.sampled_from([1, 16, 64]),
        head_dim=st.sampled_from([1, 96, 128]),
        seq_len=st.sampled_from([1, 32, 128]),
        KV_len=st.sampled_from([1, 32, 128]),
        num_q_heads=st.sampled_from([32]),
        num_kv_heads=st.sampled_from([8]),
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

        B = 2
        S = 32
        H = 64
        L = 32
        num_q_heads = 16
        num_kv_heads = 4

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

        B = 2
        S = 32
        H = 64
        L = 32
        num_q_heads = 16
        num_kv_heads = 4

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

        B = 2
        S = 32
        H = 64
        L = 32
        num_q_heads = 15
        num_kv_heads = 4

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

    @settings(
        deadline=None,
        max_examples=10,
    )
    @given(
        batch=st.sampled_from([1, 16, 64]),
        head_dim=st.sampled_from([1, 96, 128]),
        seq_len=st.sampled_from([1, 32, 128]),
        KV_len=st.sampled_from([1, 32, 128]),
        num_q_heads=st.sampled_from([32]),
        num_kv_heads=st.sampled_from([8]),
        input_dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    def test_gqa_per_head_sink_mask(
        self, batch, seq_len, head_dim, KV_len, num_q_heads, num_kv_heads, input_dtype
    ):
        """Mask shape [B, num_q_heads, S, L] triggers the per-head sink path
        where the kernel reshapes to [B, num_kv_heads, N_rep, S, L]."""

        input_Q, input_K, input_V, input_mask = prepare_gqa_input(
            batch,
            seq_len,
            head_dim,
            KV_len,
            num_q_heads,
            num_kv_heads,
            input_dtype,
            per_head_mask=True,
        )
        pace_gqa_output = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, input_mask
        )

        Torch_gqa_Direct_output = Torch_direct_GQA(
            input_Q, input_K, input_V, input_mask, input_dtype
        )

        threshold = 1e-5
        if input_dtype == torch.bfloat16:
            threshold = 1e-1

        self.assertEqual(
            Torch_gqa_Direct_output,
            pace_gqa_output,
            atol=threshold,
            rtol=threshold,
        )

    @settings(
        deadline=None,
        max_examples=10,
    )
    @given(
        batch=st.sampled_from([1, 16, 64]),
        head_dim=st.sampled_from([1, 96, 128]),
        seq_len=st.sampled_from([1, 32, 128]),
        KV_len=st.sampled_from([1, 32, 128]),
        num_q_heads=st.sampled_from([32]),
        num_kv_heads=st.sampled_from([8]),
        input_dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    def test_gqa_broadcast_mask(
        self, batch, seq_len, head_dim, KV_len, num_q_heads, num_kv_heads, input_dtype
    ):
        """Mask shape [B, 1, S, L] triggers the broadcast path
        where the kernel reshapes to [B, 1, 1, S, L]."""

        input_Q, input_K, input_V, input_mask = prepare_gqa_input(
            batch,
            seq_len,
            head_dim,
            KV_len,
            num_q_heads,
            num_kv_heads,
            input_dtype,
            per_head_mask=False,
        )
        pace_gqa_output = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, input_mask
        )

        Torch_gqa_Direct_output = Torch_direct_GQA(
            input_Q, input_K, input_V, input_mask, input_dtype
        )

        threshold = 1e-5
        if input_dtype == torch.bfloat16:
            threshold = 1e-1

        self.assertEqual(
            Torch_gqa_Direct_output,
            pace_gqa_output,
            atol=threshold,
            rtol=threshold,
        )

    def test_gqa_per_head_sink_mask_distinct_values(self):
        """Verify per-head mask actually applies different mask values per head,
        not just broadcasting a single mask to all heads."""

        B = 1
        S = 1
        H = 64
        L = 8
        num_q_heads = 8
        num_kv_heads = 4
        dtype = torch.float32

        input_Q, input_K, input_V, _ = prepare_gqa_input(
            B, S, H, L, num_q_heads, num_kv_heads, dtype
        )

        broadcast_mask = torch.zeros(B, 1, S, L, dtype=dtype)
        per_head_mask = torch.zeros(B, num_q_heads, S, L, dtype=dtype)
        per_head_mask[:, 0, :, -1] = -1e9

        out_broadcast = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, broadcast_mask
        )
        out_per_head = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, per_head_mask
        )

        self.assertFalse(
            torch.allclose(out_broadcast, out_per_head, atol=1e-5, rtol=1e-5),
            "Per-head sink mask should produce different output than broadcast mask "
            "when mask values differ across heads",
        )

    def test_gqa_per_head_sink_mask_uniform_matches_broadcast(self):
        """When per-head mask has identical values across all heads it should
        produce the same result as the equivalent broadcast mask."""

        B = 1
        S = 1
        H = 64
        L = 8
        num_q_heads = 8
        num_kv_heads = 4
        dtype = torch.float32

        input_Q, input_K, input_V, _ = prepare_gqa_input(
            B, S, H, L, num_q_heads, num_kv_heads, dtype
        )

        broadcast_mask = torch.randn(B, 1, S, L, dtype=dtype)
        per_head_mask = broadcast_mask.expand(B, num_q_heads, S, L).contiguous()

        out_broadcast = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, broadcast_mask
        )
        out_per_head = torch.ops.pace.grouped_query_attention(
            input_Q, input_K, input_V, per_head_mask
        )

        self.assertEqual(
            out_broadcast,
            out_per_head,
            atol=1e-5,
            rtol=1e-5,
        )
