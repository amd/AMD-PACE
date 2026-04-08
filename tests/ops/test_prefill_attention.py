# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for torch.ops.pace.prefill_attention correctness.

Validates BRGeMM tiled prefill against PyTorch FP32 reference SDPA
within BF16 tolerance across various GQA/MHA configs, including
batched inputs with left/right padding.

Run: python -m pytest tests/ops/test_prefill_attention.py -v
  or: python -m unittest -v tests.ops.test_prefill_attention
"""

import unittest

import torch

import pace  # noqa: F401

BF16_ATOL = 0.125
BF16_RTOL = 0.1


def _reference_causal_sdpa(query, key, value):
    """PyTorch FP32 reference: causal SDPA with GQA support.

    Handles q_len <= kv_len by building an explicit offset causal mask
    (Q positions at the end of KV range, matching prefill_tile behavior).

    Args:
        query: [B, N_q, q_len, H] bfloat16
        key:   [B, N_kv, kv_len, H] bfloat16
        value: [B, N_kv, kv_len, H] bfloat16

    Returns:
        [B, N_q, q_len, H] bfloat16
    """
    q_f = query.float()
    k_f = key.float()
    v_f = value.float()

    q_len = q_f.size(2)
    kv_len = k_f.size(2)
    num_q_heads = q_f.size(1)
    num_kv_heads = k_f.size(1)
    n_rep = num_q_heads // num_kv_heads

    if n_rep > 1:
        k_f = k_f.repeat_interleave(n_rep, dim=1)
        v_f = v_f.repeat_interleave(n_rep, dim=1)

    if q_len == kv_len:
        out = torch.nn.functional.scaled_dot_product_attention(
            q_f, k_f, v_f, is_causal=True
        )
    else:
        q_positions = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
        kv_positions = torch.arange(kv_len).unsqueeze(0)
        mask = (kv_positions > q_positions).unsqueeze(0).unsqueeze(0)
        attn_mask = torch.where(mask, float("-inf"), 0.0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_f, k_f, v_f, attn_mask=attn_mask
        )

    return out.to(torch.bfloat16)


class TestPrefillAttention(unittest.TestCase):
    """Correctness tests: torch.ops.pace.prefill_attention vs PyTorch SDPA."""

    def _run_test(self, B, S, N_q, N_kv, H):
        q = torch.randn(B, N_q, S, H, dtype=torch.bfloat16)
        k = torch.randn(B, N_kv, S, H, dtype=torch.bfloat16)
        v = torch.randn(B, N_kv, S, H, dtype=torch.bfloat16)

        pace_out = torch.ops.pace.prefill_attention(q, k, v, [], [])
        ref_out = _reference_causal_sdpa(q, k, v)

        torch.testing.assert_close(pace_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_small_mha(self):
        """MHA: B=1, S=32, N_q=N_kv=4, H=64."""
        self._run_test(B=1, S=32, N_q=4, N_kv=4, H=64)

    def test_small_gqa(self):
        """GQA: B=1, S=32, N_q=8, N_kv=4, H=64."""
        self._run_test(B=1, S=32, N_q=8, N_kv=4, H=64)

    def test_llama8b_config(self):
        """Llama-3.1-8B: B=1, S=128, N_q=32, N_kv=8, H=128."""
        self._run_test(B=1, S=128, N_q=32, N_kv=8, H=128)

    def test_multi_block(self):
        """Prefill spanning multiple blocks: S=256."""
        self._run_test(B=1, S=256, N_q=8, N_kv=4, H=64)

    def test_long_seq(self):
        """Long sequence: S=512."""
        self._run_test(B=1, S=512, N_q=32, N_kv=8, H=128)

    def test_batch_2(self):
        """Batched: B=2."""
        self._run_test(B=2, S=128, N_q=32, N_kv=8, H=128)

    def test_head_dim_256(self):
        """HD=256: exercises non-128 paths."""
        self._run_test(B=1, S=64, N_q=4, N_kv=4, H=256)

    def test_short_seq(self):
        """Short: S=16, still goes through tiled path."""
        self._run_test(B=1, S=16, N_q=8, N_kv=4, H=64)

    def test_no_nan(self):
        """Output contains no NaN or Inf."""
        q = torch.randn(2, 32, 128, 128, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 128, 128, dtype=torch.bfloat16)
        v = torch.randn(2, 8, 128, 128, dtype=torch.bfloat16)

        out = torch.ops.pace.prefill_attention(q, k, v, [], [])
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_matches_existing_gqa_op(self):
        """Compare against torch.ops.pace.grouped_query_attention."""
        B, S, N_q, N_kv, H = 1, 64, 32, 8, 128
        q = torch.randn(B, N_q, S, H, dtype=torch.bfloat16)
        k = torch.randn(B, N_kv, S, H, dtype=torch.bfloat16)
        v = torch.randn(B, N_kv, S, H, dtype=torch.bfloat16)

        mask = torch.triu(
            torch.full((S, S), float("-inf"), dtype=torch.bfloat16), diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        pace_prefill = torch.ops.pace.prefill_attention(q, k, v, [], [])
        pace_gqa = torch.ops.pace.grouped_query_attention(q, k, v, mask)

        torch.testing.assert_close(
            pace_prefill, pace_gqa, atol=BF16_ATOL, rtol=BF16_RTOL
        )

    def test_left_padding(self):
        """Left-padded batch: different real Q lengths, shared KV length.

        Simulates BMC usage where Q/K/V are all left-padded to the same
        length. Pad positions contain pad token embeddings (not zeros).
        The C++ op uses q_offsets to skip pad tokens.
        """
        N_q, N_kv, H = 8, 4, 64
        S_padded = 12

        # Generate real tokens per sequence
        q0_real = torch.randn(1, N_q, 12, H, dtype=torch.bfloat16)
        k0_real = torch.randn(1, N_kv, 12, H, dtype=torch.bfloat16)
        v0_real = torch.randn(1, N_kv, 12, H, dtype=torch.bfloat16)
        ref0 = _reference_causal_sdpa(q0_real, k0_real, v0_real)

        q1_real = torch.randn(1, N_q, 8, H, dtype=torch.bfloat16)
        k1_real = torch.randn(1, N_kv, 8, H, dtype=torch.bfloat16)
        v1_real = torch.randn(1, N_kv, 8, H, dtype=torch.bfloat16)
        ref1 = _reference_causal_sdpa(q1_real, k1_real, v1_real)

        # Left-pad: real tokens at end, pad token embeddings at start
        q = torch.randn(2, N_q, S_padded, H, dtype=torch.bfloat16)
        k = torch.randn(2, N_kv, S_padded, H, dtype=torch.bfloat16)
        v = torch.randn(2, N_kv, S_padded, H, dtype=torch.bfloat16)

        q[0, :, :] = q0_real
        k[0, :, :] = k0_real
        v[0, :, :] = v0_real
        q[1, :, 4:] = q1_real
        k[1, :, 4:] = k1_real
        v[1, :, 4:] = v1_real

        out = torch.ops.pace.prefill_attention(q, k, v, [0, 4], [12, 8])

        torch.testing.assert_close(
            out[0], ref0.squeeze(0), atol=BF16_ATOL, rtol=BF16_RTOL
        )
        torch.testing.assert_close(
            out[1, :, 4:], ref1.squeeze(0), atol=BF16_ATOL, rtol=BF16_RTOL
        )

    def test_right_padding(self):
        """Right-padded batch: real tokens at start."""
        N_q, N_kv, H = 8, 4, 64
        S_padded = 12

        q0_real = torch.randn(1, N_q, 12, H, dtype=torch.bfloat16)
        k0_real = torch.randn(1, N_kv, 12, H, dtype=torch.bfloat16)
        v0_real = torch.randn(1, N_kv, 12, H, dtype=torch.bfloat16)
        ref0 = _reference_causal_sdpa(q0_real, k0_real, v0_real)

        q1_real = torch.randn(1, N_q, 8, H, dtype=torch.bfloat16)
        k1_real = torch.randn(1, N_kv, 8, H, dtype=torch.bfloat16)
        v1_real = torch.randn(1, N_kv, 8, H, dtype=torch.bfloat16)
        ref1 = _reference_causal_sdpa(q1_real, k1_real, v1_real)

        q = torch.randn(2, N_q, S_padded, H, dtype=torch.bfloat16)
        k = torch.randn(2, N_kv, S_padded, H, dtype=torch.bfloat16)
        v = torch.randn(2, N_kv, S_padded, H, dtype=torch.bfloat16)

        q[0, :, :] = q0_real
        k[0, :, :] = k0_real
        v[0, :, :] = v0_real
        q[1, :, :8] = q1_real
        k[1, :, :8] = k1_real
        v[1, :, :8] = v1_real

        out = torch.ops.pace.prefill_attention(q, k, v, [0, 0], [12, 8])

        torch.testing.assert_close(
            out[0], ref0.squeeze(0), atol=BF16_ATOL, rtol=BF16_RTOL
        )
        torch.testing.assert_close(
            out[1, :, :8], ref1.squeeze(0), atol=BF16_ATOL, rtol=BF16_RTOL
        )


if __name__ == "__main__":
    unittest.main()
