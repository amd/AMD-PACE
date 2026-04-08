# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for SLAB attention kernel correctness: causal, sliding window,
sinks, combined sliding+sinks, multi-token decode, GQA, and MHA.

Run: python -m unittest -v tests.attention.test_slab_attention
"""

import math

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401

# BF16 attention tolerance: 2 ULPs absolute, PyTorch's bf16 default relative
BF16_ATOL = 0.016
BF16_RTOL = 1.6e-2


def _create_pool(total_blocks=64, num_kv_heads=4, head_dim=64, block_size=16):
    return torch.classes.pace.SlabPool(total_blocks, num_kv_heads, head_dim, block_size)


def _reference_causal_attention(q, k, v, scale):
    """PyTorch FP32 reference for causal attention with proper offset mask.

    When q_len < kv_len (decode/MTD), the Q positions are at the END
    of the sequence: [kv_len - q_len, ..., kv_len - 1]. is_causal=True
    doesn't handle this offset, so we build an explicit mask.
    """
    bs, q_len, h, d = q.shape
    kv_len = k.shape[1]

    q_f = q.transpose(1, 2).float()
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale

    q_positions = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
    kv_positions = torch.arange(kv_len).unsqueeze(0)
    mask = kv_positions > q_positions
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f)
    return out.transpose(1, 2).to(torch.bfloat16)


def _setup_pool_with_kv(bs, seq_len, h_kv, d, blk_size=16, max_seq=None):
    """Create pool, sequences, and fill with random KV data."""
    if max_seq is None:
        max_seq = seq_len + 256
    bps = (max_seq + blk_size - 1) // blk_size
    total_blocks = bps * bs + 64
    pool = _create_pool(total_blocks, h_kv, d, blk_size)
    seq_ids = list(range(bs))
    for sid in seq_ids:
        pool.create_sequence(sid, max_seq)
    k = torch.randn(bs, seq_len, h_kv, d, dtype=torch.bfloat16)
    v = torch.randn(bs, seq_len, h_kv, d, dtype=torch.bfloat16)
    pool.cache_update(seq_ids, k, v, [])
    return pool, seq_ids, k, v


class TestSlabCausalAttention(TestCase):
    def test_prefill_accuracy(self):
        """Prefill matches PyTorch SDPA within BF16 tolerance."""
        BS, S, H_q, H_kv, D = 1, 32, 8, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k, v = _setup_pool_with_kv(BS, S, H_kv, D)
        q = torch.randn(BS, S, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))

        k_exp = k.repeat_interleave(H_q // H_kv, dim=2)
        v_exp = v.repeat_interleave(H_q // H_kv, dim=2)
        ref_out = _reference_causal_attention(q, k_exp, v_exp, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_prefill_accuracy_gqa_hd128(self):
        """Llama-3.1-8B config: NQ=32, NKV=8, HD=128, block_size=64."""
        BS, S, H_q, H_kv, D = 1, 128, 32, 8, 128
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k, v = _setup_pool_with_kv(BS, S, H_kv, D, blk_size=64)
        q = torch.randn(BS, S, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [S], [], scale, 0, torch.tensor([]))

        k_exp = k.repeat_interleave(H_q // H_kv, dim=2)
        v_exp = v.repeat_interleave(H_q // H_kv, dim=2)
        ref_out = _reference_causal_attention(q, k_exp, v_exp, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_decode_after_prefill(self):
        """Decode step matches reference after prefill."""
        BS, S, H_q, H_kv, D = 1, 64, 8, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(BS, S, H_kv, D)

        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        pool.cache_update(seq_ids, k_dec, v_dec, [])

        q_dec = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q_dec, [], [], scale, 0, torch.tensor([]))

        k_full = torch.cat([k_pf, k_dec], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        v_full = torch.cat([v_pf, v_dec], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        ref_out = _reference_causal_attention(q_dec, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_decode_accuracy_bs8_gqa(self):
        """BS=8 decode with GQA (n_rep=4)."""
        BS, S, H_q, H_kv, D = 8, 64, 32, 8, 128
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(BS, S, H_kv, D, blk_size=64)

        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        pool.cache_update(seq_ids, k_dec, v_dec, [])

        q_dec = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q_dec, [], [], scale, 0, torch.tensor([]))

        k_full = torch.cat([k_pf, k_dec], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        v_full = torch.cat([v_pf, v_dec], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        ref_out = _reference_causal_attention(q_dec, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_decode_accuracy_mha(self):
        """MHA (n_rep=1), triggers DECODE_HEAD dispatch path."""
        BS, S, H_q, H_kv, D = 2, 32, 8, 8, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(BS, S, H_kv, D)

        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        pool.cache_update(seq_ids, k_dec, v_dec, [])

        q_dec = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q_dec, [], [], scale, 0, torch.tensor([]))

        k_full = torch.cat([k_pf, k_dec], dim=1)
        v_full = torch.cat([v_pf, v_dec], dim=1)
        ref_out = _reference_causal_attention(q_dec, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_decode_accuracy_head_dim_256(self):
        """head_dim=256 exercises non-128 fallback paths in SV accumulation."""
        BS, S, H_q, H_kv, D = 1, 32, 4, 4, 256
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(BS, S, H_kv, D)

        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        pool.cache_update(seq_ids, k_dec, v_dec, [])

        q_dec = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q_dec, [], [], scale, 0, torch.tensor([]))

        k_full = torch.cat([k_pf, k_dec], dim=1)
        v_full = torch.cat([v_pf, v_dec], dim=1)
        ref_out = _reference_causal_attention(q_dec, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_multi_token_decode_accuracy(self):
        """MTD path (1 < q_len <= 64) matches reference."""
        BS, H_q, H_kv, D = 1, 8, 4, 64
        MTD_LEN = 8
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(BS, 32, H_kv, D)

        k_mtd = torch.randn(BS, MTD_LEN, H_kv, D, dtype=torch.bfloat16)
        v_mtd = torch.randn(BS, MTD_LEN, H_kv, D, dtype=torch.bfloat16)
        pool.cache_update(seq_ids, k_mtd, v_mtd, [])

        q_mtd = torch.randn(BS, MTD_LEN, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(
            seq_ids, q_mtd, [MTD_LEN], [], scale, 0, torch.tensor([])
        )

        k_full = torch.cat([k_pf, k_mtd], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        v_full = torch.cat([v_pf, v_mtd], dim=1).repeat_interleave(H_q // H_kv, dim=2)
        ref_out = _reference_causal_attention(q_mtd, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_batched_decode_no_nan(self):
        """Multiple sequences in a single decode call produce valid output."""
        BS, H_q, H_kv, D = 4, 8, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool = _create_pool(total_blocks=256, num_kv_heads=H_kv, head_dim=D)
        seq_ids = list(range(BS))
        for sid in seq_ids:
            pool.create_sequence(sid, 256)
            pool.cache_update(
                [sid],
                torch.randn(1, 32, H_kv, D, dtype=torch.bfloat16),
                torch.randn(1, 32, H_kv, D, dtype=torch.bfloat16),
                [],
            )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        out = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        self.assertEqual(out.shape, (BS, 1, H_q, D))
        self.assertFalse(torch.isnan(out).any())

    def test_long_decode_across_blocks(self):
        """Decode across multiple block boundaries stays within BF16 bounds."""
        BS, H_q, H_kv, D = 1, 4, 4, 64
        scale = 1.0 / math.sqrt(D)
        PREFILL = 16
        DECODE_STEPS = 64

        pool, seq_ids, k_pf, v_pf = _setup_pool_with_kv(
            BS, PREFILL, H_kv, D, blk_size=16, max_seq=PREFILL + DECODE_STEPS + 64
        )

        all_k, all_v = [k_pf], [v_pf]
        for _ in range(DECODE_STEPS):
            k_new = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
            v_new = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
            pool.cache_update(seq_ids, k_new, v_new, [])
            all_k.append(k_new)
            all_v.append(v_new)

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))

        k_full = torch.cat(all_k, dim=1)
        v_full = torch.cat(all_v, dim=1)
        ref_out = _reference_causal_attention(q, k_full, v_full, scale)

        self.assertEqual(slab_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)


class TestSlabSlidingWindowAttention(TestCase):
    def test_sliding_window_changes_output(self):
        BS, H_q, H_kv, D = 1, 4, 4, 64
        window = 8
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, 40, H_kv, D, blk_size=16)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        out_full = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        out_windowed = pool.attention(
            seq_ids, q, [], [], scale, window, torch.tensor([])
        )

        self.assertEqual(out_full.shape, out_windowed.shape)
        self.assertFalse(
            torch.allclose(out_full.float(), out_windowed.float(), atol=1e-3)
        )

    def test_sliding_window_prefill(self):
        BS, S, H_q, H_kv, D = 1, 64, 4, 4, 64
        window = 16
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, S, H_kv, D, blk_size=16)

        q = torch.randn(BS, S, H_q, D, dtype=torch.bfloat16)
        out_full = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        out_windowed = pool.attention(
            seq_ids, q, [], [], scale, window, torch.tensor([])
        )

        self.assertFalse(
            torch.allclose(out_full.float(), out_windowed.float(), atol=1e-3)
        )

    def test_sliding_window_decode_no_nan(self):
        BS, H_q, H_kv, D = 1, 4, 4, 64
        window = 8
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, 40, H_kv, D, blk_size=16)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        out = pool.attention(seq_ids, q, [], [], scale, window, torch.tensor([]))
        self.assertFalse(torch.isnan(out).any())


class TestSlabSinksAttention(TestCase):
    def test_sinks_changes_output(self):
        BS, S, H_q, H_kv, D = 1, 16, 4, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, S, H_kv, D)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        sinks = torch.ones(H_q, dtype=torch.float32) * 5.0

        out_no_sinks = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        out_with_sinks = pool.attention(seq_ids, q, [], [], scale, 0, sinks)

        self.assertFalse(
            torch.allclose(out_no_sinks.float(), out_with_sinks.float(), atol=1e-3)
        )

    def test_sinks_during_prefill(self):
        BS, S, H_q, H_kv, D = 1, 32, 4, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, S, H_kv, D)

        q = torch.randn(BS, S, H_q, D, dtype=torch.bfloat16)
        sinks = torch.ones(H_q, dtype=torch.float32) * 5.0

        out_no_sinks = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        out_with_sinks = pool.attention(seq_ids, q, [], [], scale, 0, sinks)

        self.assertFalse(
            torch.allclose(out_no_sinks.float(), out_with_sinks.float(), atol=1e-3)
        )


class TestSlabCombinedSlidingAndSinks(TestCase):
    def test_combined_sliding_and_sinks(self):
        BS, H_q, H_kv, D = 1, 4, 4, 64
        window = 8
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, 20, H_kv, D)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        sinks = torch.ones(H_q, dtype=torch.float32) * 3.0
        out = pool.attention(seq_ids, q, [], [], scale, window, sinks)
        self.assertEqual(out.shape, (BS, 1, H_q, D))
        self.assertFalse(torch.isnan(out).any())

    def test_sinks_differ_from_no_sinks(self):
        BS, H_q, H_kv, D = 1, 4, 4, 64
        scale = 1.0 / math.sqrt(D)

        pool, seq_ids, _, _ = _setup_pool_with_kv(BS, 20, H_kv, D)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )

        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        sinks = torch.ones(H_q, dtype=torch.float32) * 3.0

        out_plain = pool.attention(seq_ids, q, [], [], scale, 0, torch.tensor([]))
        out_sinks = pool.attention(seq_ids, q, [], [], scale, 0, sinks)

        self.assertFalse(
            torch.allclose(out_plain.float(), out_sinks.float(), atol=1e-3)
        )


class TestSlabRaggedPadding(TestCase):
    """Test left-padding handling via q_start_offsets and cache_update src_offset."""

    def _run_padded_vs_individual(self, batch_sizes, head_config, blk_size=16):
        """Compare batched padded attention against per-sequence references.

        Creates sequences with random lengths, left-pads to max length,
        runs SLAB attention on the padded batch, and verifies each sequence
        matches a standalone non-padded run.
        """
        H_q, H_kv, D = head_config
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        for BS in batch_sizes:
            # Random per-sequence lengths between 3 and 20
            seq_lens = [torch.randint(3, 21, (1,)).item() for _ in range(BS)]
            max_len = max(seq_lens)

            # Per-sequence reference: run each independently (no padding)
            ref_outputs = []
            for i in range(BS):
                S = seq_lens[i]
                bps = (S + 256 + blk_size - 1) // blk_size
                pool_i = _create_pool(bps + 4, H_kv, D, blk_size)
                pool_i.create_sequence(0, S + 256)
                k_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
                v_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
                q_i = torch.randn(1, S, H_q, D, dtype=torch.bfloat16)
                pool_i.cache_update([0], k_i, v_i, [S])
                out_i = pool_i.attention([0], q_i, [S], [], scale, 0, torch.tensor([]))
                ref_outputs.append((q_i, k_i, v_i, out_i))

            # Batched padded: left-pad all sequences to max_len
            Q_padded = torch.zeros(BS, max_len, H_q, D, dtype=torch.bfloat16)
            K_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
            V_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
            for i in range(BS):
                S = seq_lens[i]
                q_i, k_i, v_i, _ = ref_outputs[i]
                Q_padded[i, max_len - S :] = q_i[0]
                K_padded[i, max_len - S :] = k_i[0]
                V_padded[i, max_len - S :] = v_i[0]

            # Create batched pool and run with seq_lens (triggers offset logic)
            bps = (max_len + 256 + blk_size - 1) // blk_size
            pool = _create_pool(bps * BS + 64, H_kv, D, blk_size)
            seq_ids = list(range(BS))
            for sid in seq_ids:
                pool.create_sequence(sid, max_len + 256)
            pool.cache_update(seq_ids, K_padded, V_padded, seq_lens)
            batched_out = pool.attention(
                seq_ids, Q_padded, seq_lens, [], scale, 0, torch.tensor([])
            )

            # Verify each sequence matches its standalone reference
            for i in range(BS):
                S = seq_lens[i]
                _, _, _, ref_out = ref_outputs[i]
                slab_seq = batched_out[i, max_len - S :]
                self.assertEqual(
                    slab_seq,
                    ref_out[0],
                    atol=BF16_ATOL,
                    rtol=BF16_RTOL,
                    msg=f"BS={BS}, seq {i} (len={S}) mismatch",
                )

    def test_padded_mha(self):
        """MHA (n_rep=1) with random left-padding, multiple batch sizes."""
        self._run_padded_vs_individual([2, 4, 8], (4, 4, 64))

    def test_padded_gqa(self):
        """GQA (n_rep=4) with random left-padding, multiple batch sizes."""
        self._run_padded_vs_individual([2, 4, 8], (8, 2, 64))

    def test_padded_gqa_hd128(self):
        """GQA with head_dim=128 and random left-padding."""
        self._run_padded_vs_individual([2, 4], (32, 8, 128), blk_size=64)

    def test_padded_extreme_padding(self):
        """Test with extreme padding ratios (short seqs padded to long max)."""
        H_q, H_kv, D = 4, 4, 64
        scale = 1.0 / math.sqrt(D)
        blk_size = 16
        torch.manual_seed(123)

        # seq_lens: 3, 3, 3, 50 → max_len=50, extreme padding for first 3
        seq_lens = [3, 3, 3, 50]
        BS = len(seq_lens)
        max_len = max(seq_lens)

        # Per-sequence references
        ref_outputs = []
        for i in range(BS):
            S = seq_lens[i]
            bps = (S + 256 + blk_size - 1) // blk_size
            pool_i = _create_pool(bps + 4, H_kv, D, blk_size)
            pool_i.create_sequence(0, S + 256)
            k_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
            v_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
            q_i = torch.randn(1, S, H_q, D, dtype=torch.bfloat16)
            pool_i.cache_update([0], k_i, v_i, [S])
            out_i = pool_i.attention([0], q_i, [S], [], scale, 0, torch.tensor([]))
            ref_outputs.append((q_i, k_i, v_i, out_i))

        # Batched padded
        Q_padded = torch.zeros(BS, max_len, H_q, D, dtype=torch.bfloat16)
        K_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
        V_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
        for i in range(BS):
            S = seq_lens[i]
            q_i, k_i, v_i, _ = ref_outputs[i]
            Q_padded[i, max_len - S :] = q_i[0]
            K_padded[i, max_len - S :] = k_i[0]
            V_padded[i, max_len - S :] = v_i[0]

        bps = (max_len + 256 + blk_size - 1) // blk_size
        pool = _create_pool(bps * BS + 64, H_kv, D, blk_size)
        seq_ids = list(range(BS))
        for sid in seq_ids:
            pool.create_sequence(sid, max_len + 256)
        pool.cache_update(seq_ids, K_padded, V_padded, seq_lens)
        batched_out = pool.attention(
            seq_ids, Q_padded, seq_lens, [], scale, 0, torch.tensor([])
        )

        for i in range(BS):
            S = seq_lens[i]
            _, _, _, ref_out = ref_outputs[i]
            slab_seq = batched_out[i, max_len - S :]
            self.assertEqual(
                slab_seq,
                ref_out[0],
                atol=BF16_ATOL,
                rtol=BF16_RTOL,
                msg=f"Extreme padding: seq {i} (len={S}, padded to {max_len})",
            )


def _reference_sliding_window_attention(q, k, v, scale, window):
    """FP32 reference for causal sliding-window attention."""
    bs, q_len, h, d = q.shape
    kv_len = k.shape[1]
    q_f = q.transpose(1, 2).float()
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale

    q_positions = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
    kv_positions = torch.arange(kv_len).unsqueeze(0)
    causal = kv_positions > q_positions
    outside_window = (q_positions - kv_positions) >= window
    mask = causal | outside_window
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_f)
    return out.transpose(1, 2).to(torch.bfloat16)


class TestSlabSlidingWindowCorrectness(TestCase):
    """Verify sliding window attention matches a reference implementation."""

    def test_sliding_window_decode_correctness(self):
        """Decode with sliding window matches reference."""
        BS, S, H_q, H_kv, D = 1, 32, 4, 4, 64
        window = 8
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        pool, seq_ids, k, v = _setup_pool_with_kv(BS, S, H_kv, D)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )
        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [], [], scale, window, torch.tensor([]))

        # Re-seed and regenerate to match pool data
        torch.manual_seed(42)
        pool2, seq_ids2, k2, v2 = _setup_pool_with_kv(BS, S, H_kv, D)
        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        q2 = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)

        full_k = torch.cat([k2, k_dec], dim=1)
        full_v = torch.cat([v2, v_dec], dim=1)
        k_exp = full_k.repeat_interleave(H_q // H_kv, dim=2)
        v_exp = full_v.repeat_interleave(H_q // H_kv, dim=2)
        ref = _reference_sliding_window_attention(q2, k_exp, v_exp, scale, window)

        self.assertEqual(slab_out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    def test_sliding_window_prefill_correctness(self):
        """Prefill with sliding window matches reference."""
        BS, S, H_q, H_kv, D = 1, 32, 4, 4, 64
        window = 8
        scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        pool, seq_ids, k, v = _setup_pool_with_kv(BS, S, H_kv, D)
        q = torch.randn(BS, S, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [], [], scale, window, torch.tensor([]))

        k_exp = k.repeat_interleave(H_q // H_kv, dim=2)
        v_exp = v.repeat_interleave(H_q // H_kv, dim=2)
        ref = _reference_sliding_window_attention(q, k_exp, v_exp, scale, window)

        self.assertEqual(slab_out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)


def _reference_sink_attention(q, k, v, scale, sinks):
    """FP32 reference for causal attention with per-head sink biases.

    The C++ implementation prepends a virtual token with score=sink_bias
    and V=zeros to the online softmax.  In closed form, this is equivalent
    to: attn_weights = softmax([sink_bias, real_scores...]) and
    output = attn_weights[1:] @ V  (virtual token contributes zero to V).
    """
    bs, q_len, h, d = q.shape
    kv_len = k.shape[1]
    q_f = q.transpose(1, 2).float()  # [bs, h, q, d]
    k_f = k.transpose(1, 2).float()
    v_f = v.transpose(1, 2).float()

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # [bs, h, q, kv]

    q_pos = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
    kv_pos = torch.arange(kv_len).unsqueeze(0)
    causal_mask = kv_pos > q_pos
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Prepend virtual sink score per head: [bs, h, q, 1+kv]
    sink_col = sinks.view(1, h, 1, 1).float().expand(bs, h, q_len, 1)
    scores_with_sink = torch.cat([sink_col, scores], dim=-1)

    attn = torch.softmax(scores_with_sink, dim=-1)  # [bs, h, q, 1+kv]

    # Virtual token V=0, so only the real kv slice contributes
    out = torch.matmul(attn[:, :, :, 1:], v_f)  # [bs, h, q, d]
    return out.transpose(1, 2).to(torch.bfloat16)


class TestSlabSinksCorrectness(TestCase):
    """Verify attention sinks match the reference virtual-token formulation."""

    def test_sinks_decode_correctness(self):
        """Decode with sinks matches reference (virtual zero-V token formulation)."""
        BS, S, H_q, H_kv, D = 1, 32, 4, 4, 64
        scale = 1.0 / math.sqrt(D)
        sinks = torch.ones(H_q, dtype=torch.float32) * 3.0
        torch.manual_seed(42)

        pool, seq_ids, k, v = _setup_pool_with_kv(BS, S, H_kv, D)
        pool.cache_update(
            seq_ids,
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16),
            [],
        )
        q = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)
        slab_out = pool.attention(seq_ids, q, [], [], scale, 0, sinks)

        torch.manual_seed(42)
        _, _, k2, v2 = _setup_pool_with_kv(BS, S, H_kv, D)
        k_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        v_dec = torch.randn(BS, 1, H_kv, D, dtype=torch.bfloat16)
        q2 = torch.randn(BS, 1, H_q, D, dtype=torch.bfloat16)

        full_k = torch.cat([k2, k_dec], dim=1)
        full_v = torch.cat([v2, v_dec], dim=1)
        k_exp = full_k.repeat_interleave(H_q // H_kv, dim=2)
        v_exp = full_v.repeat_interleave(H_q // H_kv, dim=2)
        ref = _reference_sink_attention(q2, k_exp, v_exp, scale, sinks)

        self.assertEqual(slab_out, ref, atol=BF16_ATOL, rtol=BF16_RTOL)


class TestSlabQStartOffsets(TestCase):
    """Test q_start_offsets with non-empty values for 4D padded input."""

    def test_4d_padded_matches_individual(self):
        """4D padded Q with offset-based attention matches per-sequence results."""
        H_q, H_kv, D = 8, 4, 64
        scale = 1.0 / math.sqrt(D)
        blk_size = 16
        torch.manual_seed(42)

        # Two sequences with different lengths
        seq_lens = [5, 12]
        BS = len(seq_lens)
        max_len = max(seq_lens)

        # Per-sequence reference
        ref_outputs = []
        for i in range(BS):
            S = seq_lens[i]
            bps = (S + 256 + blk_size - 1) // blk_size
            pool_i = _create_pool(bps + 4, H_kv, D, blk_size)
            pool_i.create_sequence(0, S + 256)
            k_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
            v_i = torch.randn(1, S, H_kv, D, dtype=torch.bfloat16)
            q_i = torch.randn(1, S, H_q, D, dtype=torch.bfloat16)
            pool_i.cache_update([0], k_i, v_i, [S])
            out_i = pool_i.attention([0], q_i, [S], [], scale, 0, torch.tensor([]))
            ref_outputs.append((q_i, k_i, v_i, out_i))

        # Batched with explicit q_start_offsets
        Q_padded = torch.zeros(BS, max_len, H_q, D, dtype=torch.bfloat16)
        K_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
        V_padded = torch.zeros(BS, max_len, H_kv, D, dtype=torch.bfloat16)
        for i in range(BS):
            S = seq_lens[i]
            q_i, k_i, v_i, _ = ref_outputs[i]
            Q_padded[i, max_len - S :] = q_i[0]
            K_padded[i, max_len - S :] = k_i[0]
            V_padded[i, max_len - S :] = v_i[0]

        bps = (max_len + 256 + blk_size - 1) // blk_size
        pool = _create_pool(bps * BS + 64, H_kv, D, blk_size)
        pool_seq_ids = list(range(BS))
        for sid in pool_seq_ids:
            pool.create_sequence(sid, max_len + 256)
        pool.cache_update(pool_seq_ids, K_padded, V_padded, seq_lens)

        # Pass non-empty q_start_offsets explicitly
        q_offsets = [i * max_len + (max_len - seq_lens[i]) for i in range(BS)]
        batched_out = pool.attention(
            pool_seq_ids,
            Q_padded.reshape(BS * max_len, H_q, D),
            seq_lens,
            q_offsets,
            scale,
            0,
            torch.tensor([]),
        )
        batched_out = batched_out.reshape(BS, max_len, H_q, D)

        for i in range(BS):
            S = seq_lens[i]
            _, _, _, ref_out = ref_outputs[i]
            slab_seq = batched_out[i, max_len - S :]
            self.assertEqual(
                slab_seq,
                ref_out[0],
                atol=BF16_ATOL,
                rtol=BF16_RTOL,
                msg=f"seq {i} (len={S}) with explicit q_start_offsets",
            )
