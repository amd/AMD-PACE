# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m pytest tests/ops/test_paged_attention.py -v

import torch
import torch.nn.functional as F
import unittest

import pace._C  # noqa: F401
from pace.llm.attention.paged.ops import (
    paged_attention_reshape_and_cache,
    paged_attention_with_kv_cache,
    get_paged_attention_scheduler_metadata,
    get_optimal_attention_isa,
)


def _reference_attention(query, key, value, scale, causal=True):
    """Reference SDPA implementation for correctness checking."""
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if causal:
        seq_len_q = scores.shape[-2]
        seq_len_k = scores.shape[-1]
        mask = torch.triu(
            torch.full((seq_len_q, seq_len_k), float("-inf"), device=scores.device),
            diagonal=seq_len_k - seq_len_q + 1,
        )
        scores = scores + mask
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value.float())


class TestGetOptimalAttentionISA(unittest.TestCase):
    def test_block_size_16_returns_vec16(self):
        isa = get_optimal_attention_isa(torch.bfloat16, 16)
        self.assertEqual(isa, "vec16")

    def test_block_size_32_returns_vec(self):
        isa = get_optimal_attention_isa(torch.bfloat16, 32)
        self.assertEqual(isa, "vec")

    def test_head_dim_80_returns_vec16(self):
        isa = get_optimal_attention_isa(torch.bfloat16, 32, head_dim=80)
        self.assertEqual(isa, "vec16")

    def test_head_dim_128_block_32_returns_vec(self):
        isa = get_optimal_attention_isa(torch.bfloat16, 32, head_dim=128)
        self.assertEqual(isa, "vec")

    def test_float32_block_16(self):
        isa = get_optimal_attention_isa(torch.float32, 16)
        self.assertEqual(isa, "vec16")


class TestGetSchedulerMetadata(unittest.TestCase):
    def test_single_request_decode(self):
        seq_lens = torch.tensor([10], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        meta = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        self.assertIsInstance(meta, torch.Tensor)
        self.assertGreater(meta.numel(), 0)

    def test_batched_requests(self):
        seq_lens = torch.tensor([10, 20, 5], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        meta = get_paged_attention_scheduler_metadata(
            num_reqs=3,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        self.assertIsInstance(meta, torch.Tensor)

    def test_prefill_metadata(self):
        seq_lens = torch.tensor([64], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 64], dtype=torch.int32)
        meta = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        self.assertIsInstance(meta, torch.Tensor)


class TestReshapeAndCache(unittest.TestCase):
    def _setup(
        self,
        num_tokens=4,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        num_blocks=4,
        dtype=torch.bfloat16,
    ):
        key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
        value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        return key, value, key_cache, value_cache

    def test_basic_cache_update(self):
        """Verify that values written to cache can be read back correctly."""
        key, value, key_cache, value_cache = self._setup(num_tokens=1)
        slot_mapping = torch.tensor([0], dtype=torch.int64)

        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )

        # Value should be stored row-major -- direct comparison
        stored_value = value_cache[0, :, 0, :]  # block 0, offset 0
        torch.testing.assert_close(stored_value, value[0], rtol=0, atol=0)

    def test_multiple_tokens_different_blocks(self):
        """Tokens mapped to different blocks should be stored correctly."""
        block_size = 16
        key, value, key_cache, value_cache = self._setup(
            num_tokens=3, block_size=block_size, num_blocks=4
        )
        # Token 0 -> block 0, offset 5
        # Token 1 -> block 2, offset 3
        # Token 2 -> block 1, offset 0
        slot_mapping = torch.tensor(
            [
                0 * block_size + 5,
                2 * block_size + 3,
                1 * block_size + 0,
            ],
            dtype=torch.int64,
        )

        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )

        # Check values are at correct positions
        torch.testing.assert_close(value_cache[0, :, 5, :], value[0], rtol=0, atol=0)
        torch.testing.assert_close(value_cache[2, :, 3, :], value[1], rtol=0, atol=0)
        torch.testing.assert_close(value_cache[1, :, 0, :], value[2], rtol=0, atol=0)

    def test_negative_slot_skipped(self):
        """Tokens with slot_mapping=-1 should be skipped."""
        key, value, key_cache, value_cache = self._setup(num_tokens=2)
        slot_mapping = torch.tensor([0, -1], dtype=torch.int64)

        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )

        # Token 0 should be cached, but all other positions should remain zero
        torch.testing.assert_close(value_cache[0, :, 0, :], value[0], rtol=0, atol=0)
        # Block 0, offsets 1+ should still be zero (only offset 0 was written)
        self.assertTrue(torch.all(value_cache[0, :, 1:, :] == 0))

    def test_float32_dtype(self):
        """Should work with float32 as well."""
        key, value, key_cache, value_cache = self._setup(
            num_tokens=2, dtype=torch.float32
        )
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )
        torch.testing.assert_close(value_cache[0, :, 0, :], value[0], rtol=0, atol=0)
        torch.testing.assert_close(value_cache[0, :, 1, :], value[1], rtol=0, atol=0)


class TestPagedAttentionWithKVCache(unittest.TestCase):
    def _run_single_request_decode(
        self,
        num_heads=8,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        seq_len=32,
        dtype=torch.bfloat16,
    ):
        """Test decode (query_len=1) against reference SDPA."""
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        scale = head_dim**-0.5
        num_tokens = seq_len

        # Create and cache all K/V for the sequence
        all_keys = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
        all_values = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )

        # Use sequential block allocation
        block_table_list = list(range(num_blocks))
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            all_keys, all_values, key_cache, value_cache, slot_mapping, "vec16"
        )

        # Now run decode for the last token
        query = torch.randn(1, num_heads, head_dim, dtype=dtype)
        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_table = torch.tensor(
            [block_table_list[:num_blocks_needed]], dtype=torch.int32
        )

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )

        paged_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        # Reference: query [1, num_heads, head_dim], all_keys [seq_len, num_kv_heads, head_dim]
        groups = num_heads // num_kv_heads
        ref_k = (
            all_keys.float().permute(1, 0, 2).unsqueeze(0)
        )  # [1, num_kv_heads, seq_len, head_dim]
        ref_v = all_values.float().permute(1, 0, 2).unsqueeze(0)
        if groups > 1:
            ref_k = ref_k.repeat_interleave(groups, dim=1)
            ref_v = ref_v.repeat_interleave(groups, dim=1)
        # query [1, num_heads, head_dim] -> permute to [1, num_heads, 1, head_dim]
        # For decode (q_len=1 at the last position), the query attends to all KV tokens
        ref_q = (
            query.float().permute(1, 0, 2).unsqueeze(0)
        )  # [1, num_heads, 1, head_dim]
        ref_output = F.scaled_dot_product_attention(
            ref_q, ref_k, ref_v, is_causal=False
        )
        ref_output = ref_output[0, :, 0, :].to(dtype)  # [num_heads, head_dim]

        return output[0], ref_output

    def test_decode_mha_bf16(self):
        """Decode with MHA (num_heads == num_kv_heads), bfloat16."""
        actual, expected = self._run_single_request_decode(
            num_heads=8,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            seq_len=32,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

    def test_decode_gqa_bf16(self):
        """Decode with GQA (num_heads=32, num_kv_heads=8), bfloat16."""
        actual, expected = self._run_single_request_decode(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            seq_len=48,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

    def test_decode_float32(self):
        """Decode with float32 for higher precision check."""
        actual, expected = self._run_single_request_decode(
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            block_size=16,
            seq_len=16,
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_decode_block_size_32(self):
        """Decode with block_size=32 (uses 'vec' ISA)."""
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 32
        seq_len = 64
        dtype = torch.bfloat16
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        scale = head_dim**-0.5

        all_keys = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        all_values = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        slot_mapping = torch.arange(seq_len, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            all_keys, all_values, key_cache, value_cache, slot_mapping, "vec"
        )

        query = torch.randn(1, num_heads, head_dim, dtype=dtype)
        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec",
            enable_kv_split=True,
        )
        paged_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        ref_k = all_keys.float().permute(1, 0, 2).unsqueeze(0)
        ref_v = all_values.float().permute(1, 0, 2).unsqueeze(0)
        ref_q = query.float().permute(1, 0, 2).unsqueeze(0)
        ref_output = F.scaled_dot_product_attention(
            ref_q, ref_k, ref_v, is_causal=False
        )
        ref_output = ref_output[0, :, 0, :].to(dtype)

        torch.testing.assert_close(output[0], ref_output, rtol=0.05, atol=0.05)

    def test_prefill_single_request(self):
        """Prefill (query_len == seq_len) for a single request."""
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 16
        seq_len = 32
        dtype = torch.bfloat16
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        scale = head_dim**-0.5

        query = torch.randn(seq_len, num_heads, head_dim, dtype=dtype)
        key = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        value = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        slot_mapping = torch.arange(seq_len, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )

        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32)
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        paged_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        # Reference
        ref_q = (
            query.float().permute(1, 0, 2).unsqueeze(0)
        )  # [1, num_heads, seq_len, head_dim]
        ref_k = key.float().permute(1, 0, 2).unsqueeze(0)
        ref_v = value.float().permute(1, 0, 2).unsqueeze(0)
        ref_output = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, is_causal=True)
        ref_output = (
            ref_output[0].permute(1, 0, 2).to(dtype)
        )  # [seq_len, num_heads, head_dim]

        torch.testing.assert_close(output, ref_output, rtol=0.05, atol=0.05)

    def test_batched_decode(self):
        """Batched decode with multiple requests."""
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 16
        dtype = torch.bfloat16
        scale = head_dim**-0.5
        seq_lens = [20, 35, 10]
        num_reqs = len(seq_lens)
        max_seq_len = max(seq_lens)
        total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens) + 4
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

        key_cache = torch.zeros(
            total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )

        # Pre-fill cache for each request with sequential block allocation
        all_keys_list = []
        all_values_list = []
        block_offset = 0
        block_table = torch.zeros(num_reqs, max_blocks_per_seq, dtype=torch.int32)
        slot_mapping_list = []

        for req_idx, sl in enumerate(seq_lens):
            keys = torch.randn(sl, num_kv_heads, head_dim, dtype=dtype)
            values = torch.randn(sl, num_kv_heads, head_dim, dtype=dtype)
            all_keys_list.append(keys)
            all_values_list.append(values)

            req_blocks = (sl + block_size - 1) // block_size
            for b in range(req_blocks):
                block_table[req_idx, b] = block_offset + b

            slots = torch.arange(sl, dtype=torch.int64) + block_offset * block_size
            slot_mapping_list.append(slots)

            paged_attention_reshape_and_cache(
                keys, values, key_cache, value_cache, slots, "vec16"
            )
            block_offset += req_blocks

        # Decode query (1 token per request)
        query = torch.randn(num_reqs, num_heads, head_dim, dtype=dtype)
        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=num_reqs,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        paged_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        # Verify each request independently
        for req_idx, sl in enumerate(seq_lens):
            ref_k = all_keys_list[req_idx].float().permute(1, 0, 2).unsqueeze(0)
            ref_v = all_values_list[req_idx].float().permute(1, 0, 2).unsqueeze(0)
            ref_q = query[req_idx : req_idx + 1].float().permute(1, 0, 2).unsqueeze(0)
            ref_out = F.scaled_dot_product_attention(
                ref_q, ref_k, ref_v, is_causal=False
            )
            ref_out = ref_out[0, :, 0, :].to(dtype)
            torch.testing.assert_close(
                output[req_idx],
                ref_out,
                rtol=0.05,
                atol=0.05,
                msg=f"Mismatch for request {req_idx} (seq_len={sl})",
            )

    def test_non_contiguous_block_table(self):
        """Block table sliced from a larger tensor (non-contiguous) should work."""
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 16
        seq_len = 16
        dtype = torch.bfloat16
        scale = head_dim**-0.5

        key = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        value = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(4, num_kv_heads, block_size, head_dim, dtype=dtype)
        value_cache = torch.zeros(4, num_kv_heads, block_size, head_dim, dtype=dtype)
        slot_mapping = torch.arange(seq_len, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, "vec16"
        )

        # Create non-contiguous block_table by slicing columns from a wider tensor
        big_block_table = torch.zeros(4, 10, dtype=torch.int32)
        big_block_table[0, 0] = 0
        big_block_table[1, 0] = 1
        block_table = big_block_table[
            :2, :1
        ]  # 2 rows, 1 col from 10-wide -> non-contiguous
        self.assertFalse(block_table.is_contiguous())
        block_table = block_table[:1]  # take first row only

        query = torch.randn(1, num_heads, head_dim, dtype=dtype)
        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        paged_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )
        self.assertFalse(torch.all(output == 0), "Output should not be all zeros")


class TestSlidingWindowAttention(unittest.TestCase):
    """Tests for paged attention with sliding window enabled."""

    def test_sliding_window_decode(self):
        """Decode with sliding_window > 0 produces different output than full attention
        when sequence length exceeds the window."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 16
        seq_len = 64
        sliding_window = 16
        dtype = torch.bfloat16
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        scale = head_dim**-0.5

        all_keys = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        all_values = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        slot_mapping = torch.arange(seq_len, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            all_keys, all_values, key_cache, value_cache, slot_mapping, "vec16"
        )

        query = torch.randn(1, num_heads, head_dim, dtype=dtype)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )

        # Run with full attention (no sliding window)
        output_full = torch.zeros_like(query)
        paged_attention_with_kv_cache(
            query=query.clone(),
            key_cache=key_cache,
            value_cache=value_cache,
            output=output_full,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        # Run with sliding window
        sw_left = sliding_window - 1
        sw_right = 0
        output_sw = torch.zeros_like(query)
        paged_attention_with_kv_cache(
            query=query.clone(),
            key_cache=key_cache,
            value_cache=value_cache,
            output=output_sw,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(sw_left, sw_right),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        self.assertFalse(
            torch.all(output_sw == 0), "Sliding window output should not be all zeros"
        )
        self.assertFalse(
            torch.allclose(output_full, output_sw, atol=1e-3),
            "Sliding window output should differ from full attention "
            "when seq_len > window size",
        )

        # Reference: attend only to the last `sliding_window` tokens
        windowed_keys = all_keys[-sliding_window:]
        windowed_values = all_values[-sliding_window:]
        ref_q = query.float().permute(1, 0, 2).unsqueeze(0)
        ref_k = windowed_keys.float().permute(1, 0, 2).unsqueeze(0)
        ref_v = windowed_values.float().permute(1, 0, 2).unsqueeze(0)
        ref_output = F.scaled_dot_product_attention(
            ref_q, ref_k, ref_v, is_causal=False
        )
        ref_output = ref_output[0, :, 0, :].to(dtype)
        torch.testing.assert_close(
            output_sw[0],
            ref_output,
            rtol=0.05,
            atol=0.05,
            msg="Sliding window kernel output should match windowed reference",
        )


class TestSinksPassthrough(unittest.TestCase):
    """Tests for paged attention with attention sinks (s_aux) enabled."""

    def test_sinks_bf16_no_crash(self):
        """Verify kernel accepts a non-None bf16 sinks tensor without crashing."""
        num_heads, num_kv_heads, head_dim = 8, 8, 128
        block_size = 16
        seq_len = 32
        dtype = torch.bfloat16
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        scale = head_dim**-0.5

        all_keys = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        all_values = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype
        )
        slot_mapping = torch.arange(seq_len, dtype=torch.int64)

        paged_attention_reshape_and_cache(
            all_keys, all_values, key_cache, value_cache, slot_mapping, "vec16"
        )

        query = torch.randn(1, num_heads, head_dim, dtype=dtype)
        output = torch.zeros_like(query)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)
        # Pad to multiple of 16 — kernel vectorizes in chunks of 16.
        # Production code (backend.py) handles this automatically.
        padded_num_heads = ((num_heads + 15) // 16) * 16
        sinks = torch.zeros(padded_num_heads, dtype=torch.bfloat16)
        sinks[:num_heads] = torch.randn(num_heads, dtype=torch.bfloat16)

        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_lens=seq_lens_t,
            dtype=dtype,
            query_start_loc=query_start_loc,
            causal=True,
            sliding_window_size=-1,
            isa="vec16",
            enable_kv_split=True,
        )
        # Run without sinks (baseline)
        output_no_sinks = torch.zeros_like(query)
        paged_attention_with_kv_cache(
            query=query.clone(),
            key_cache=key_cache,
            value_cache=value_cache,
            output=output_no_sinks,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=None,
        )

        # Run with sinks
        paged_attention_with_kv_cache(
            query=query.clone(),
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=scheduler_metadata,
            s_aux=sinks,
        )
        self.assertFalse(
            torch.all(output == 0), "Output with sinks should not be all zeros"
        )
        self.assertFalse(
            torch.isnan(output).any(), "Output with sinks should not contain NaN"
        )
        self.assertFalse(
            torch.allclose(output, output_no_sinks, atol=1e-3),
            "Sinks should produce different output than no-sinks baseline",
        )


if __name__ == "__main__":
    unittest.main()
