# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m unittest -v llm_infra/test_cache.py

from torch.testing._internal.common_utils import TestCase
import torch
from transformers import PretrainedConfig
from pace.llm.attention import KVCacheType, KVCacheManager
from pace.llm.attention.contiguous.cache import BMCKVCache, DynamicKVCache


class MockConfig(PretrainedConfig):
    """Mock configuration for testing purposes."""

    def __init__(self, num_hidden_layers=2):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers


class TestKVCacheManager(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.kv_cache_manager = KVCacheManager(
            self.config, self.max_seq_length, KVCacheType.DYNAMIC
        )

    def test_initialization(self):
        self.assertEqual(len(self.kv_cache_manager.cache_objects), 2)
        self.assertIsInstance(self.kv_cache_manager.cache_objects[0], DynamicKVCache)

    def test_getitem(self):
        cache_layer = self.kv_cache_manager[0]
        self.assertIsInstance(cache_layer, DynamicKVCache)


class TestBMCKVCache(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.bmc_cache = BMCKVCache(self.max_seq_length)
        self.key_states = torch.rand(2, 10, 64)
        self.value_states = torch.rand(2, 10, 64)

    def test_update_cache(self):
        key, value = self.bmc_cache.update_cache(
            self.key_states, self.value_states, concat_dim=1
        )
        self.assertEqual(key.shape[1], 11)  # Full BMC buffer (segment-aligned)
        self.assertEqual(value.shape[1], 11)


class TestDynamicKVCache(TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=2)
        self.max_seq_length = 128
        self.Dynamic_cache = DynamicKVCache(self.max_seq_length)
        self.key_states = torch.rand(2, 10, 64)
        self.value_states = torch.rand(2, 10, 64)

    def test_update_cache(self):
        key, value = self.Dynamic_cache.update_cache(
            self.key_states, self.value_states, concat_dim=1
        )
        self.assertEqual(key.shape[1], 10)  # Should match the input sequence length
        self.assertEqual(value.shape[1], 10)


# ---------------------------------------------------------------------------
#  Accuracy tests: the optimized BMCKVCache (production code) is verified
#  against a self-contained reference implementation that uses no
#  optimizations (torch.zeros, >= expansion).
# ---------------------------------------------------------------------------


def _ref_update_cache(
    key, value, seq_len, concat_dim, tokens_per_split, key_states, value_states
):
    """Reference (unoptimized) update_cache: torch.zeros + >= expansion."""
    token_count = key_states.size(concat_dim)
    updated_seq_len = seq_len + token_count
    need_segment = key is None or updated_seq_len >= key.size(concat_dim)
    if need_segment:
        segment_idx = (updated_seq_len - 1) // tokens_per_split
        new_shape = list(key_states.shape)
        new_shape[concat_dim] = (segment_idx + 1) * tokens_per_split
        new_key = torch.zeros(new_shape, dtype=key_states.dtype)
        new_value = torch.zeros(new_shape, dtype=value_states.dtype)
        if key is not None and value is not None:
            new_key.narrow(concat_dim, 0, key.size(concat_dim)).copy_(key)
            new_value.narrow(concat_dim, 0, value.size(concat_dim)).copy_(value)
        key = new_key
        value = new_value
    key.narrow(concat_dim, seq_len, token_count).copy_(key_states)
    value.narrow(concat_dim, seq_len, token_count).copy_(value_states)
    return key, value, updated_seq_len


def _run_prefill_decode_reference(
    max_seq_length, key_list, value_list, concat_dim, tokens_per_split=None
):
    """Run prefill + decode using the reference (unoptimized) logic."""
    if tokens_per_split is None:
        tokens_per_split = BMCKVCache(max_seq_length).tokens_per_split
    key, value, seq_len = None, None, 0
    expansions = 0
    for k, v in zip(key_list, value_list):
        old_ptr = key.data_ptr() if key is not None else None
        key, value, seq_len = _ref_update_cache(
            key, value, seq_len, concat_dim, tokens_per_split, k, v
        )
        if key.data_ptr() != old_ptr:
            expansions += 1
    key_out = key.narrow(concat_dim, 0, seq_len).clone()
    val_out = value.narrow(concat_dim, 0, seq_len).clone()
    return key_out, val_out, seq_len, expansions


def _run_prefill_decode_optimized(max_seq_length, key_list, value_list, concat_dim):
    """Run prefill + decode using the production BMCKVCache."""
    cache = BMCKVCache(max_seq_length)
    expansions = 0
    for k, v in zip(key_list, value_list):
        old_ptr = cache.key.data_ptr() if cache.key is not None else None
        cache.update_cache(k, v, concat_dim)
        if cache.key.data_ptr() != old_ptr:
            expansions += 1
    key_out = cache.key.narrow(concat_dim, 0, cache.seq_len).clone()
    val_out = cache.value.narrow(concat_dim, 0, cache.seq_len).clone()
    return key_out, val_out, cache.seq_len, expansions


class TestBMCAllocModeAccuracy(TestCase):
    """Verify that the optimized BMCKVCache is bit-identical to the
    reference implementation with no optimizations."""

    def _assert_matches_reference(self, max_seq, key_list, value_list, concat_dim=1):
        """Run reference and optimized paths; assert KV equality."""
        key_ref, val_ref, sl_ref, exp_ref = _run_prefill_decode_reference(
            max_seq, key_list, value_list, concat_dim
        )
        key_opt, val_opt, sl_opt, exp_opt = _run_prefill_decode_optimized(
            max_seq, key_list, value_list, concat_dim
        )
        self.assertEqual(sl_ref, sl_opt, "seq_len mismatch vs reference")
        self.assertTrue(
            torch.equal(key_ref, key_opt),
            f"Key mismatch vs reference (ref expansions={exp_ref}, optimized expansions={exp_opt})",
        )
        self.assertTrue(
            torch.equal(val_ref, val_opt),
            f"Value mismatch vs reference (ref expansions={exp_ref}, optimized expansions={exp_opt})",
        )
        self.assertLessEqual(
            exp_opt, exp_ref, "Optimized path should not expand more than reference"
        )
        return exp_ref, exp_opt

    # ── Test cases ─────────────────────────────────────────────────────────

    def test_prefill_only(self):
        """Single prefill insert; no decode steps."""
        B, H = 4, 32
        max_seq = 128
        torch.manual_seed(42)
        k = torch.randn(B, 10, H)
        v = torch.randn(B, 10, H)
        self._assert_matches_reference(max_seq, [k], [v], concat_dim=1)

    def test_prefill_plus_short_decode(self):
        """Prefill + a few decode tokens (no segment boundary crossed)."""
        B, H = 4, 32
        max_seq = 128
        concat_dim = 1
        torch.manual_seed(42)
        keys = [torch.randn(B, 5, H)]
        vals = [torch.randn(B, 5, H)]
        for _ in range(3):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        self._assert_matches_reference(max_seq, keys, vals, concat_dim)

    def test_cross_segment_boundary(self):
        """Decode past at least one segment boundary."""
        B, H = 2, 16
        max_seq = 64  # tokens_per_split = 8
        concat_dim = 1
        torch.manual_seed(123)
        keys = [torch.randn(B, 3, H)]
        vals = [torch.randn(B, 3, H)]
        for _ in range(20):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        exp_ref, exp_def = self._assert_matches_reference(
            max_seq, keys, vals, concat_dim
        )
        self.assertLess(
            exp_def,
            exp_ref,
            "Default path should have strictly fewer expansions past boundaries",
        )

    def test_exact_boundary_alignment(self):
        """Prefill lands exactly on a segment boundary."""
        B, H = 2, 16
        max_seq = 64  # tokens_per_split = 8
        concat_dim = 1
        torch.manual_seed(7)
        keys = [torch.randn(B, 8, H)]
        vals = [torch.randn(B, 8, H)]
        for _ in range(10):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        self._assert_matches_reference(max_seq, keys, vals, concat_dim)

    def test_single_token_prefill(self):
        """Prefill with just 1 token, then decode."""
        B, H = 2, 16
        max_seq = 64
        concat_dim = 1
        torch.manual_seed(99)
        keys = [torch.randn(B, 1, H)]
        vals = [torch.randn(B, 1, H)]
        for _ in range(15):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        self._assert_matches_reference(max_seq, keys, vals, concat_dim)

    def test_large_decode_sequence(self):
        """Longer decode (100 tokens) to cross many boundaries."""
        B, H = 2, 16
        max_seq = 256  # tokens_per_split = 16
        concat_dim = 1
        torch.manual_seed(0)
        keys = [torch.randn(B, 5, H)]
        vals = [torch.randn(B, 5, H)]
        for _ in range(100):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        exp_ref, exp_def = self._assert_matches_reference(
            max_seq, keys, vals, concat_dim
        )
        self.assertLess(exp_def, exp_ref)

    def test_bfloat16_dtype(self):
        """Verify accuracy holds for bf16 (the production dtype)."""
        B, H = 4, 32
        max_seq = 128
        concat_dim = 1
        torch.manual_seed(42)
        keys = [torch.randn(B, 5, H, dtype=torch.bfloat16)]
        vals = [torch.randn(B, 5, H, dtype=torch.bfloat16)]
        for _ in range(30):
            keys.append(torch.randn(B, 1, H, dtype=torch.bfloat16))
            vals.append(torch.randn(B, 1, H, dtype=torch.bfloat16))
        self._assert_matches_reference(max_seq, keys, vals, concat_dim)

    def test_float32_dtype(self):
        """Verify accuracy holds for float32."""
        B, H = 4, 32
        max_seq = 128
        concat_dim = 1
        torch.manual_seed(42)
        keys = [torch.randn(B, 5, H, dtype=torch.float32)]
        vals = [torch.randn(B, 5, H, dtype=torch.float32)]
        for _ in range(30):
            keys.append(torch.randn(B, 1, H, dtype=torch.float32))
            vals.append(torch.randn(B, 1, H, dtype=torch.float32))
        self._assert_matches_reference(max_seq, keys, vals, concat_dim)

    def test_4d_tensor_shape(self):
        """4-D shape (B, N, S, H) matching real attention layout."""
        B, N, H = 4, 12, 64
        max_seq = 256
        concat_dim = 2
        torch.manual_seed(42)
        keys = [torch.randn(B, N, 10, H)]
        vals = [torch.randn(B, N, 10, H)]
        for _ in range(50):
            keys.append(torch.randn(B, N, 1, H))
            vals.append(torch.randn(B, N, 1, H))
        exp_ref, exp_def = self._assert_matches_reference(
            max_seq, keys, vals, concat_dim
        )
        self.assertLess(exp_def, exp_ref)

    def test_unused_tail_is_zeros(self):
        """Verify that the unused tail of the allocated buffer is all zeros."""
        B, H = 2, 16
        max_seq = 64  # tokens_per_split = 8
        concat_dim = 1
        torch.manual_seed(55)
        cache = BMCKVCache(max_seq)
        k = torch.randn(B, 5, H)
        v = torch.randn(B, 5, H)
        cache.update_cache(k, v, concat_dim)
        alloc_len = cache.key.size(concat_dim)
        if alloc_len > cache.seq_len:
            tail_k = cache.key.narrow(
                concat_dim, cache.seq_len, alloc_len - cache.seq_len
            )
            tail_v = cache.value.narrow(
                concat_dim, cache.seq_len, alloc_len - cache.seq_len
            )
            self.assertTrue(
                torch.all(tail_k == 0),
                "Key tail is not all zeros after prefill",
            )
            self.assertTrue(
                torch.all(tail_v == 0),
                "Value tail is not all zeros after prefill",
            )

    def test_expansion_count_reduction(self):
        """Verify that default path halves the expansion count vs reference."""
        B, H = 2, 16
        max_seq = 256  # tokens_per_split = 16
        concat_dim = 1
        torch.manual_seed(0)
        keys = [torch.randn(B, 1, H)]
        vals = [torch.randn(B, 1, H)]
        for _ in range(160):
            keys.append(torch.randn(B, 1, H))
            vals.append(torch.randn(B, 1, H))
        exp_ref, exp_def = self._assert_matches_reference(
            max_seq, keys, vals, concat_dim
        )
        self.assertGreater(
            exp_ref,
            exp_def * 1.5,
            f"Expected reference ({exp_ref}) >> default ({exp_def})",
        )
