# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for SLAB Python cache layer: SlabPoolManager, SlabPoolContext,
SlabPoolLayerView. Tests duck-type compatibility with KVCacheManager.
"""

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401

from pace.llm.attention.slab.cache import (
    SlabPoolManager,
    SlabPoolContext,
    SlabPoolLayerView,
)


class _MockConfig:
    """Minimal config for SlabPoolManager tests.

    Default: 4 q_heads, 4 kv_heads, hidden_size=256 → head_dim=64.
    """

    def __init__(
        self,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=256,
        max_position_embeddings=512,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings


class TestSlabPoolManager(TestCase):
    def test_creation_with_max_total_tokens(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        self.assertEqual(mgr.num_layers, 2)
        self.assertEqual(mgr.num_kv_heads, 4)
        self.assertEqual(mgr.head_dim, 64)

    def test_creation_with_memory_budget(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, kv_cache_memory_gb=0.01)
        self.assertGreater(mgr.max_total_tokens, 0)

    def test_create_and_remove_sequence(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="seq-0")
        self.assertEqual(token, "seq-0")
        self.assertEqual(mgr.get_sequence_length(token), 0)
        mgr.remove_sequence(token)

    def test_create_sequence_auto_token(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence()
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 0)

    def test_duplicate_token_raises(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        mgr.create_sequence(token="dup")
        with self.assertRaisesRegex(ValueError, "already exists"):
            mgr.create_sequence(token="dup")

    def test_get_active_tokens(self):
        config = _MockConfig()
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        mgr.create_sequence(token="a")
        mgr.create_sequence(token="b")
        self.assertEqual(set(mgr.get_active_tokens()), {"a", "b"})

    def test_truncate_sequence(self):
        config = _MockConfig(num_hidden_layers=1)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="t")

        k = torch.randn(1, 10, 4, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 10, 4, 64, dtype=torch.bfloat16)
        mgr.update_cache_batched(0, [token], k, v)
        self.assertEqual(mgr.get_sequence_length(token), 10)

        mgr.truncate_sequence(token, 3)
        self.assertEqual(mgr.get_sequence_length(token), 7)

    def test_compute_max_tokens_from_memory(self):
        tokens = SlabPoolManager.compute_max_tokens_from_memory(
            kv_cache_memory_gb=1.0,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        self.assertGreater(tokens, 0)
        expected = int(1.0 * 1024**3) // (32 * 2 * 8 * 128 * 2)
        self.assertEqual(tokens, expected)


class TestSlabPoolContext(TestCase):
    def test_duck_type_len(self):
        config = _MockConfig(num_hidden_layers=1)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="ctx")
        ctx = SlabPoolContext(mgr, [token])
        self.assertEqual(len(ctx), 0)

    def test_duck_type_getitem(self):
        config = _MockConfig(num_hidden_layers=2)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="ctx")
        ctx = SlabPoolContext(mgr, [token])
        layer_view = ctx[0]
        self.assertIsInstance(layer_view, SlabPoolLayerView)

    def test_duck_type_remove_cache(self):
        config = _MockConfig(num_hidden_layers=1)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="ctx")

        k = torch.randn(1, 10, 4, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 10, 4, 64, dtype=torch.bfloat16)
        mgr.update_cache_batched(0, [token], k, v)

        ctx = SlabPoolContext(mgr, [token])
        self.assertEqual(len(ctx), 10)
        ctx.remove_cache(3)
        self.assertEqual(len(ctx), 7)


class TestSlabPoolLayerView(TestCase):
    def test_update_and_seq_len(self):
        config = _MockConfig(num_hidden_layers=1)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="lv")
        view = SlabPoolLayerView(mgr, [token], layer_idx=0)

        k = torch.randn(1, 5, 4, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 5, 4, 64, dtype=torch.bfloat16)
        view.update(k, v)
        self.assertEqual(view.seq_len, 5)

    def test_attend(self):
        config = _MockConfig(num_hidden_layers=1)
        mgr = SlabPoolManager(config, max_total_tokens=1024)
        token = mgr.create_sequence(token="lv")
        view = SlabPoolLayerView(mgr, [token], layer_idx=0)

        k = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16)
        view.update(k, v)

        k_new = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16)
        v_new = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16)
        view.update(k_new, v_new)

        q = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16)
        out = view.attend(q, scale=1.0 / 8.0)
        self.assertEqual(out.shape, (1, 1, 4, 64))
        self.assertFalse(torch.isnan(out).any())


class TestSlabCacheBackend(TestCase):
    """Test SlabCache create_context, remove_context, and _ensure_manager."""

    def test_create_and_remove_context(self):
        """create_context returns SlabPoolContext, remove_context cleans up."""
        from pace.llm.attention.slab.cache import SlabCache

        config = _MockConfig()
        cache = SlabCache(config)
        ctx = cache.create_context(config, max_seq_length=64, batch_size=2)
        self.assertIsNotNone(ctx)
        self.assertEqual(len(ctx.tokens), 2)
        cache.remove_context(ctx)
        self.assertEqual(len(ctx.tokens), 0)

    def test_ensure_manager_resizes(self):
        """_ensure_manager creates a larger pool when needed."""
        from pace.llm.attention.slab.cache import SlabCache

        config = _MockConfig()
        cache = SlabCache(config)

        # First context: small
        ctx1 = cache.create_context(config, max_seq_length=32, batch_size=1)
        old_max = cache._manager.max_total_tokens

        cache.remove_context(ctx1)

        # Second context: larger — should trigger resize
        ctx2 = cache.create_context(config, max_seq_length=256, batch_size=4)
        new_max = cache._manager.max_total_tokens
        self.assertGreater(new_max, old_max)
        cache.remove_context(ctx2)

    def test_merge_contexts(self):
        """merge_contexts combines multiple single-sequence contexts."""
        from pace.llm.attention.slab.cache import SlabCache

        config = _MockConfig()
        cache = SlabCache(config)
        ctx1 = cache.create_context(config, max_seq_length=64, batch_size=1)
        ctx2 = cache.create_context(config, max_seq_length=64, batch_size=1)

        merged = cache.merge_contexts([ctx1, ctx2])
        self.assertEqual(len(merged.tokens), 2)

        cache.remove_context(merged)

    def test_context_with_kv_cache_memory_gb(self):
        """Server path: create cache with memory budget."""
        from pace.llm.attention.slab.cache import SlabCache

        config = _MockConfig()
        cache = SlabCache(config, kv_cache_memory_gb=0.01)
        self.assertIsNotNone(cache._manager)
        self.assertGreater(cache._manager.max_total_tokens, 0)

        ctx = cache.create_context(config, max_seq_length=64)
        self.assertIsNotNone(ctx)
        cache.remove_context(ctx)
