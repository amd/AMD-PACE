# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m pytest tests/llm_infra/test_paged_cache.py -v

import os
import types
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import TestCase

from pace.llm.attention.paged.cache import (
    PagedCache,
    PagedKVCache,
    PagedKVCachePool,
    SharedPagedKVCache,
)
from pace.llm.attention.recurrent_state import (
    MergedRecurrentState,
    RecurrentStateStore,
)
from pace.llm.attention.specs import LayerKVSpec


class TestPagedKVCacheInit(TestCase):
    def test_block_count_exact(self):
        cache = PagedKVCache(max_seq_length=32, block_size=16)
        self.assertEqual(cache.num_blocks, 2)

    def test_block_count_rounds_up(self):
        cache = PagedKVCache(max_seq_length=33, block_size=16)
        self.assertEqual(cache.num_blocks, 3)

    def test_block_count_single_token(self):
        cache = PagedKVCache(max_seq_length=1, block_size=16)
        self.assertEqual(cache.num_blocks, 1)

    def test_cache_tensor_shapes(self):
        cache = PagedKVCache(
            max_seq_length=64, num_kv_heads=8, head_dim=64, block_size=16
        )
        expected_shape = (4, 8, 16, 64)
        self.assertEqual(cache.key_cache.shape, expected_shape)
        self.assertEqual(cache.value_cache.shape, expected_shape)

    def test_initial_state(self):
        cache = PagedKVCache(max_seq_length=48, block_size=16)
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(len(cache.allocated_blocks), 0)
        self.assertEqual(len(cache.free_blocks), 3)
        self.assertIsNone(cache.block_table)


class TestPagedKVCacheBlockAllocation(TestCase):
    def setUp(self):
        self.cache = PagedKVCache(
            max_seq_length=64, num_kv_heads=4, head_dim=32, block_size=16
        )

    def test_allocate_block(self):
        block_idx = self.cache._allocate_block()
        self.assertIn(block_idx, self.cache.allocated_blocks)
        self.assertNotIn(block_idx, self.cache.free_blocks)
        self.assertEqual(len(self.cache.allocated_blocks), 1)
        self.assertEqual(len(self.cache.free_blocks), 3)

    def test_allocate_all_blocks(self):
        for _ in range(4):
            self.cache._allocate_block()
        self.assertEqual(len(self.cache.allocated_blocks), 4)
        self.assertEqual(len(self.cache.free_blocks), 0)

    def test_allocate_block_raises_when_exhausted(self):
        for _ in range(4):
            self.cache._allocate_block()
        with self.assertRaises(RuntimeError):
            self.cache._allocate_block()

    def test_free_block(self):
        block_idx = self.cache._allocate_block()
        self.cache._free_block(block_idx)
        self.assertNotIn(block_idx, self.cache.allocated_blocks)
        self.assertIn(block_idx, self.cache.free_blocks)

    def test_free_nonexistent_block_is_noop(self):
        self.cache._free_block(999)
        self.assertEqual(len(self.cache.free_blocks), 4)

    def test_ensure_blocks_allocated(self):
        self.cache._ensure_blocks_allocated(20)
        self.assertEqual(len(self.cache.allocated_blocks), 2)
        self.assertIsNotNone(self.cache.block_table)
        self.assertEqual(self.cache.block_table.shape[0], 2)

    def test_ensure_blocks_no_redundant_allocation(self):
        self.cache._ensure_blocks_allocated(16)
        blocks_after_first = list(self.cache.allocated_blocks)
        self.cache._ensure_blocks_allocated(10)
        self.assertEqual(self.cache.allocated_blocks, blocks_after_first)

    def test_ensure_blocks_incremental(self):
        self.cache._ensure_blocks_allocated(16)
        self.assertEqual(len(self.cache.allocated_blocks), 1)
        self.cache._ensure_blocks_allocated(32)
        self.assertEqual(len(self.cache.allocated_blocks), 2)
        self.cache._ensure_blocks_allocated(48)
        self.assertEqual(len(self.cache.allocated_blocks), 3)


class TestPagedKVCacheSlotMapping(TestCase):
    def setUp(self):
        from pace.llm.attention.paged.utils import compute_slot_mapping

        self.compute_slot_mapping = compute_slot_mapping
        self.cache = PagedKVCache(
            max_seq_length=64, num_kv_heads=4, head_dim=32, block_size=16
        )
        self.cache._ensure_blocks_allocated(64)

    def test_slot_mapping_first_block(self):
        slots = self.compute_slot_mapping(self.cache, 1, 16, 0)
        self.assertEqual(slots.shape[0], 16)
        block0 = self.cache.allocated_blocks[0]
        for i in range(16):
            self.assertEqual(slots[i].item(), block0 * 16 + i)

    def test_slot_mapping_cross_block(self):
        slots = self.compute_slot_mapping(self.cache, 1, 4, 14)
        self.assertEqual(slots.shape[0], 4)
        block0 = self.cache.allocated_blocks[0]
        block1 = self.cache.allocated_blocks[1]
        self.assertEqual(slots[0].item(), block0 * 16 + 14)
        self.assertEqual(slots[1].item(), block0 * 16 + 15)
        self.assertEqual(slots[2].item(), block1 * 16 + 0)
        self.assertEqual(slots[3].item(), block1 * 16 + 1)

    def test_slot_mapping_empty(self):
        slots = self.compute_slot_mapping(self.cache, 1, 0, 0)
        self.assertEqual(slots.shape[0], 0)

    def test_slot_mapping_single_token(self):
        slots = self.compute_slot_mapping(self.cache, 1, 1, 5)
        block0 = self.cache.allocated_blocks[0]
        self.assertEqual(slots[0].item(), block0 * 16 + 5)


class TestPagedKVCacheRemove(TestCase):
    def setUp(self):
        self.cache = PagedKVCache(
            max_seq_length=64, num_kv_heads=4, head_dim=32, block_size=16
        )
        self.cache._ensure_blocks_allocated(48)
        self.cache.seq_len = 48

    def test_remove_within_block(self):
        self.cache.remove_cache(5)
        self.assertEqual(self.cache.seq_len, 43)
        self.assertEqual(len(self.cache.allocated_blocks), 3)

    def test_remove_frees_blocks(self):
        self.cache.remove_cache(20)
        self.assertEqual(self.cache.seq_len, 28)
        self.assertEqual(len(self.cache.allocated_blocks), 2)

    def test_remove_all_tokens(self):
        self.cache.remove_cache(48)
        self.assertEqual(self.cache.seq_len, 0)
        self.assertEqual(len(self.cache.allocated_blocks), 0)
        self.assertIsNone(self.cache.block_table)

    def test_remove_too_many_raises(self):
        with self.assertRaises(ValueError):
            self.cache.remove_cache(100)


class TestPagedKVCacheRemoveInPlaceTruncate(TestCase):
    """remove_cache truncates block_table in place (O(1) view) instead of
    rebuilding the whole tensor, while staying bit-identical to the previous
    full-rebuild behaviour.

    This is the speculative-decode rollback path: every step appends k+1 tokens
    and discards the rejected suffix, so rollback cost must not scale with the
    context length already in the cache.
    """

    def setUp(self):
        # 160 tokens at block_size 16 -> exactly 10 blocks.
        self.cache = PagedKVCache(
            max_seq_length=256, num_kv_heads=4, head_dim=32, block_size=16
        )
        self.cache._ensure_blocks_allocated(160)
        self.cache.seq_len = 160

    def test_block_table_matches_allocated_blocks(self):
        self.cache.remove_cache(40)  # 160 -> 120 tokens, 10 -> 8 blocks
        self.assertEqual(self.cache.seq_len, 120)
        self.assertEqual(len(self.cache.allocated_blocks), 8)
        expected = torch.tensor(self.cache.allocated_blocks, dtype=torch.int32)
        self.assertTrue(torch.equal(self.cache.block_table, expected))

    def test_cross_block_remove_is_view_not_rebuild(self):
        orig = self.cache.block_table  # keep alive so its storage isn't reused
        orig_ptr = orig.data_ptr()
        self.cache.remove_cache(40)  # frees 2 tail blocks
        # Slicing shares storage with the original tensor (same data_ptr); a
        # torch.tensor() rebuild would allocate fresh storage with a different
        # pointer (original is kept alive via `orig`).
        self.assertEqual(self.cache.block_table.data_ptr(), orig_ptr)
        self.assertEqual(self.cache.block_table.size(0), 8)

    def test_within_block_remove_keeps_block_table_view(self):
        orig = self.cache.block_table
        orig_values = orig.clone()
        self.cache.remove_cache(5)  # 160 -> 155 tokens, still 10 blocks
        self.assertEqual(len(self.cache.allocated_blocks), 10)
        self.assertEqual(self.cache.block_table.size(0), 10)
        # No blocks freed, no rebuild: identical storage and values.
        self.assertEqual(self.cache.block_table.data_ptr(), orig.data_ptr())
        self.assertTrue(torch.equal(self.cache.block_table, orig_values))

    def test_freed_blocks_returned_to_pool(self):
        before_free = len(self.cache.free_blocks)
        freed_expected = list(self.cache.allocated_blocks[-2:])
        self.cache.remove_cache(40)  # 10 -> 8 blocks, frees 2
        self.assertEqual(len(self.cache.free_blocks), before_free + 2)
        for b in freed_expected:
            self.assertIn(b, self.cache.free_blocks)
        for b in freed_expected:
            self.assertNotIn(b, self.cache.allocated_blocks)

    def test_remove_to_zero_clears_block_table(self):
        self.cache.remove_cache(160)
        self.assertEqual(self.cache.seq_len, 0)
        self.assertIsNone(self.cache.block_table)
        self.assertEqual(len(self.cache.allocated_blocks), 0)

    def test_exact_block_boundary_remove(self):
        # Remove exactly 2 blocks worth (32 tokens): 160 -> 128 (8 blocks).
        self.cache.remove_cache(32)
        self.assertEqual(self.cache.seq_len, 128)
        self.assertEqual(len(self.cache.allocated_blocks), 8)
        self.assertEqual(self.cache.block_table.size(0), 8)
        self.assertTrue(
            torch.equal(
                self.cache.block_table,
                torch.tensor(self.cache.allocated_blocks, dtype=torch.int32),
            )
        )

    def test_equivalence_with_full_rebuild_random(self):
        """Across random start lengths and remove sizes, the in-place result is
        identical to the old full-rebuild semantics (seq_len, allocated_blocks,
        and block_table values)."""
        import random

        rng = random.Random(0)
        for _ in range(200):
            cache = PagedKVCache(
                max_seq_length=512, num_kv_heads=2, head_dim=16, block_size=16
            )
            start_len = rng.randint(1, 512)
            cache._ensure_blocks_allocated(start_len)
            cache.seq_len = start_len
            remove_len = rng.randint(0, start_len)

            # Reference mirrors the previous implementation exactly.
            ref_blocks = list(cache.allocated_blocks)
            ref_new_len = start_len - remove_len
            ref_needed = (ref_new_len + 16 - 1) // 16
            ref_blocks = ref_blocks[:ref_needed]
            ref_table = (
                torch.tensor(ref_blocks, dtype=torch.int32) if ref_new_len > 0 else None
            )

            cache.remove_cache(remove_len)

            self.assertEqual(cache.seq_len, ref_new_len)
            self.assertEqual(cache.allocated_blocks, ref_blocks)
            if ref_table is None:
                self.assertIsNone(cache.block_table)
            else:
                self.assertEqual(cache.block_table.size(0), ref_table.size(0))
                self.assertTrue(torch.equal(cache.block_table, ref_table))

    def test_speculative_rollback_loop_no_block_leak(self):
        """Simulate the PARD spec-decode loop: grow by draft_size+1, roll back
        the rejected suffix, repeat. block_table must stay consistent and no
        blocks may leak (allocated + free == total) across many cycles."""
        cache = PagedKVCache(
            max_seq_length=4096, num_kv_heads=2, head_dim=16, block_size=16
        )
        cache._ensure_blocks_allocated(100)  # prefill
        cache.seq_len = 100

        draft_size = 12
        for step in range(300):
            grown = cache.seq_len + draft_size + 1
            cache._ensure_blocks_allocated(grown)
            cache.seq_len = grown

            accepted = step % (draft_size + 1)  # 0..draft_size accepted
            trim = (draft_size + 1) - accepted
            if trim > 0:
                cache.remove_cache(trim)

            expected_blocks = (cache.seq_len + 15) // 16
            self.assertEqual(len(cache.allocated_blocks), expected_blocks)
            if cache.seq_len > 0:
                self.assertEqual(cache.block_table.size(0), expected_blocks)
                self.assertTrue(
                    torch.equal(
                        cache.block_table,
                        torch.tensor(cache.allocated_blocks, dtype=torch.int32),
                    )
                )

        # No block leak: every block is either allocated or free.
        self.assertEqual(
            len(cache.allocated_blocks) + len(cache.free_blocks),
            cache.num_blocks,
        )


class TestPagedKVCacheUpdateCache(TestCase):
    def test_update_batch_size_gt1_raises(self):
        cache = PagedKVCache(
            max_seq_length=64, num_kv_heads=4, head_dim=32, block_size=16
        )
        key = torch.randn(2, 4, 1, 32)
        value = torch.randn(2, 4, 1, 32)
        with self.assertRaises(Exception):
            cache.update_cache(key, value, concat_dim=2)


class TestPagedKVCachePool(TestCase):
    def setUp(self):
        PagedKVCachePool.reset()

    def tearDown(self):
        PagedKVCachePool.reset()

    def test_initialize_creates_pool(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        self.assertIsNotNone(pool)
        self.assertEqual(pool.total_blocks, 10)
        self.assertEqual(pool.num_layers, 2)

    def test_singleton_same_params(self):
        pool1 = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool2 = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        self.assertIs(pool1, pool2)

    def test_omitted_specs_do_not_reuse_a_per_layer_pool(self):
        """A uniform caller must not inherit a grouped/shared layout."""
        mixed = PagedKVCachePool.initialize(
            total_blocks=8,
            block_size=16,
            num_kv_heads=2,
            head_dim=64,
            num_layers=2,
            specs=[LayerKVSpec(2, 64), LayerKVSpec(2, 128)],
        )
        self.assertIsNone(mixed.key_cache)

        uniform = PagedKVCachePool.initialize(
            total_blocks=8,
            block_size=16,
            num_kv_heads=2,
            head_dim=64,
            num_layers=2,
        )
        self.assertIsNot(mixed, uniform)
        self.assertIsNotNone(uniform.key_cache)

    def test_singleton_different_params_recreates(self):
        pool1 = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool2 = PagedKVCachePool.initialize(
            total_blocks=20,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        self.assertIsNot(pool1, pool2)
        self.assertEqual(pool2.total_blocks, 20)

    def test_cache_tensor_shapes(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=8,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=3,
        )
        expected_shape = (3, 8, 4, 16, 64)
        self.assertEqual(pool.key_cache.shape, expected_shape)
        self.assertEqual(pool.value_cache.shape, expected_shape)

    def test_allocate_blocks(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        blocks = pool.allocate_blocks(request_id=1, num_blocks=3)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(len(pool.free_blocks), 7)

    def test_allocate_blocks_exhausted_raises(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=4,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool.allocate_blocks(request_id=1, num_blocks=4)
        with self.assertRaises(RuntimeError):
            pool.allocate_blocks(request_id=2, num_blocks=1)

    def test_free_blocks_for_request(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool.allocate_blocks(request_id=1, num_blocks=3)
        pool.allocate_blocks(request_id=2, num_blocks=2)
        pool.free_blocks_for_request(1)
        self.assertEqual(len(pool.free_blocks), 8)
        self.assertNotIn(1, pool.allocated_blocks)
        self.assertIn(2, pool.allocated_blocks)

    def test_free_nonexistent_request_is_safe(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool.free_blocks_for_request(999)

    def test_ensure_blocks_for_request(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        blocks = pool.ensure_blocks_for_request(request_id=1, required_blocks=3)
        self.assertEqual(len(blocks), 3)
        blocks = pool.ensure_blocks_for_request(request_id=1, required_blocks=3)
        self.assertEqual(len(blocks), 3)
        blocks = pool.ensure_blocks_for_request(request_id=1, required_blocks=5)
        self.assertEqual(len(blocks), 5)

    def test_get_stats(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool.allocate_blocks(request_id=1, num_blocks=3)
        pool.allocate_blocks(request_id=2, num_blocks=2)
        stats = pool.get_stats()
        self.assertEqual(stats["total_blocks"], 10)
        self.assertEqual(stats["free_blocks"], 5)
        self.assertEqual(stats["allocated_blocks"], 5)
        self.assertEqual(stats["active_requests"], 2)
        self.assertAlmostEqual(stats["utilization_pct"], 50.0)

    def test_reset_allocations(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        pool.allocate_blocks(request_id=1, num_blocks=5)
        pool.reset_allocations()
        self.assertEqual(len(pool.free_blocks), 10)
        self.assertEqual(len(pool.allocated_blocks), 0)

    def test_get_cache_tensors_per_layer(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=8,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=3,
        )
        for layer_idx in range(3):
            k, v = pool.get_cache_tensors(layer_idx)
            self.assertEqual(k.shape, (8, 4, 16, 64))
            self.assertEqual(v.shape, (8, 4, 16, 64))
            self.assertTrue(k.data_ptr() != v.data_ptr())

    def test_request_isolation(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )
        blocks1 = pool.allocate_blocks(request_id=1, num_blocks=3)
        blocks2 = pool.allocate_blocks(request_id=2, num_blocks=3)
        self.assertEqual(len(set(blocks1) & set(blocks2)), 0)


class TestPagedKVCachePoolPerLayerSpecs(TestCase):
    """Grouped storage and KV-sharing aliasing driven by per-layer specs."""

    SPECS = [
        LayerKVSpec(num_kv_heads=4, head_dim=64),
        LayerKVSpec(num_kv_heads=2, head_dim=128),
        LayerKVSpec(num_kv_heads=4, head_dim=64),
        LayerKVSpec(num_kv_heads=2, head_dim=128),
        LayerKVSpec(num_kv_heads=4, head_dim=64, kv_sharing_target=2),
        LayerKVSpec(num_kv_heads=2, head_dim=128, kv_sharing_target=3),
    ]

    def setUp(self):
        PagedKVCachePool.reset()

    def tearDown(self):
        PagedKVCachePool.reset()

    def _pool(self, specs=None, total_blocks=8, block_size=16):
        specs = self.SPECS if specs is None else specs
        return PagedKVCachePool.initialize(
            total_blocks=total_blocks,
            block_size=block_size,
            num_kv_heads=4,
            head_dim=64,
            num_layers=len(specs),
            dtype=torch.float32,
            specs=specs,
        )

    def test_mixed_specs_create_one_group_per_signature(self):
        pool = self._pool()
        self.assertEqual(len(pool._groups), 2)
        self.assertEqual(pool._groups[(4, 64)]["layers"], [0, 2])
        self.assertEqual(pool._groups[(2, 128)]["layers"], [1, 3])
        self.assertEqual(pool._groups[(4, 64)]["key"].shape, (2, 8, 4, 16, 64))
        self.assertEqual(pool._groups[(2, 128)]["key"].shape, (2, 8, 2, 16, 128))
        # Grouped layout has no single unified tensor to alias.
        self.assertIsNone(pool.key_cache)
        self.assertIsNone(pool.value_cache)

    def test_get_cache_tensors_uses_per_layer_dims(self):
        pool = self._pool()
        for layer_idx, spec in enumerate(self.SPECS):
            k, v = pool.get_cache_tensors(layer_idx)
            expected = (8, spec.num_kv_heads, 16, spec.head_dim)
            self.assertEqual(k.shape, expected)
            self.assertEqual(v.shape, expected)
            self.assertNotEqual(k.data_ptr(), v.data_ptr())

    def test_storage_layers_do_not_alias_each_other(self):
        pool = self._pool()
        ptrs = [pool.get_cache_tensors(i)[0].data_ptr() for i in range(4)]
        self.assertEqual(len(set(ptrs)), 4)

    def test_shared_layer_aliases_target_storage(self):
        pool = self._pool()
        for shared_idx, target_idx in ((4, 2), (5, 3)):
            k_shared, v_shared = pool.get_cache_tensors(shared_idx)
            k_target, v_target = pool.get_cache_tensors(target_idx)
            self.assertEqual(k_shared.data_ptr(), k_target.data_ptr())
            self.assertEqual(v_shared.data_ptr(), v_target.data_ptr())
            self.assertEqual(k_shared.shape, k_target.shape)

    def test_shared_layer_reads_target_writes(self):
        pool = self._pool()
        k_target, v_target = pool.get_cache_tensors(2)
        k_target[3, :, 5, :] = 1.25
        v_target[3, :, 5, :] = -2.5

        k_shared, v_shared = pool.get_cache_tensors(4)
        self.assertEqual(k_shared[3, :, 5, :].unique().tolist(), [1.25])
        self.assertEqual(v_shared[3, :, 5, :].unique().tolist(), [-2.5])
        self.assertEqual(pool.get_cache_tensors(1)[0].abs().sum().item(), 0.0)

    def test_shared_layers_allocate_no_extra_storage(self):
        shared = self._pool()
        storage_only = self._pool(specs=self.SPECS[:4])
        self.assertEqual(
            sum(g["key"].numel() for g in shared._groups.values()),
            sum(g["key"].numel() for g in storage_only._groups.values()),
        )

    def test_uniform_specs_preserve_legacy_unified_layout(self):
        specs = [LayerKVSpec(num_kv_heads=4, head_dim=64) for _ in range(3)]
        pool = self._pool(specs=specs)
        self.assertEqual(len(pool._groups), 1)
        self.assertEqual(pool.key_cache.shape, (3, 8, 4, 16, 64))
        self.assertEqual(pool.value_cache.shape, (3, 8, 4, 16, 64))
        for layer_idx in range(3):
            k, _ = pool.get_cache_tensors(layer_idx)
            self.assertEqual(k.data_ptr(), pool.key_cache[layer_idx].data_ptr())

    def test_specs_none_defaults_to_uniform_layout(self):
        pool = PagedKVCachePool.initialize(
            total_blocks=8,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=3,
        )
        self.assertEqual(len(pool.specs), 3)
        self.assertEqual(pool.key_cache.shape, (3, 8, 4, 16, 64))

    def test_spec_count_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "layer KV specs"):
            PagedKVCachePool(
                total_blocks=8,
                block_size=16,
                num_kv_heads=4,
                head_dim=64,
                num_layers=len(self.SPECS) + 1,
                specs=self.SPECS,
            )

    def test_shared_paged_kv_cache_reports_target_dims(self):
        pool = self._pool()
        shared = SharedPagedKVCache(
            pool=pool, request_id=0, layer_idx=5, max_seq_length=32
        )
        self.assertEqual(shared.num_kv_heads, 2)
        self.assertEqual(shared.head_dim, 128)
        self.assertTrue(shared.kv_sharing_readonly)

        owner = SharedPagedKVCache(
            pool=pool, request_id=0, layer_idx=3, max_seq_length=32
        )
        self.assertFalse(owner.kv_sharing_readonly)


class TestSharedPagedKVCache(TestCase):
    def setUp(self):
        PagedKVCachePool.reset()
        self.pool = PagedKVCachePool.initialize(
            total_blocks=20,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )

    def tearDown(self):
        PagedKVCachePool.reset()

    def test_init(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(cache.block_size, 16)
        self.assertEqual(cache.num_kv_heads, 4)
        self.assertEqual(cache.head_dim, 64)

    def test_ensure_blocks_delegates_to_pool(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        cache._ensure_blocks_allocated(48)
        pool_blocks = self.pool.get_blocks_for_request(1)
        self.assertEqual(len(pool_blocks), 3)

    def test_allocated_blocks_property(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        cache._ensure_blocks_allocated(32)
        self.assertEqual(len(cache.allocated_blocks), 2)
        self.assertEqual(cache.allocated_blocks, self.pool.get_blocks_for_request(1))

    def test_get_cache_tensors(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        k, v = cache.get_cache_tensors()
        pool_k, pool_v = self.pool.get_cache_tensors(0)
        self.assertEqual(k.data_ptr(), pool_k.data_ptr())
        self.assertEqual(v.data_ptr(), pool_v.data_ptr())

    def test_remove_cache(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        cache.seq_len = 30
        cache.remove_cache(10)
        self.assertEqual(cache.seq_len, 20)

    def test_remove_cache_too_many_raises(self):
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        cache.seq_len = 10
        with self.assertRaises(ValueError):
            cache.remove_cache(20)

    def test_multiple_requests_share_pool(self):
        cache1 = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        cache2 = SharedPagedKVCache(
            pool=self.pool, request_id=2, layer_idx=0, max_seq_length=128
        )
        cache1._ensure_blocks_allocated(32)
        cache2._ensure_blocks_allocated(48)
        blocks1 = set(cache1.allocated_blocks)
        blocks2 = set(cache2.allocated_blocks)
        self.assertEqual(len(blocks1 & blocks2), 0)
        self.assertEqual(len(self.pool.free_blocks), 15)

    def test_update_cache_raises_error(self):
        """SharedPagedKVCache.update_cache should not be called directly."""
        cache = SharedPagedKVCache(
            pool=self.pool, request_id=1, layer_idx=0, max_seq_length=128
        )
        key = torch.randn(1, 4, 5, 64)
        value = torch.randn(1, 4, 5, 64)
        with self.assertRaises(RuntimeError):
            cache.update_cache(key, value, concat_dim=2)


class TestPagedKVCachePoolBlocksTensor(TestCase):
    """The pool keeps a persistent int32 tensor mirror of each request's
    allocated blocks, in sync with the Python list, so merge_contexts can copy
    block rows without rebuilding torch.tensor(list) per step."""

    def setUp(self):
        PagedKVCachePool.reset()
        self.pool = PagedKVCachePool.initialize(
            total_blocks=64,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            num_layers=2,
        )

    def tearDown(self):
        PagedKVCachePool.reset()

    def test_tensor_created_on_allocate(self):
        self.pool.allocate_blocks(request_id=1, num_blocks=3)
        t = self.pool.get_blocks_tensor(1)
        self.assertIsNotNone(t)
        self.assertEqual(t.dtype, torch.int32)
        self.assertEqual(t.tolist(), self.pool.get_blocks_for_request(1))

    def test_tensor_grows_with_incremental_allocation(self):
        self.pool.allocate_blocks(request_id=7, num_blocks=2)
        self.pool.ensure_blocks_for_request(7, 5)  # +3 more
        t = self.pool.get_blocks_tensor(7)
        self.assertEqual(t.numel(), 5)
        self.assertEqual(t.tolist(), self.pool.get_blocks_for_request(7))

    def test_tensor_removed_on_free(self):
        self.pool.allocate_blocks(request_id=2, num_blocks=2)
        self.assertIsNotNone(self.pool.get_blocks_tensor(2))
        self.pool.free_blocks_for_request(2)
        self.assertIsNone(self.pool.get_blocks_tensor(2))

    def test_tensor_cleared_on_reset(self):
        self.pool.allocate_blocks(request_id=3, num_blocks=2)
        self.pool.reset_allocations()
        self.assertIsNone(self.pool.get_blocks_tensor(3))
        self.assertEqual(self.pool.allocated_blocks_tensor, {})

    def test_missing_request_returns_none(self):
        self.assertIsNone(self.pool.get_blocks_tensor(999))


def _make_config(**overrides):
    """Create a minimal mock config for PagedCache."""
    defaults = dict(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=256,
        num_hidden_layers=2,
        max_position_embeddings=4096,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestPagedCacheAllocation(TestCase):
    def test_pool_is_sized_when_first_offline_batch_arrives(self):
        config = _make_config()
        cache = PagedCache(
            config,
            block_size=16,
            enable_prefix_cache=False,
        )

        self.assertIsNone(cache._pool)

        cache.create_context(
            config,
            max_seq_length=17,
            batch_size=3,
        )

        self.assertEqual(cache._pool.total_blocks, 6)

    def test_pool_resize_clears_prefix_map(self):
        config = _make_config()
        cache = PagedCache(
            config,
            block_size=16,
            enable_prefix_cache=True,
        )
        context = cache.create_context(config, max_seq_length=16)
        cache._prefix_map.insert(b"old-pool", 0)
        cache.remove_context(context)

        cache.create_context(config, max_seq_length=64, batch_size=2)

        self.assertIsNone(cache._prefix_map.peek(b"old-pool"))

    def test_reserve_pool_honors_explicit_and_default_capacity(self):
        cases = (
            ({"max_total_tokens": 64}, {}),
            ({}, {"PACE_MAX_CACHE_TOKENS": "64"}),
        )
        for cache_kwargs, environment in cases:
            with self.subTest(cache_kwargs=cache_kwargs):
                config = _make_config()
                cache = PagedCache(
                    config,
                    block_size=16,
                    enable_prefix_cache=False,
                    **cache_kwargs,
                )

                self.assertIsNone(cache._pool)
                with patch.dict(os.environ, environment):
                    cache.reserve_pool()

                self.assertEqual(cache._pool.total_blocks, 4)

    def test_reserve_pool_honors_memory_budget(self):
        config = _make_config()
        probe = PagedCache(
            config,
            block_size=16,
            enable_prefix_cache=False,
        )
        memory_gb = probe._bytes_per_token * 64 / 1024**3
        cache = PagedCache(
            config,
            block_size=16,
            kv_cache_memory_gb=memory_gb,
            enable_prefix_cache=False,
        )

        cache.reserve_pool()

        self.assertEqual(cache._pool.total_blocks, 4)

    def test_offline_pool_honors_an_explicit_limit(self):
        config = _make_config()
        probe = PagedCache(config, block_size=16, enable_prefix_cache=False)
        cases = (
            ({"max_total_tokens": 64}, 4),
            ({"max_total_tokens": 256}, 16),
            ({"kv_cache_memory_gb": probe._bytes_per_token * 64 / 1024**3}, 4),
        )
        for cache_kwargs, expected_blocks in cases:
            with self.subTest(cache_kwargs=cache_kwargs):
                cache = PagedCache(
                    config,
                    block_size=16,
                    enable_prefix_cache=False,
                    **cache_kwargs,
                )

                # This batch alone would size a lazy pool to 6 blocks.
                cache.create_context(config, max_seq_length=17, batch_size=3)

                self.assertEqual(cache._pool.total_blocks, expected_blocks)


class TestPagedCacheMergeContexts(TestCase):
    """Tests for PagedCache.merge_contexts — the server decode metadata builder."""

    def _make_cache_and_contexts(self, num_reqs, seq_lens, block_size=16):
        config = _make_config()
        cache = PagedCache(config, block_size=block_size, max_total_tokens=8192)
        contexts = []
        for seq_len in seq_lens:
            ctx = cache.create_context(config, max_seq_length=4096)
            for co in ctx.cache_objects:
                co.seq_len = seq_len
                co._ensure_blocks_allocated(seq_len)
            contexts.append(ctx)
        return cache, contexts

    def test_basic_decode_slot_mapping(self):
        """slot_mapping should point to the correct physical slot for each request."""
        cache, contexts = self._make_cache_and_contexts(2, [10, 20], block_size=16)
        merged = cache.merge_contexts(contexts, query_len=1)
        meta = merged.paged_attn_metadata

        self.assertEqual(meta.slot_mapping.shape[0], 2)
        for i, ctx in enumerate(contexts):
            layer_cache = ctx.cache_objects[0]
            past_len = 10 if i == 0 else 20
            block_idx = past_len // 16
            block_offset = past_len % 16
            blocks = layer_cache.allocated_blocks
            expected_slot = blocks[block_idx] * 16 + block_offset
            self.assertEqual(
                int(meta.slot_mapping[i]),
                expected_slot,
                f"Request {i}: expected slot {expected_slot}, got {int(meta.slot_mapping[i])}",
            )

    def test_decode_crossing_block_boundary(self):
        """When past_len is at a block boundary, a new block should be allocated."""
        block_size = 16
        cache, contexts = self._make_cache_and_contexts(
            1, [block_size - 1], block_size=block_size
        )

        # First decode: past_len=15, still within block 0.
        merged = cache.merge_contexts(contexts, query_len=1)
        meta = merged.paged_attn_metadata
        blocks_before = contexts[0].cache_objects[0].allocated_blocks[:]
        self.assertEqual(len(blocks_before), 1)

        # After merge_contexts, seq_len is updated to 16.
        # Next decode: past_len=16, crosses into block 1.
        merged = cache.merge_contexts(contexts, query_len=1)
        meta = merged.paged_attn_metadata
        blocks_after = contexts[0].cache_objects[0].allocated_blocks
        self.assertEqual(
            len(blocks_after), 2, "A second block should be allocated at the boundary"
        )

        # slot_mapping should point into the second block at offset 0.
        expected_slot = blocks_after[1] * block_size + 0
        self.assertEqual(int(meta.slot_mapping[0]), expected_slot)

    def test_seq_lens_updated_after_merge(self):
        """merge_contexts should update seq_len on all layer cache objects."""
        cache, contexts = self._make_cache_and_contexts(2, [5, 10])
        cache.merge_contexts(contexts, query_len=1)

        for i, (ctx, expected) in enumerate(zip(contexts, [6, 11])):
            for layer_idx, co in enumerate(ctx.cache_objects):
                self.assertEqual(
                    co.seq_len,
                    expected,
                    f"Request {i}, layer {layer_idx}: seq_len should be {expected}",
                )

    def test_block_table_shape(self):
        """block_table should have shape [num_reqs, max_blocks_per_seq]."""
        cache, contexts = self._make_cache_and_contexts(3, [5, 20, 40], block_size=16)
        merged = cache.merge_contexts(contexts, query_len=1)
        meta = merged.paged_attn_metadata

        self.assertEqual(meta.block_table.shape[0], 3)
        max_blocks = (41 + 16 - 1) // 16  # longest seq_len=40 + 1 decode token
        self.assertEqual(meta.block_table.shape[1], max_blocks)

    def test_multiple_decode_steps(self):
        """Simulate several decode steps; slot_mapping should advance correctly."""
        block_size = 4
        cache, contexts = self._make_cache_and_contexts(1, [0], block_size=block_size)

        for step in range(10):
            merged = cache.merge_contexts(contexts, query_len=1)
            meta = merged.paged_attn_metadata

            past_len = step
            block_idx = past_len // block_size
            block_offset = past_len % block_size
            blocks = contexts[0].cache_objects[0].allocated_blocks
            expected_slot = blocks[block_idx] * block_size + block_offset
            self.assertEqual(
                int(meta.slot_mapping[0]),
                expected_slot,
                f"Step {step}: slot mismatch",
            )

    def _reference_block_table(self, contexts, max_blocks_per_seq):
        """Build the block_table the old torch.tensor(list) path would produce."""
        num_reqs = len(contexts)
        ref = torch.zeros(num_reqs, max_blocks_per_seq, dtype=torch.int32)
        for i, ctx in enumerate(contexts):
            blocks = ctx.cache_objects[0].allocated_blocks
            nb = min(len(blocks), max_blocks_per_seq)
            if nb:
                ref[i, :nb] = torch.tensor(blocks[:nb], dtype=torch.int32)
        return ref

    def test_block_table_matches_list_rebuild_reference(self):
        """The tensor-mirror fast path must produce the same block_table the
        torch.tensor(list) rebuild produced."""
        cache, contexts = self._make_cache_and_contexts(3, [5, 20, 40], block_size=16)
        merged = cache.merge_contexts(contexts, query_len=1)
        meta = merged.paged_attn_metadata
        ref = self._reference_block_table(contexts, meta.block_table.shape[1])
        self.assertTrue(torch.equal(meta.block_table, ref))

    def test_block_table_correct_across_boundary_crossings(self):
        """Across many decode steps (including block-boundary allocations that
        refresh the tensor mirror), block_table stays equal to the reference."""
        block_size = 4
        cache, contexts = self._make_cache_and_contexts(
            2, [3, 7], block_size=block_size
        )
        for _ in range(20):
            merged = cache.merge_contexts(contexts, query_len=1)
            meta = merged.paged_attn_metadata
            ref = self._reference_block_table(contexts, meta.block_table.shape[1])
            self.assertTrue(torch.equal(meta.block_table, ref))

    def test_merge_contexts_mixed_block_table_matches_reference(self):
        """merge_contexts_mixed uses the same tensor-mirror fill path."""
        cache, contexts = self._make_cache_and_contexts(3, [5, 20, 40], block_size=16)
        merged = cache.merge_contexts_mixed([(ctx, 1) for ctx in contexts])
        meta = merged.paged_attn_metadata
        ref = self._reference_block_table(contexts, meta.block_table.shape[1])
        self.assertTrue(torch.equal(meta.block_table, ref))


class TestPagedCacheRecurrentState(TestCase):
    """Tests for per-request GDN state on packed paged batches."""

    def _make_cache_and_contexts(self, seq_lens, hybrid=True, block_size=16):
        layer_types = ["linear_attention", "full_attention"] if hybrid else None
        config = _make_config(layer_types=layer_types)
        cache = PagedCache(config, block_size=block_size, max_total_tokens=8192)
        contexts = []
        for seq_len in seq_lens:
            ctx = cache.create_context(config, max_seq_length=4096)
            for co in ctx.cache_objects:
                co.seq_len = seq_len
                co._ensure_blocks_allocated(seq_len)
            contexts.append(ctx)
        return cache, contexts

    @staticmethod
    def _query_lens(merged):
        qsl = merged.paged_attn_metadata.query_start_loc
        return (qsl[1:] - qsl[:-1]).tolist()

    def test_context_owns_a_recurrent_store(self):
        _, contexts = self._make_cache_and_contexts([10])
        store = contexts[0].kv_cache_manager.recurrent_state
        self.assertIsInstance(store, RecurrentStateStore)
        self.assertEqual(store.num_layers, 1)

    def test_decode_merge_exposes_per_request_stores(self):
        cache, contexts = self._make_cache_and_contexts([10, 20, 30])
        stores = [c.kv_cache_manager.recurrent_state for c in contexts]

        merged = cache.merge_contexts(contexts, query_len=1)

        self.assertIsInstance(merged.recurrent_state, MergedRecurrentState)
        self.assertEqual(merged.recurrent_state.stores, stores)
        self.assertEqual(self._query_lens(merged), [1, 1, 1])

    def test_mixed_merge_carries_per_request_query_lens(self):
        cache, contexts = self._make_cache_and_contexts([5, 20, 40])
        stores = [c.kv_cache_manager.recurrent_state for c in contexts]
        query_lens = [4, 1, 7]

        merged = cache.merge_contexts_mixed(list(zip(contexts, query_lens)))

        self.assertEqual(merged.recurrent_state.stores, stores)
        self.assertEqual(self._query_lens(merged), query_lens)

    def test_full_attention_model_is_left_untouched(self):
        cache, contexts = self._make_cache_and_contexts([10, 20], hybrid=False)

        merged = cache.merge_contexts(contexts, query_len=1)

        self.assertIsNone(getattr(merged, "recurrent_state", None))
