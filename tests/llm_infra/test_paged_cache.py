# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
# python -m pytest tests/llm_infra/test_paged_cache.py -v

import torch
from torch.testing._internal.common_utils import TestCase

from pace.llm.attention.paged.cache import (
    PagedKVCache,
    PagedKVCachePool,
    SharedPagedKVCache,
)


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
