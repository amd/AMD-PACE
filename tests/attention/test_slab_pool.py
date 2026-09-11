# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for SlabPool C++ pool: pool management, sequence lifecycle,
block allocation, cache update correctness, input validation.
"""

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401


def _create_pool(total_blocks=64, num_kv_heads=4, head_dim=64, block_size=16):
    return torch.classes.pace.SlabPool(total_blocks, num_kv_heads, head_dim, block_size)


class TestSlabPoolCreation(TestCase):
    def test_pool_creation(self):
        pool = _create_pool()
        self.assertEqual(pool.get_free_block_count(), 64)

    def test_pool_with_autotune_block_size(self):
        from pace.llm.attention.slab.cache import autotune_block_size

        bs = autotune_block_size(4, 64)
        pool = torch.classes.pace.SlabPool(32, 4, 64, bs)
        self.assertEqual(pool.get_free_block_count(), 32)

    def test_rejects_nonpositive_total_blocks(self):
        with self.assertRaisesRegex(RuntimeError, "total_blocks"):
            _create_pool(total_blocks=0)

    def test_rejects_nonpositive_num_kv_heads(self):
        with self.assertRaisesRegex(RuntimeError, "num_kv_heads"):
            _create_pool(num_kv_heads=0)

    def test_rejects_nonpositive_head_dim(self):
        with self.assertRaisesRegex(RuntimeError, "head_dim"):
            _create_pool(head_dim=0)


class TestSlabPoolSequenceLifecycle(TestCase):
    def test_create_sequence(self):
        pool = _create_pool()
        pool.create_sequence(0, 256)
        self.assertEqual(pool.get_sequence_length(0), 0)

    def test_create_multiple_sequences(self):
        pool = _create_pool()
        for i in range(4):
            pool.create_sequence(i, 128)
        for i in range(4):
            self.assertEqual(pool.get_sequence_length(i), 0)

    def test_remove_sequence(self):
        pool = _create_pool()
        pool.create_sequence(0, 256)
        pool.remove_sequence(0)

    def test_free_blocks_restored_after_remove(self):
        pool = _create_pool(total_blocks=32, num_kv_heads=2, head_dim=32, block_size=16)
        initial_free = pool.get_free_block_count()
        pool.create_sequence(0, 64)
        key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        after_update = pool.get_free_block_count()
        self.assertLess(after_update, initial_free)
        pool.remove_sequence(0)
        self.assertEqual(pool.get_free_block_count(), initial_free)


class TestSlabPoolCacheUpdate(TestCase):
    def test_sequence_length_after_update(self):
        pool = _create_pool(num_kv_heads=2, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        key = torch.randn(1, 4, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 4, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        self.assertEqual(pool.get_sequence_length(0), 4)

    def test_incremental_updates(self):
        pool = _create_pool(num_kv_heads=2, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        for _ in range(5):
            key = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
            value = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
            pool.cache_update([0], key, value, [])
        self.assertEqual(pool.get_sequence_length(0), 15)

    def test_batched_update(self):
        pool = _create_pool(num_kv_heads=2, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        pool.create_sequence(1, 64)
        keys = torch.randn(2, 4, 2, 32, dtype=torch.bfloat16)
        values = torch.randn(2, 4, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0, 1], keys, values, [])
        self.assertEqual(pool.get_sequence_length(0), 4)
        self.assertEqual(pool.get_sequence_length(1), 4)

    def test_truncate_sequence(self):
        pool = _create_pool(num_kv_heads=2, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        self.assertEqual(pool.get_sequence_length(0), 16)
        pool.truncate_sequence(0, 4)
        self.assertEqual(pool.get_sequence_length(0), 12)


class TestSlabPoolInputValidation(TestCase):
    def test_head_dim_mismatch_single_update(self):
        """cache_update must reject tensors with wrong head_dim."""
        pool = _create_pool(num_kv_heads=4, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        wrong_key = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
        wrong_val = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "head_dim"):
            pool.cache_update([0], wrong_key, wrong_val, [])

    def test_head_dim_mismatch_batched_update(self):
        """cache_update must reject tensors with wrong head_dim (batched)."""
        pool = _create_pool(num_kv_heads=4, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        wrong_key = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
        wrong_val = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "head_dim"):
            pool.cache_update([0], wrong_key, wrong_val, [])

    def test_num_kv_heads_mismatch_batched_update(self):
        """cache_update must reject tensors with wrong num_kv_heads."""
        pool = _create_pool(num_kv_heads=4, head_dim=64, block_size=16)
        pool.create_sequence(0, 64)
        wrong_key = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16)
        wrong_val = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "num_kv_heads"):
            pool.cache_update([0], wrong_key, wrong_val, [])

    def test_head_dim_mismatch_attention(self):
        """attention must reject query with wrong head_dim."""
        pool = _create_pool(num_kv_heads=4, head_dim=32, block_size=16)
        pool.create_sequence(0, 64)
        key = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
        val = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, val, [])
        wrong_q = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "head_dim"):
            pool.attention([0], wrong_q, [], [], 0.125, 0, torch.tensor([]), 0)

    def test_wrong_dtype_rejected(self):
        """cache_update must reject non-BFloat16 tensors."""
        pool = _create_pool(num_kv_heads=4, head_dim=64, block_size=16)
        pool.create_sequence(0, 64)
        key = torch.randn(1, 4, 4, 64, dtype=torch.float32)
        val = torch.randn(1, 4, 4, 64, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "BFloat16"):
            pool.cache_update([0], key, val, [])

    def test_duplicate_seq_id_rejected(self):
        """create_sequence must reject duplicate seq_id."""
        pool = _create_pool()
        pool.create_sequence(0, 64)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            pool.create_sequence(0, 64)


class TestSlabPoolPrefixBlockValidation(TestCase):
    def test_assign_prefix_blocks_rejects_invalid_indices_atomically(self):
        for invalid_idx in (-1, 4):
            with self.subTest(invalid_idx=invalid_idx):
                pool = _create_pool(
                    total_blocks=4,
                    num_kv_heads=2,
                    head_dim=32,
                    block_size=16,
                )
                pool.create_sequence(0, 64)

                with self.assertRaisesRegex(RuntimeError, "block index.*out of range"):
                    pool.assign_prefix_blocks(0, [invalid_idx], 16)

                self.assertEqual(pool.get_sequence_length(0), 0)
                self.assertEqual(pool.get_free_block_count(), 4)

    def test_retain_blocks_rejects_invalid_indices_atomically(self):
        for invalid_idx in (-1, 4):
            with self.subTest(invalid_idx=invalid_idx):
                pool = _create_pool(total_blocks=4)

                with self.assertRaisesRegex(RuntimeError, "block index.*out of range"):
                    pool.retain_blocks([0, invalid_idx])

                pool.release_retained_blocks([0])
                self.assertEqual(pool.get_free_block_count(), 4)

    def test_retain_blocks_rejects_unallocated_block(self):
        pool = _create_pool(total_blocks=4)

        with self.assertRaisesRegex(RuntimeError, "unallocated block"):
            pool.retain_blocks([0])

        self.assertEqual(pool.get_free_block_count(), 4)

    def test_retain_blocks_allocated_validation_is_atomic(self):
        pool = _create_pool(
            total_blocks=4,
            num_kv_heads=2,
            head_dim=32,
            block_size=16,
        )
        pool.create_sequence(0, 64)
        key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        allocated = pool.get_block_indices(0)[0]
        unallocated = next(index for index in range(4) if index != allocated)

        with self.assertRaisesRegex(RuntimeError, "unallocated block"):
            pool.retain_blocks([allocated, unallocated])

        pool.remove_sequence(0)
        self.assertEqual(pool.get_free_block_count(), 4)

    def test_assign_prefix_blocks_requires_retained_allocated_blocks(self):
        pool = _create_pool(
            total_blocks=4,
            num_kv_heads=2,
            head_dim=32,
            block_size=16,
        )
        pool.create_sequence(0, 64)
        pool.create_sequence(1, 64)

        with self.assertRaisesRegex(RuntimeError, "not allocated"):
            pool.assign_prefix_blocks(1, [0], 16)

        key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        block = pool.get_block_indices(0)[0]
        with self.assertRaisesRegex(RuntimeError, "not retained"):
            pool.assign_prefix_blocks(1, [block], 16)

        self.assertEqual(pool.get_sequence_length(1), 0)

    def test_assign_prefix_blocks_rejects_duplicate_sources(self):
        pool = _create_pool(
            total_blocks=4,
            num_kv_heads=2,
            head_dim=32,
            block_size=16,
        )
        pool.create_sequence(0, 64)
        pool.create_sequence(1, 64)
        key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
        pool.cache_update([0], key, value, [])
        block = pool.get_block_indices(0)[0]
        pool.retain_blocks([block])

        with self.assertRaisesRegex(RuntimeError, "duplicate prefix source"):
            pool.assign_prefix_blocks(1, [block, block], 32)

        self.assertEqual(pool.get_sequence_length(1), 0)

    def test_release_retained_blocks_rejects_invalid_indices_atomically(self):
        for invalid_idx in (-1, 4):
            with self.subTest(invalid_idx=invalid_idx):
                pool = _create_pool(
                    total_blocks=4,
                    num_kv_heads=2,
                    head_dim=32,
                    block_size=16,
                )
                pool.create_sequence(0, 64)
                key = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
                value = torch.randn(1, 16, 2, 32, dtype=torch.bfloat16)
                pool.cache_update([0], key, value, [])
                retained_idx = pool.get_block_indices(0)[0]
                pool.retain_blocks([retained_idx])

                with self.assertRaisesRegex(RuntimeError, "block index.*out of range"):
                    pool.release_retained_blocks([retained_idx, invalid_idx])

                self.assertEqual(pool.get_free_block_count(), 3)
                pool.release_retained_blocks([retained_idx])
                self.assertEqual(pool.get_free_block_count(), 4)


class TestSlabPoolExhaustion(TestCase):
    """Test pool behavior when all blocks are consumed."""

    def test_cache_update_raises_on_full_pool(self):
        """cache_update must raise when no free blocks remain."""
        # Small pool: 2 blocks, block_size=16, head_dim=64, 2 kv_heads
        pool = _create_pool(total_blocks=2, num_kv_heads=2, head_dim=64, block_size=16)
        pool.create_sequence(0, 64)

        # Fill 32 tokens (2 blocks * 16 tokens/block = full)
        k = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        pool.cache_update([0], k, v, [])

        # Pool is full — next update should fail
        k2 = torch.randn(1, 1, 2, 64, dtype=torch.bfloat16)
        v2 = torch.randn(1, 1, 2, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "no free blocks"):
            pool.cache_update([0], k2, v2, [])

    def test_free_blocks_restored_after_exhaust_and_remove(self):
        """After exhaustion and removal, blocks return to free list."""
        pool = _create_pool(total_blocks=4, num_kv_heads=2, head_dim=64, block_size=16)
        initial_free = pool.get_free_block_count()

        pool.create_sequence(0, 64)
        k = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16)
        pool.cache_update([0], k, v, [])
        self.assertEqual(pool.get_free_block_count(), 0)

        pool.remove_sequence(0)
        self.assertEqual(pool.get_free_block_count(), initial_free)


class TestSlabPoolReservation(TestCase):
    """reserve_blocks pre-allocates a block table without writing any KV."""

    def test_reserve_allocates_without_advancing_seq_len(self):
        pool = _create_pool(total_blocks=8, block_size=16)
        pool.create_sequence(0, 256)

        pool.reserve_blocks(0, 3)

        self.assertEqual(len(pool.get_block_indices(0)), 3)
        self.assertEqual(pool.get_sequence_length(0), 0)
        self.assertEqual(pool.get_free_block_count(), 5)

    def test_reserve_is_idempotent(self):
        pool = _create_pool(total_blocks=8, block_size=16)
        pool.create_sequence(0, 256)

        pool.reserve_blocks(0, 3)
        reserved = pool.get_block_indices(0)
        pool.reserve_blocks(0, 3)

        self.assertEqual(pool.get_block_indices(0), reserved)
        self.assertEqual(pool.get_free_block_count(), 5)

    def test_cache_update_writes_into_reserved_blocks(self):
        pool = _create_pool(total_blocks=8, num_kv_heads=2, head_dim=64, block_size=16)
        pool.create_sequence(0, 256)
        pool.reserve_blocks(0, 2)
        reserved = pool.get_block_indices(0)

        k = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        pool.cache_update([0], k, v, [])

        self.assertEqual(pool.get_block_indices(0), reserved)
        self.assertEqual(pool.get_sequence_length(0), 32)
        self.assertEqual(pool.get_free_block_count(), 6)


class TestSlabPoolDoubleFree(TestCase):
    """A retained block released early must not be freed twice."""

    def test_remove_sequence_skips_blocks_already_released(self):
        pool = _create_pool(total_blocks=4, num_kv_heads=2, head_dim=64, block_size=16)
        pool.create_sequence(0, 64)
        k = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 32, 2, 64, dtype=torch.bfloat16)
        pool.cache_update([0], k, v, [])
        blocks = pool.get_block_indices(0)

        # Evicting a published block releases it while the sequence still
        # lists it; teardown then reaches the same block a second time.
        pool.retain_blocks(blocks)
        pool.release_retained_blocks(blocks)
        pool.remove_sequence(0)

        self.assertEqual(pool.get_free_block_count(), 4)

    def test_pool_stays_usable_after_a_skipped_double_free(self):
        pool = _create_pool(total_blocks=4, num_kv_heads=2, head_dim=64, block_size=16)
        pool.create_sequence(0, 64)
        k = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16)
        pool.cache_update([0], k, v, [])
        blocks = pool.get_block_indices(0)
        pool.retain_blocks(blocks)
        pool.release_retained_blocks(blocks)
        pool.remove_sequence(0)

        # Every block must still be handed out exactly once.
        pool.create_sequence(1, 64)
        pool.cache_update([1], k, v, [])

        self.assertEqual(sorted(pool.get_block_indices(1)), sorted(blocks))
        self.assertEqual(pool.get_free_block_count(), 0)
