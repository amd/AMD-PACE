# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
# python -m pytest tests/llm_infra/test_prefix_cache.py -v

import unittest

from pace.llm.attention.prefix_cache import (
    BlockHashMap,
    NONE_HASH,
    _HASH_ALGO,
    compute_block_hashes,
    hash_block_tokens,
)


class TestChainedBlockHash(unittest.TestCase):
    """Tests for the parent-chained block hash function."""

    def test_same_tokens_same_parent_same_hash(self):
        tokens = (1, 2, 3, 4)
        self.assertEqual(
            hash_block_tokens(NONE_HASH, tokens), hash_block_tokens(NONE_HASH, tokens)
        )

    def test_different_tokens_different_hash(self):
        h1 = hash_block_tokens(NONE_HASH, (1, 2, 3, 4))
        h2 = hash_block_tokens(NONE_HASH, (5, 6, 7, 8))
        self.assertNotEqual(h1, h2)

    def test_different_parent_different_hash(self):
        parent_a = hash_block_tokens(NONE_HASH, (10, 20, 30, 40))
        parent_b = hash_block_tokens(NONE_HASH, (50, 60, 70, 80))
        h1 = hash_block_tokens(parent_a, (1, 2, 3, 4))
        h2 = hash_block_tokens(parent_b, (1, 2, 3, 4))
        self.assertNotEqual(h1, h2)

    def test_hash_digest_size(self):
        h = hash_block_tokens(NONE_HASH, (1, 2, 3, 4))
        self.assertEqual(len(h), 16 if _HASH_ALGO == "xxhash" else 32)

    def test_order_matters(self):
        h1 = hash_block_tokens(NONE_HASH, (1, 2, 3, 4))
        h2 = hash_block_tokens(NONE_HASH, (4, 3, 2, 1))
        self.assertNotEqual(h1, h2)


class TestComputeBlockHashes(unittest.TestCase):
    """Tests for compute_block_hashes chain construction."""

    def test_chain_length(self):
        hashes = compute_block_hashes(list(range(12)), block_size=4)
        self.assertEqual(len(hashes), 3)

    def test_partial_block_ignored(self):
        hashes = compute_block_hashes(list(range(10)), block_size=4)
        self.assertEqual(len(hashes), 2)

    def test_chaining(self):
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        manual_h0 = hash_block_tokens(NONE_HASH, tuple(range(0, 4)))
        manual_h1 = hash_block_tokens(manual_h0, tuple(range(4, 8)))
        self.assertEqual(hashes[0], manual_h0)
        self.assertEqual(hashes[1], manual_h1)

    def test_shared_prefix_same_hashes(self):
        prefix = list(range(100, 108))
        ha = compute_block_hashes(prefix + list(range(200, 204)), block_size=4)
        hb = compute_block_hashes(prefix + list(range(300, 304)), block_size=4)
        self.assertEqual(ha[0], hb[0])
        self.assertEqual(ha[1], hb[1])
        self.assertNotEqual(ha[2], hb[2])

    def test_divergent_prefixes_different_hashes(self):
        ha = compute_block_hashes([1, 2, 3, 4, 10, 20, 30, 40], block_size=4)
        hb = compute_block_hashes([5, 6, 7, 8, 10, 20, 30, 40], block_size=4)
        self.assertNotEqual(ha[0], hb[0])
        self.assertNotEqual(ha[1], hb[1])


class TestBlockHashMapBasic(unittest.TestCase):
    """Basic BlockHashMap operations."""

    def test_empty_match(self):
        m = BlockHashMap()
        matched, num_blocks = m.match_prefix(
            compute_block_hashes(list(range(8)), block_size=4)
        )
        self.assertEqual(matched, [])
        self.assertEqual(num_blocks, 0)

    def test_insert_and_match(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        m.insert(hashes[0], 10)
        m.insert(hashes[1], 20)
        matched, num_blocks = m.match_prefix(hashes)
        self.assertEqual(matched, [10, 20])
        self.assertEqual(num_blocks, 2)

    def test_partial_match(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        m.insert(hashes[0], 10)
        matched, num_blocks = m.match_prefix(hashes)
        self.assertEqual(matched, [10])
        self.assertEqual(num_blocks, 1)

    def test_no_match_on_different_prefix(self):
        m = BlockHashMap()
        hashes_a = compute_block_hashes(list(range(8)), block_size=4)
        hashes_b = compute_block_hashes(list(range(100, 108)), block_size=4)
        m.insert(hashes_a[0], 10)
        m.insert(hashes_a[1], 20)
        matched, num_blocks = m.match_prefix(hashes_b)
        self.assertEqual(matched, [])
        self.assertEqual(num_blocks, 0)


class TestBlockHashMapDedup(unittest.TestCase):
    """Inserting an existing hash returns the duplicate block for freeing."""

    def test_first_insert_returns_none(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        self.assertIsNone(m.insert(h, 10))

    def test_second_insert_returns_duplicate(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 10)
        self.assertEqual(m.insert(h, 20), 20)

    def test_dedup_increments_refcount(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 10)
        m.insert(h, 20)
        self.assertEqual(m._map[h].ref_count, 2)

    def test_dedup_stats(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 10)
        m.insert(h, 20)
        stats = m.get_stats()
        self.assertEqual(stats["prefix_cache_dedup_frees"], 1)
        self.assertEqual(stats["prefix_cache_total_inserts"], 1)


class TestBlockHashMapRefCounting(unittest.TestCase):
    """Reference counting and LRU tracking."""

    def setUp(self):
        self.m = BlockHashMap()
        self.h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        self.m.insert(self.h, 10)

    def test_initial_refcount_is_one(self):
        self.assertEqual(self.m._map[self.h].ref_count, 1)

    def test_increment_removes_from_lru(self):
        self.m.decrement_ref(self.h)
        self.assertIn(self.h, self.m._lru)
        self.m.increment_ref(self.h)
        self.assertNotIn(self.h, self.m._lru)

    def test_decrement_to_zero_adds_to_lru(self):
        self.m.decrement_ref(self.h)
        self.assertIn(self.h, self.m._lru)
        self.assertEqual(self.m._map[self.h].ref_count, 0)

    def test_decrement_clamps_at_zero(self):
        self.m.decrement_ref(self.h)
        self.m.decrement_ref(self.h)
        self.assertEqual(self.m._map[self.h].ref_count, 0)


class TestBlockHashMapEviction(unittest.TestCase):
    """LRU eviction."""

    def test_evict_returns_block_indices(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 42)
        m.decrement_ref(h)
        self.assertEqual(m.evict(1), [42])
        self.assertEqual(m.num_cached, 0)

    def test_evict_respects_lru_order(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(12)), block_size=4)
        for h, idx in zip(hashes, (10, 20, 30)):
            m.insert(h, idx)
        for h in hashes:
            m.decrement_ref(h)
        self.assertEqual(m.evict(1), [10])
        self.assertEqual(m.evict(1), [20])

    def test_evict_skips_referenced_blocks(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        m.insert(hashes[0], 10)
        m.insert(hashes[1], 20)
        m.decrement_ref(hashes[1])
        self.assertEqual(m.evict(2), [20])

    def test_evict_more_than_available(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 10)
        m.decrement_ref(h)
        self.assertEqual(m.evict(5), [10])

    def test_eviction_stats(self):
        m = BlockHashMap()
        h = compute_block_hashes([1, 2, 3, 4], block_size=4)[0]
        m.insert(h, 10)
        m.decrement_ref(h)
        m.evict(1)
        self.assertEqual(m.get_stats()["prefix_cache_total_evictions"], 1)

    def test_num_evictable(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        m.insert(hashes[0], 10)
        m.insert(hashes[1], 20)
        self.assertEqual(m.num_evictable, 0)
        m.decrement_ref(hashes[0])
        self.assertEqual(m.num_evictable, 1)
        m.decrement_ref(hashes[1])
        self.assertEqual(m.num_evictable, 2)


class TestBlockHashMapStats(unittest.TestCase):
    """Comprehensive stats reporting."""

    def test_full_stats(self):
        m = BlockHashMap()
        hashes = compute_block_hashes(list(range(8)), block_size=4)
        m.insert(hashes[0], 10)
        m.insert(hashes[1], 20)
        # Query 1: 8-token prompt, both blocks hit -> 8 cached tokens.
        _, nb1 = m.match_prefix(hashes)
        m.record_query(8, nb1 * 4)
        # Query 2: 8-token prompt, no match -> 0 cached tokens.
        _, nb2 = m.match_prefix(
            compute_block_hashes(list(range(100, 108)), block_size=4)
        )
        m.record_query(8, nb2 * 4)
        stats = m.get_stats()
        self.assertEqual(stats["prefix_cache_queries"], 16)  # tokens
        self.assertEqual(stats["prefix_cache_hits"], 8)  # cached tokens
        self.assertAlmostEqual(stats["prefix_cache_hit_rate"], 0.5)
        self.assertEqual(stats["prefix_cache_cached_blocks"], 2)
        self.assertEqual(stats["prefix_cache_total_inserts"], 2)


class TestChatPrefixScenario(unittest.TestCase):
    """Simulates an agentic / chat-completions use case."""

    def setUp(self):
        self.m = BlockHashMap()
        self.block_size = 4
        self.system_prefix = list(range(1000, 1016))
        self.system_hashes = compute_block_hashes(self.system_prefix, self.block_size)
        for i, h in enumerate(self.system_hashes):
            self.m.insert(h, i)

    def test_user_turn_reuses_system_prefix(self):
        user_turn = self.system_prefix + list(range(2000, 2008))
        matched, num_blocks = self.m.match_prefix(
            compute_block_hashes(user_turn, self.block_size)
        )
        self.assertEqual(num_blocks, 4)
        self.assertEqual(matched, [0, 1, 2, 3])

    def test_different_users_same_prefix(self):
        h1 = compute_block_hashes(
            self.system_prefix + list(range(2000, 2008)), self.block_size
        )
        h2 = compute_block_hashes(
            self.system_prefix + list(range(3000, 3008)), self.block_size
        )
        m1, n1 = self.m.match_prefix(h1)
        m2, n2 = self.m.match_prefix(h2)
        self.assertEqual(m1, m2)
        self.assertEqual(n1, n2)

    def test_accumulating_context(self):
        turn1_tokens = self.system_prefix + list(range(2000, 2008))
        turn1_hashes = compute_block_hashes(turn1_tokens, self.block_size)
        for i, h in enumerate(turn1_hashes[4:], start=10):
            self.m.insert(h, i)
        turn2_hashes = compute_block_hashes(
            turn1_tokens + list(range(3000, 3004)), self.block_size
        )
        _, num_blocks = self.m.match_prefix(turn2_hashes)
        self.assertEqual(num_blocks, 6)

    def test_divergent_prefix_no_false_match(self):
        same_suffix = list(range(2000, 2008))
        ha = compute_block_hashes(self.system_prefix + same_suffix, self.block_size)
        hb = compute_block_hashes(
            list(range(5000, 5016)) + same_suffix, self.block_size
        )
        for h in ha[4:]:
            self.m.insert(h, 99)
        _, num_blocks = self.m.match_prefix(hb)
        self.assertEqual(num_blocks, 0)


class TestPrefixCacheMixinProtocol(unittest.TestCase):
    """Tests the mixin protocol against the real PrefixCacheMixin."""

    class _MockContext:
        def __init__(self, block_indices=None):
            self.block_indices = block_indices or []
            self.prefix_matched_tokens = 0
            self.prefix_matched_blocks = []
            self.prefix_token_ids = None
            self.prefix_held_hashes = []

    class _MockBackend:
        """Minimal PrefixCacheMixin subclass backed by an in-memory block list."""

        def __init__(self, block_size, enable):
            from pace.llm.attention.prefix_cache import PrefixCacheMixin

            self._impl = PrefixCacheMixin()
            self._impl._init_prefix_cache(block_size, enable)
            self._impl._get_block_indices_for_context = lambda ctx: ctx.block_indices
            self.freed_duplicates = []
            self._impl._free_duplicate_block = self.freed_duplicates.append

        def __getattr__(self, name):
            return getattr(self._impl, name)

    def test_disabled_returns_no_match(self):
        backend = self._MockBackend(block_size=4, enable=False)
        self.assertFalse(backend.prefix_cache_enabled)
        blocks, matched = backend.try_prefix_match([1, 2, 3, 4])
        self.assertEqual(blocks, [])
        self.assertEqual(matched, 0)

    def test_full_lifecycle(self):
        backend = self._MockBackend(block_size=4, enable=True)
        tokens = list(range(100, 108))
        backend.insert_prefix(self._MockContext(block_indices=[10, 20]), tokens)

        blocks, matched = backend.try_prefix_match(tokens + [999, 998, 997, 996])
        self.assertEqual(blocks, [10, 20])
        self.assertEqual(matched, 8)

        ctx2 = self._MockContext()
        backend.claim_prefix_blocks(ctx2, tokens, blocks, matched)
        self.assertEqual(ctx2.prefix_matched_tokens, 8)
        backend._release_prefix_refs(ctx2)

    def test_dedup_frees_duplicate_blocks(self):
        backend = self._MockBackend(block_size=4, enable=True)
        tokens = list(range(100, 108))
        backend.insert_prefix(self._MockContext(block_indices=[10, 20]), tokens)
        backend.insert_prefix(self._MockContext(block_indices=[30, 40]), tokens)
        self.assertEqual(backend.freed_duplicates, [30, 40])

    def test_chained_hash_prevents_false_match(self):
        backend = self._MockBackend(block_size=4, enable=True)
        backend.insert_prefix(
            self._MockContext(block_indices=[0, 1]), [1, 2, 3, 4, 10, 20, 30, 40]
        )
        _, matched = backend.try_prefix_match([5, 6, 7, 8, 10, 20, 30, 40])
        self.assertEqual(matched, 0)

    def test_insert_records_and_release_frees_all_refs(self):
        backend = self._MockBackend(block_size=4, enable=True)
        tokens = list(range(100, 108))
        ctx = self._MockContext(block_indices=[10, 20])
        backend.insert_prefix(ctx, tokens)
        hashes = compute_block_hashes(tokens, 4)
        self.assertEqual(ctx.prefix_held_hashes, hashes)
        self.assertTrue(all(backend._prefix_map._map[h].ref_count == 1 for h in hashes))
        backend._release_prefix_refs(ctx)
        self.assertTrue(all(backend._prefix_map._map[h].ref_count == 0 for h in hashes))
        self.assertEqual(ctx.prefix_held_hashes, [])
        backend._release_prefix_refs(ctx)  # idempotent, no double-free
        self.assertTrue(all(backend._prefix_map._map[h].ref_count == 0 for h in hashes))

    def test_dedup_repoints_owner_before_free(self):
        backend = self._MockBackend(block_size=4, enable=True)
        backend._impl._repoint_duplicate_block = (
            lambda ctx, pos, old, canon: ctx.block_indices.__setitem__(pos, canon)
        )
        tokens = list(range(100, 108))
        backend.insert_prefix(self._MockContext(block_indices=[10, 20]), tokens)
        ctx = self._MockContext(block_indices=[30, 40])
        backend.insert_prefix(ctx, tokens)
        self.assertEqual(backend.freed_duplicates, [30, 40])
        self.assertEqual(ctx.block_indices, [10, 20])

    def test_same_block_reinsert_is_not_a_dedup(self):
        backend = self._MockBackend(block_size=4, enable=True)
        tokens = list(range(100, 108))
        ctx = self._MockContext(block_indices=[10, 20])
        backend.insert_prefix(ctx, tokens)
        backend.insert_prefix(ctx, tokens)
        self.assertEqual(backend.freed_duplicates, [])
        hashes = compute_block_hashes(tokens, 4)
        self.assertTrue(all(backend._prefix_map._map[h].ref_count == 1 for h in hashes))


class TestPrefixCacheConfigInit(unittest.TestCase):
    """Tests for _init_prefix_cache_from_config block-size resolution."""

    class _FakeConfig:
        num_key_value_heads = 8
        num_attention_heads = 8
        hidden_size = 512
        head_dim = 64

    def _make_mixin(self):
        from pace.llm.attention.prefix_cache import PrefixCacheMixin

        return PrefixCacheMixin()

    def test_explicit_block_size_wins_over_autotune(self):
        mixin = self._make_mixin()
        resolved = mixin._init_prefix_cache_from_config(
            self._FakeConfig(),
            "Test",
            autotune_fn=lambda kv, hd: 128,
            block_size=32,
        )
        self.assertEqual(resolved, 32)

    def test_autotune_fn_used_when_no_block_size(self):
        captured = {}

        def autotune(num_kv_heads, head_dim):
            captured["kv"] = num_kv_heads
            captured["hd"] = head_dim
            return 256

        mixin = self._make_mixin()
        resolved = mixin._init_prefix_cache_from_config(
            self._FakeConfig(), "Test", autotune_fn=autotune
        )
        self.assertEqual(resolved, 256)
        self.assertEqual(captured, {"kv": 8, "hd": 64})

    def test_defaults_to_16_without_autotune(self):
        mixin = self._make_mixin()
        resolved = mixin._init_prefix_cache_from_config(self._FakeConfig(), "Test")
        self.assertEqual(resolved, 16)


class TestHashAlgorithmSelection(unittest.TestCase):
    """Verify the hash algorithm was resolved correctly."""

    def test_algo_is_known(self):
        self.assertIn(_HASH_ALGO, ("xxhash", "sha256"))

    def test_none_hash_matches_algo(self):
        self.assertEqual(len(NONE_HASH), 16 if _HASH_ALGO == "xxhash" else 32)

    def test_consistency_across_calls(self):
        tokens = tuple(range(1, 17))
        self.assertEqual(
            hash_block_tokens(NONE_HASH, tokens), hash_block_tokens(NONE_HASH, tokens)
        )

    def test_xxhash_is_default(self):
        # xxhash is a hard runtime dependency (no availability fallback), so
        # the default algorithm must resolve to xxhash when unset.
        import os

        if os.getenv("PACE_PREFIX_HASH", "xxhash").lower() == "xxhash":
            self.assertEqual(_HASH_ALGO, "xxhash")


class TestSlabReuseInsertSafety(unittest.TestCase):
    """Slab dedup repoints the owner onto the canonical block, then frees the dup.

    Slab prefix reuse is zero-copy. On dedup the owning sequence is first
    repointed (``reassign_block``) onto the canonical block, so releasing the
    now-unreferenced duplicate back to the pool is safe and reclaims memory.
    """

    class _FakeManager:
        """Records retain/release/reassign calls; serves block indices per context."""

        def __init__(self):
            self.retained = []
            self.released = []
            self.reassigns = []
            self._blocks_by_token = {}

        def get_block_indices(self, token):
            return self._blocks_by_token.get(token, [])

        def retain_blocks(self, blocks):
            self.retained.extend(blocks)

        def release_retained_blocks(self, blocks):
            self.released.extend(blocks)

        def reassign_block(self, token, block_pos, new_pool_block):
            self.reassigns.append((token, block_pos, new_pool_block))
            self._blocks_by_token[token][block_pos] = new_pool_block

    class _Ctx:
        def __init__(self, token, matched_tokens=0):
            self.tokens = [token]
            self.prefix_matched_tokens = matched_tokens
            self.prefix_held_hashes = []

    def _make_cache(self, block_size=4):
        from pace.llm.attention.slab.cache import SlabCache

        cache = SlabCache.__new__(SlabCache)
        cache._manager = self._FakeManager()
        cache._prefix_map = BlockHashMap()
        cache._prefix_block_size = block_size
        return cache

    def test_dedup_repoints_owner_then_releases_duplicate(self):
        cache = self._make_cache()
        tokens = list(range(100, 108))  # two full blocks of size 4
        cache._manager._blocks_by_token["A"] = [10, 20]
        cache._manager._blocks_by_token["B"] = [30, 40]

        cache.insert_prefix(self._Ctx("A"), tokens)
        # Second identical request: blocks 30/40 dedup to canonical 10/20. B is
        # repointed onto 10/20, and its duplicates 30/40 are released.
        cache.insert_prefix(self._Ctx("B"), tokens)
        self.assertEqual(cache._manager.reassigns, [("B", 0, 10), ("B", 1, 20)])
        self.assertEqual(cache._manager.released, [30, 40])
        self.assertEqual(cache._manager._blocks_by_token["B"], [10, 20])

    def test_reused_prefix_blocks_are_skipped(self):
        cache = self._make_cache()
        # 3 full blocks: first 2 reused (matched_tokens=8), 3rd is new suffix.
        tokens = list(range(200, 212))
        cache._manager._blocks_by_token["S"] = [11, 22, 33]

        inserted = cache.insert_prefix(self._Ctx("S", matched_tokens=8), tokens)

        # Only the suffix block (index 33) is retained and inserted.
        self.assertEqual(cache._manager.retained, [33])
        self.assertEqual(inserted, 1)
        self.assertEqual(cache._manager.released, [])
        # The reused prefix region was not registered as a standalone entry:
        # a fresh lookup breaks on the (unregistered) first prefix block.
        _, matched_tokens = cache.try_prefix_match(tokens)
        self.assertEqual(matched_tokens, 0)

    def test_free_duplicate_block_releases(self):
        cache = self._make_cache()
        cache._free_duplicate_block(99)
        self.assertEqual(cache._manager.released, [99])


if __name__ == "__main__":
    unittest.main()
