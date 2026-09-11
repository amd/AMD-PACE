# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""vLLM-style automatic prefix caching for PACE's paged and slab KV caches.

- **Hash-map index** keyed by parent-chained block hashes. Each block's
  identity is ``hash(parent_hash, token_ids_in_block)``, so two sequences
  sharing the first N blocks produce identical hashes for those N blocks.
- **LRU eviction**: blocks with ``ref_count == 0`` are tracked in an
  ``OrderedDict`` and evicted oldest-first when the pool needs space.
- **Dedup on insert**: if a block hash already exists, the duplicate physical
  block is returned to the caller for freeing instead of leaking.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import xxhash
from transformers import PretrainedConfig

# Digest is trusted without token re-verification; use sha256 for untrusted/multi-tenant.
_HASH_ALGO = os.getenv("PACE_PREFIX_HASH", "xxhash").lower()
if _HASH_ALGO not in ("xxhash", "sha256"):
    raise ValueError(
        f"PACE_PREFIX_HASH must be 'xxhash' or 'sha256', got '{_HASH_ALGO}'"
    )


def _make_none_hash() -> bytes:
    if _HASH_ALGO == "xxhash":
        return xxhash.xxh128(b"PACE_PREFIX_CACHE_NONE").digest()
    return hashlib.sha256(b"PACE_PREFIX_CACHE_NONE").digest()


NONE_HASH = _make_none_hash()


def hash_block_tokens(parent_hash: bytes, token_ids: Tuple[int, ...]) -> bytes:
    """Compute a chained block hash from the parent hash and this block's tokens.

    The hash of block *i* depends on the hash of block *i-1* (or ``NONE_HASH``
    for the first block), so two blocks with identical tokens but different
    preceding prefixes hash differently — eliminating false-prefix matches.
    """
    payload = struct.pack(f"<{len(token_ids)}i", *token_ids)
    if _HASH_ALGO == "xxhash":
        h = xxhash.xxh128(parent_hash)
        h.update(payload)
        return h.digest()
    h = hashlib.sha256(parent_hash)
    h.update(payload)
    return h.digest()


def compute_block_hashes(token_ids: List[int], block_size: int) -> List[bytes]:
    """Compute chained hashes for all full blocks in a token sequence."""
    num_full = len(token_ids) // block_size
    hashes: List[bytes] = []
    parent = NONE_HASH
    for i in range(num_full):
        start = i * block_size
        bh = hash_block_tokens(parent, tuple(token_ids[start : start + block_size]))
        hashes.append(bh)
        parent = bh
    return hashes


@dataclass
class CachedBlock:
    """A cached KV block tracked by the prefix cache."""

    block_idx: int
    block_hash: bytes
    ref_count: int = 0
    last_access_time: float = field(default_factory=time.monotonic)


class BlockHashMap:
    """Hash-map index for prefix caching with LRU eviction.

    Cached blocks with ``ref_count == 0`` are tracked in an ``OrderedDict``
    (LRU order, oldest first). ``evict(n)`` pops the oldest entries and returns
    their physical block indices for the pool to reclaim.
    """

    def __init__(self):
        self._map: Dict[bytes, CachedBlock] = {}
        self._lru: OrderedDict[bytes, None] = OrderedDict()

        self.total_query_tokens = 0
        self.total_hit_tokens = 0
        self.total_evictions = 0
        self.total_inserts = 0
        self.total_dedup_frees = 0

    def match_prefix(self, block_hashes: List[bytes]) -> Tuple[List[int], int]:
        """Find the longest cached prefix (full blocks only).

        Returns ``(matched_block_indices, num_matched_blocks)``.
        """
        matched: List[int] = []
        for bh in block_hashes:
            entry = self._map.get(bh)
            if entry is None:
                break
            matched.append(entry.block_idx)
            entry.last_access_time = time.monotonic()
            if bh in self._lru:
                self._lru.move_to_end(bh)
        return matched, len(matched)

    def peek(self, block_hash: bytes) -> Optional[CachedBlock]:
        """Return the cached entry for a hash without updating LRU/access time."""
        return self._map.get(block_hash)

    def cached_block_indices(self) -> set:
        """Physical block indices currently held by the cache."""
        return {cb.block_idx for cb in self._map.values()}

    def record_query(self, query_tokens: int, hit_tokens: int) -> None:
        self.total_query_tokens += query_tokens
        self.total_hit_tokens += hit_tokens

    def insert(self, block_hash: bytes, block_idx: int) -> Optional[int]:
        """Register a computed block.

        If the hash already exists (dedup), returns the *new* ``block_idx`` that
        the caller should free. Otherwise returns ``None``.
        """
        existing = self._map.get(block_hash)
        if existing is not None:
            existing.last_access_time = time.monotonic()
            # Same physical block re-registered: not a dedup, ref already held.
            if existing.block_idx == block_idx:
                return None
            existing.ref_count += 1
            if block_hash in self._lru:
                self._lru.pop(block_hash)
            self.total_dedup_frees += 1
            return block_idx

        self._map[block_hash] = CachedBlock(
            block_idx=block_idx, block_hash=block_hash, ref_count=1
        )
        self.total_inserts += 1
        return None

    def increment_ref(self, block_hash: bytes) -> None:
        entry = self._map.get(block_hash)
        if entry is None:
            return
        if entry.ref_count == 0 and block_hash in self._lru:
            self._lru.pop(block_hash)
        entry.ref_count += 1
        entry.last_access_time = time.monotonic()

    def decrement_ref(self, block_hash: bytes) -> None:
        entry = self._map.get(block_hash)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        if entry.ref_count == 0:
            self._lru[block_hash] = None

    def evict(self, num_blocks: int) -> List[int]:
        """Evict up to ``num_blocks`` LRU cached blocks with ref_count == 0.

        Returns the physical block indices that can be returned to the pool.
        """
        freed: List[int] = []
        while len(freed) < num_blocks and self._lru:
            bh, _ = self._lru.popitem(last=False)
            entry = self._map.pop(bh, None)
            if entry is not None:
                freed.append(entry.block_idx)
                self.total_evictions += 1
        return freed

    @property
    def num_cached(self) -> int:
        return len(self._map)

    @property
    def num_evictable(self) -> int:
        return len(self._lru)

    def get_stats(self) -> Dict[str, object]:
        hit_rate = (
            (self.total_hit_tokens / self.total_query_tokens)
            if self.total_query_tokens > 0
            else 0.0
        )
        return {
            "prefix_cache_hash_algo": _HASH_ALGO,
            "prefix_cache_queries": self.total_query_tokens,
            "prefix_cache_hits": self.total_hit_tokens,
            "prefix_cache_hit_rate": round(hit_rate, 4),
            "prefix_cache_cached_blocks": self.num_cached,
            "prefix_cache_evictable_blocks": self.num_evictable,
            "prefix_cache_total_evictions": self.total_evictions,
            "prefix_cache_total_inserts": self.total_inserts,
            "prefix_cache_dedup_frees": self.total_dedup_frees,
        }


class PrefixCacheContextMixin:
    """Mixin for cache contexts that participate in prefix caching.

    Provides the tracking fields that ``PrefixCacheMixin.claim_prefix_blocks``
    writes to. Call ``_init_prefix_fields()`` from the context's ``__init__``.
    """

    def _init_prefix_fields(self) -> None:
        self.prefix_matched_tokens: int = 0
        self.prefix_matched_blocks: List[int] = []
        self.prefix_token_ids: Optional[List[int]] = None
        # Every block hash whose ref this context owns (claimed + inserted),
        # released exactly once in _release_prefix_refs on removal.
        self.prefix_held_hashes: List[bytes] = []


class PrefixCacheMixin:
    """Adds vLLM-style automatic prefix caching to a paged Cache backend.

    Uses a ``BlockHashMap`` (parent-chained block hashes + LRU eviction).
    Subclasses call ``_init_prefix_cache_from_config(...)`` during ``__init__``
    and implement ``_get_block_indices_for_context(context) -> list[int]``.
    Backends that own a block pool also override ``_claim_prefix_blocks_impl``
    and ``_free_duplicate_block``.
    """

    def _init_prefix_cache(self, block_size: int, enable: bool) -> None:
        self._prefix_map = None
        self._prefix_block_size = block_size
        if enable:
            self._prefix_map = BlockHashMap()

    def _init_prefix_cache_from_config(
        self,
        config: PretrainedConfig,
        backend_label: str,
        *,
        autotune_fn=None,
        **kwargs,
    ) -> int:
        """Resolve block size, read the enable flag, initialize, and log.

        ``block_size`` defaults to 16 unless an ``autotune_fn`` is supplied
        (``(num_kv_heads, head_dim) -> int``), which is used when ``block_size``
        is not explicitly provided.  Backends whose physical block size is
        autotuned (e.g. SLAB) must pass ``autotune_fn`` so the prefix hashing
        granularity matches the cache's real block size.  Prefix caching is
        enabled by default on every backend; disable it as opt-out via
        ``enable_prefix_cache=False`` or ``PACE_PREFIX_CACHE=0``.
        """
        block_size = kwargs.get("block_size", 0)
        if block_size <= 0:
            if autotune_fn is not None:
                num_kv_heads = getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                )
                head_dim = getattr(
                    config,
                    "head_dim",
                    config.hidden_size // config.num_attention_heads,
                )
                block_size = autotune_fn(num_kv_heads, head_dim)
            else:
                block_size = 16
        enable = kwargs.get(
            "enable_prefix_cache", os.getenv("PACE_PREFIX_CACHE", "1") == "1"
        )
        self._init_prefix_cache(block_size, enable)
        if self.prefix_cache_enabled:
            from pace.utils.logging import PACE_LLM_INFO

            PACE_LLM_INFO(
                f"{backend_label}: prefix caching enabled "
                f"(block_size={block_size}, hash={_HASH_ALGO})"
            )
        return block_size

    @property
    def prefix_cache_enabled(self) -> bool:
        return getattr(self, "_prefix_map", None) is not None

    def try_prefix_match(self, token_ids: list) -> Tuple[list, int]:
        """Find reusable KV blocks for a token prefix.

        Returns ``(matched_block_indices, num_matched_tokens)``. Only full-block
        matches are returned: the engine skips ``num_matched_tokens`` positions
        during prefill, and partial blocks lack valid KV for all positions.
        """
        if self._prefix_map is None:
            return [], 0
        hashes = compute_block_hashes(token_ids, self._prefix_block_size)
        matched, num_blocks = self._prefix_map.match_prefix(hashes)
        num_matched_tokens = num_blocks * self._prefix_block_size
        self._prefix_map.record_query(len(token_ids), num_matched_tokens)
        return matched, num_matched_tokens

    def claim_prefix_blocks(
        self, context, token_ids: list, matched_blocks: list, num_matched_tokens: int
    ) -> None:
        """Assign matched prefix blocks to a request and increment refs."""
        if not matched_blocks:
            return
        self._claim_prefix_blocks_impl(
            context, token_ids, matched_blocks, num_matched_tokens
        )
        if self._prefix_map is not None:
            hashes = compute_block_hashes(token_ids, self._prefix_block_size)
            claimed = hashes[: len(matched_blocks)]
            for bh in claimed:
                self._prefix_map.increment_ref(bh)
            held = getattr(context, "prefix_held_hashes", None) or []
            context.prefix_held_hashes = held + list(claimed)

        context.prefix_matched_tokens = num_matched_tokens
        context.prefix_matched_blocks = list(matched_blocks)
        context.prefix_token_ids = list(token_ids)

    def _claim_prefix_blocks_impl(
        self, context, token_ids, matched_blocks, num_matched_tokens
    ) -> None:
        """Assign physical prefix blocks to a context. Override in pool backends."""
        pass

    def insert_prefix(self, context, token_ids: list) -> int:
        """Register completed full blocks; returns the number of new entries."""
        if self._prefix_map is None:
            return 0
        blocks = self._get_block_indices_for_context(context)
        if not blocks:
            return 0
        hashes = compute_block_hashes(token_ids, self._prefix_block_size)
        return self._register_blocks(context, hashes, blocks)

    def preinsert_computed_blocks(
        self, context, token_ids: list, num_full_blocks: int
    ) -> int:
        """Reserve+register a context's full prefix blocks before the forward so
        a same-step follower can reuse them (intra-batch dedup)."""
        if self._prefix_map is None or num_full_blocks <= 0:
            return 0
        blocks = self._reserve_blocks_for_context(context, num_full_blocks)
        if not blocks:
            return 0
        hashes = compute_block_hashes(token_ids, self._prefix_block_size)
        n = min(num_full_blocks, len(hashes), len(blocks))
        return self._register_blocks(context, hashes[:n], blocks[:n]) if n > 0 else 0

    def _reserve_blocks_for_context(self, context, num_full_blocks: int) -> list:
        """Eagerly allocate a context's first ``num_full_blocks`` blocks. Pool
        backends override; others don't support intra-batch dedup."""
        return []

    def _register_blocks(self, context, hashes, blocks, start: int = 0) -> int:
        """Insert ``blocks[start:]``, pinning new blocks and recording every owned
        hash on the context; duplicates are repointed to the canonical block
        before being freed. Returns the number of new entries inserted.
        """
        inserted = 0
        newly_held: List[bytes] = []
        for i in range(start, min(len(hashes), len(blocks))):
            bh = hashes[i]
            pre = self._prefix_map.peek(bh)
            dup = self._prefix_map.insert(bh, blocks[i])
            if dup is not None:
                self._repoint_duplicate_block(context, i, dup, pre.block_idx)
                self._free_duplicate_block(dup)
                newly_held.append(bh)
            elif pre is None:
                inserted += 1
                newly_held.append(bh)
        if newly_held:
            held = getattr(context, "prefix_held_hashes", None) or []
            context.prefix_held_hashes = held + newly_held
        return inserted

    def _get_block_indices_for_context(self, context) -> list:
        raise NotImplementedError

    def _free_duplicate_block(self, block_idx: int) -> None:
        """Free a duplicate block back to the pool. Override in subclass."""
        pass

    def _repoint_duplicate_block(
        self, context, block_pos: int, old_block: int, canonical_block: int
    ) -> None:
        """Repoint a context's logical block to the canonical cached block before the duplicate is freed. No-op unless a backend owns a block table."""
        pass

    def _release_prefix_refs(self, context) -> None:
        """Release every ref this context holds (claimed + inserted) exactly once so its blocks become evictable."""
        if self._prefix_map is None:
            return
        held = getattr(context, "prefix_held_hashes", None)
        if not held:
            return
        for bh in held:
            self._prefix_map.decrement_ref(bh)
        context.prefix_held_hashes = []

    def evict_prefix_blocks(self, num_blocks: int) -> list:
        """Evict LRU cached blocks. Returns physical block indices freed."""
        if self._prefix_map is None:
            return []
        return self._prefix_map.evict(num_blocks)

    def get_prefix_cache_stats(self) -> dict:
        if self._prefix_map is None:
            return {"prefix_cache_enabled": False}
        stats = self._prefix_map.get_stats()
        stats["prefix_cache_enabled"] = True
        return stats
