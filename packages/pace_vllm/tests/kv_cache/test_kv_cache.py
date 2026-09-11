# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `PaceKVCacheSpec` and the `PaceKVCache` lifecycle."""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import torch

import pace_vllm
from pace_vllm.v1.kv_cache import PaceKVCache, PaceKVCacheSpec, PaceLayerSpec

# Construct PaceKVCache instantiates `torch.classes.pace.SlabPool`, so
# libpace_cpp.so must be loaded before any of the lifecycle tests run.
pace_vllm._load_pace_native()


class TestPaceKVCacheSpec(unittest.TestCase):
    def _spec(self, **overrides) -> PaceKVCacheSpec:
        kwargs = dict(
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            num_blocks=10,
            dtype=torch.bfloat16,
        )
        kwargs.update(overrides)
        return PaceKVCacheSpec.uniform(**kwargs)

    def test_uniform_factory_replicates_num_blocks(self) -> None:
        spec = self._spec(num_layers=4, num_blocks=10)
        self.assertEqual(spec.num_layers, 4)
        self.assertEqual(spec.per_layer_num_blocks, (10, 10, 10, 10))
        self.assertEqual(spec.total_blocks, 40)

    def test_per_layer_bytes_per_block(self) -> None:
        # 2 (K + V) * num_kv_heads * block_size * head_dim * 2 (bf16 bytes).
        spec = self._spec(num_kv_heads=8, head_dim=128, block_size=16)
        self.assertEqual(spec.per_layer[0].bytes_per_block, 2 * 8 * 16 * 128 * 2)

    def test_total_bytes_scales_with_total_blocks(self) -> None:
        spec = self._spec(num_layers=2, num_blocks=5)
        # Total bytes = sum over layers of (bytes_per_block * num_blocks).
        # For uniform spec this collapses to per_layer[0].bytes_per_block
        # * total_blocks.
        self.assertEqual(
            spec.total_bytes,
            spec.per_layer[0].bytes_per_block * spec.total_blocks,
        )

    def test_spec_is_frozen(self) -> None:
        spec = self._spec()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.per_layer = ()  # type: ignore[misc]

    def test_layer_spec_is_frozen(self) -> None:
        spec = self._spec()
        layer = spec.per_layer[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            layer.num_kv_heads = 99  # type: ignore[misc]


class TestPaceKVCacheLifecycle(unittest.TestCase):
    def _make_cache(self, num_layers: int = 2) -> PaceKVCache:
        spec = PaceKVCacheSpec.uniform(
            num_layers=num_layers,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            num_blocks=4,
            dtype=torch.bfloat16,
        )
        return PaceKVCache(spec, layer_names=[f"layer.{i}" for i in range(num_layers)])

    def test_construct_with_valid_spec(self) -> None:
        kv = self._make_cache(num_layers=3)
        self.assertEqual(kv.num_layers, 3)
        self.assertEqual(kv.layer_names, ["layer.0", "layer.1", "layer.2"])

    def test_non_bf16_dtype_rejected(self) -> None:
        spec = PaceKVCacheSpec.uniform(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=8,
            num_blocks=2,
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError) as cm:
            PaceKVCache(spec)
        self.assertIn("BF16", str(cm.exception))

    def test_layer_names_length_must_match(self) -> None:
        spec = PaceKVCacheSpec.uniform(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=8,
            num_blocks=2,
            dtype=torch.bfloat16,
        )
        with self.assertRaises(ValueError):
            PaceKVCache(spec, layer_names=["only_one"])

    def test_create_then_get_sequence_id(self) -> None:
        kv = self._make_cache()
        seq_id = kv.create_sequence("req-1", max_seq_len=128)
        self.assertEqual(kv.get_sequence_id("req-1"), seq_id)
        self.assertGreater(seq_id, 0)

    def test_create_sequence_is_idempotent_for_same_request(self) -> None:
        kv = self._make_cache()
        first = kv.create_sequence("req-1", max_seq_len=128)
        second = kv.create_sequence("req-1", max_seq_len=128)
        self.assertEqual(first, second)

    def test_remove_sequence_clears_mapping(self) -> None:
        kv = self._make_cache()
        kv.create_sequence("req-1", max_seq_len=128)
        kv.remove_sequence("req-1")
        self.assertIsNone(kv.get_sequence_id("req-1"))

    def test_remove_unknown_sequence_is_a_noop(self) -> None:
        kv = self._make_cache()
        kv.remove_sequence("never-existed")  # should not raise

    def test_layer_idx_round_trip(self) -> None:
        kv = self._make_cache(num_layers=3)
        for i, name in enumerate(("layer.0", "layer.1", "layer.2")):
            with self.subTest(name=name):
                self.assertEqual(kv.layer_idx_of(name), i)

    def test_pool_for_layer_returns_a_pool(self) -> None:
        kv = self._make_cache()
        # Use identity, not equality: SlabPool is a torch ScriptObject and
        # does not implement `__ne__`.
        self.assertIsNotNone(kv.pool_for_layer(0))
        self.assertIs(kv.pool_for_layer(0), kv.pool_for_layer(0))
        self.assertIsNot(kv.pool_for_layer(0), kv.pool_for_layer(1))

    def test_owner_round_trip(self) -> None:
        kv = self._make_cache()
        marker = object()
        kv.set_owner(marker)
        self.assertIs(kv.get_owner(), marker)
        kv.set_owner(None)
        self.assertIsNone(kv.get_owner())

    def test_prefix_helpers_reject_heterogeneous_block_sizes(self) -> None:
        spec = PaceKVCacheSpec(
            per_layer=(
                PaceLayerSpec(2, 16, 32, torch.bfloat16, 4),
                PaceLayerSpec(2, 16, 16, torch.bfloat16, 4),
            )
        )
        kv = PaceKVCache(spec, layer_names=["layer.0", "layer.1"])
        seq_id = kv.create_sequence("req-1", max_seq_len=64)

        with self.assertRaisesRegex(ValueError, "uniform block_size"):
            kv.get_block_indices(seq_id)
        with self.assertRaisesRegex(ValueError, "uniform block_size"):
            kv.assign_prefix_blocks(seq_id, [0], 16)
        with self.assertRaisesRegex(ValueError, "uniform block_size"):
            kv.retain_blocks([0])
        with self.assertRaisesRegex(ValueError, "uniform block_size"):
            kv.release_retained_blocks([0])

        for layer_idx in range(kv.num_layers):
            self.assertEqual(
                kv.pool_for_layer(layer_idx).get_sequence_length(seq_id), 0
            )


class TestFromKVCacheConfigKVSharing(unittest.TestCase):
    """`from_kv_cache_config` must drop follower layers from per-pool
    accounting so the budget splits across unique target layers only.
    Chained sharing and unknown targets must raise up-front."""

    # ~2 GiB so num_blocks resolves to a positive value for these shapes.
    _BUDGET_BYTES = 2 * 1024**3

    def _build_config(self, layer_names: list[str]):
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )

        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=None,
            attention_chunk_size=None,
        )
        group = KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)
        kv_cache_config = KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[group]
        )
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=self._BUDGET_BYTES)
        )
        return kv_cache_config, vllm_config

    def test_followers_excluded_from_unique_pool_count(self) -> None:
        cfg, vllm_cfg = self._build_config(["layer_0", "layer_1", "layer_2"])
        kv = PaceKVCache.from_kv_cache_config(
            cfg, vllm_cfg, shared_kv_cache_layers={"layer_2": "layer_0"}
        )
        self.assertEqual(kv.layer_names, ["layer_0", "layer_1"])
        self.assertEqual(kv.num_layers, 2)

    def test_budget_splits_across_unique_pools_not_followers(self) -> None:
        # Build the same cfg twice and compare per-layer block counts.
        # With 3 layers no-sharing -> floor(B / page / 3); with 1 follower
        # -> floor(B / page / 2). The latter must be the larger value.
        cfg_no_share, vllm_no_share = self._build_config(
            ["layer_0", "layer_1", "layer_2"]
        )
        kv_no_share = PaceKVCache.from_kv_cache_config(cfg_no_share, vllm_no_share)

        cfg_share, vllm_share = self._build_config(["layer_0", "layer_1", "layer_2"])
        kv_share = PaceKVCache.from_kv_cache_config(
            cfg_share, vllm_share, shared_kv_cache_layers={"layer_2": "layer_0"}
        )
        # Each unique pool gets a strictly larger share when a follower is
        # excluded from the divisor.
        self.assertGreater(
            kv_share.spec.per_layer_num_blocks[0],
            kv_no_share.spec.per_layer_num_blocks[0],
        )

    def test_chained_sharing_raises(self) -> None:
        cfg, vllm_cfg = self._build_config(["layer_0", "layer_1", "layer_2"])
        with self.assertRaises(ValueError) as cm:
            PaceKVCache.from_kv_cache_config(
                cfg,
                vllm_cfg,
                shared_kv_cache_layers={
                    "layer_2": "layer_1",
                    "layer_1": "layer_0",
                },
            )
        self.assertIn("chained KV sharing", str(cm.exception))

    def test_unknown_target_raises(self) -> None:
        cfg, vllm_cfg = self._build_config(["layer_0", "layer_1", "layer_2"])
        with self.assertRaises(ValueError) as cm:
            PaceKVCache.from_kv_cache_config(
                cfg,
                vllm_cfg,
                shared_kv_cache_layers={"layer_2": "no_such_layer"},
            )
        self.assertIn("unknown target", str(cm.exception))

    def test_no_sharing_preserves_all_layers(self) -> None:
        # Default `shared_kv_cache_layers=None` must behave exactly as
        # the pre-fix factory: every attention layer gets its own pool.
        cfg, vllm_cfg = self._build_config(["layer_0", "layer_1", "layer_2"])
        kv = PaceKVCache.from_kv_cache_config(cfg, vllm_cfg)
        self.assertEqual(kv.layer_names, ["layer_0", "layer_1", "layer_2"])

    def test_budget_smaller_than_one_block_raises(self) -> None:
        # Honor VLLM_CPU_KVCACHE_SPACE strictly: if the budget can't cover
        # one block across all layers, raise instead of silently allocating
        # one block (which would exceed the user's explicit cap).
        cfg, _ = self._build_config(["layer_0", "layer_1", "layer_2"])
        tiny_vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=1024)
        )
        with self.assertRaisesRegex(ValueError, "smaller than one block"):
            PaceKVCache.from_kv_cache_config(cfg, tiny_vllm_cfg)


if __name__ == "__main__":
    unittest.main()
