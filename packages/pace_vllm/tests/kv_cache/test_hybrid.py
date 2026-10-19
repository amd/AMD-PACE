# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for hybrid-model and MLA support.

Hybrid models (Qwen3.5-MoE, Qwen3-Next, Jamba) ship Mamba groups alongside
attention groups in `KVCacheConfig`. SlabPool only services attention, so
`PaceKVCache.from_kv_cache_config` filters non-AttentionSpec layers and
`PaceModelRunner` leaves their storage to vLLM -- shipping a zero-sized
placeholder for a Mamba layer crashes the first forward.

MLA (`MLAAttentionSpec`) is a subclass of `FullAttentionSpec`, so the existing
`isinstance(layer_spec, AttentionSpec)` filter accepts it silently. SlabPool's
K/V geometry can't service MLA's projection layout; we want a loud
`NotImplementedError` instead of wrong outputs.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

import pace_vllm
from pace_vllm.v1 import kv_cache as kv_cache_module
from pace_vllm.v1.connector import PaceSlabConnector
from pace_vllm.v1.kv_cache import PaceKVCache, _pace_kvcache_budget_bytes
from pace_vllm.v1.worker.cpu_model_runner import PaceModelRunner

pace_vllm._load_pace_native()


# Budget large enough that num_blocks resolves positive for the small
# fixture geometries used here.
_BUDGET_BYTES = 2 * 1024**3


def _vllm_config(connector_block_size: int | None = None) -> SimpleNamespace:
    kv_transfer_config = None
    if connector_block_size is not None:
        kv_transfer_config = SimpleNamespace(
            kv_connector="PaceSlabConnector",
            kv_connector_extra_config={"slab_block_size": connector_block_size},
        )
    return SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES),
        kv_transfer_config=kv_transfer_config,
    )


def _attention_spec(num_kv_heads: int = 8, head_size: int = 128, block_size: int = 16):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )


def _mla_spec():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    # Standard DeepSeek-V2 shapes: kv_lora_rank=512, qk_rope_head_dim=64
    # -> head_size = 576. The exact numbers don't matter for our reject
    # path; just need the type to be MLAAttentionSpec.
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )


def _mamba_spec():
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        block_size=16,
        shapes=((128, 64),),
        dtypes=(torch.bfloat16,),
        mamba_type="mamba2",
        mamba_cache_mode="mamba2",
        num_speculative_blocks=0,
    )


def _kv_cache_config(*groups, tensors=None, num_blocks=0):
    from vllm.v1.kv_cache_interface import KVCacheConfig

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=list(tensors or []),
        kv_cache_groups=list(groups),
    )


def _kv_tensor(size, shared_by):
    from vllm.v1.kv_cache_interface import KVCacheTensor

    if "shared_by" in KVCacheTensor.__dataclass_fields__:
        return KVCacheTensor(size=size, shared_by=list(shared_by))
    return KVCacheTensor(
        size=size,
        layers=list(shared_by),
        layer_stride=size,
        block_stride=size,
    )


def _group(layer_names, spec):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    return KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)


def _uniform_group(name_to_spec):
    """Wrap a dict of `name -> spec` in `UniformTypeKVCacheSpecs` so we
    can exercise the MLA-in-uniform branch."""
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        UniformTypeKVCacheSpecs,
    )

    uniform = UniformTypeKVCacheSpecs(
        block_size=next(iter(name_to_spec.values())).block_size,
        kv_cache_specs=dict(name_to_spec),
    )
    return KVCacheGroupSpec(
        layer_names=list(name_to_spec.keys()), kv_cache_spec=uniform
    )


def _unified_allocator_runner(model_type: str, shared_layers=None) -> PaceModelRunner:
    runner = PaceModelRunner.__new__(PaceModelRunner)
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(get_resolved_kv_cache_layout=lambda: None)
    runner.shared_kv_cache_layers = dict(shared_layers or {})
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type)
    )
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_caches = []
    return runner


class TestMambaHybridAllocation(unittest.TestCase):
    """PaceModelRunner leaves Mamba storage to vLLM while slab attention uses
    zero-sized placeholders."""

    @unittest.skipUnless(
        hasattr(CPUModelRunner, "_allocate_kv_cache_tensors"),
        "raw allocation hook unavailable",
    )
    def test_allocate_raw_tensors_only_materializes_mamba_storage(self) -> None:
        spec = _mamba_spec()
        size = 2 * spec.page_size_bytes
        cfg = _kv_cache_config(
            _group(["attn.0"], _attention_spec()),
            _group(["ssm.0"], spec),
            tensors=[_kv_tensor(size, ["attn.0", "ssm.0"])],
            num_blocks=2,
        )
        runner = PaceModelRunner.__new__(PaceModelRunner)
        runner.device = torch.device("cpu")
        runner.runner_only_attn_layers = set()

        raw_tensors = PaceModelRunner._allocate_kv_cache_tensors(runner, cfg)

        self.assertEqual(raw_tensors["attn.0"].numel(), 0)
        self.assertEqual(raw_tensors["ssm.0"].numel(), size)

    @unittest.skipIf(
        hasattr(CPUModelRunner, "_allocate_kv_cache_tensors"),
        "unified allocation hook unavailable",
    )
    def test_initialize_tensors_only_keeps_mamba_storage(self) -> None:
        spec = _mamba_spec()
        size = 2 * spec.page_size_bytes
        cfg = _kv_cache_config(
            _group(["attn.0"], _attention_spec()),
            _group(["ssm.0"], spec),
            tensors=[_kv_tensor(size, ["attn.0", "ssm.0"])],
            num_blocks=2,
        )
        runner = _unified_allocator_runner("qwen3_5_moe")
        mamba_tensor = torch.ones(size, dtype=torch.int8)

        with (
            patch(
                "vllm.v1.worker.utils.allocate_kv_cache",
                return_value={
                    "attn.0": torch.ones(size, dtype=torch.int8),
                    "ssm.0": mamba_tensor,
                },
            ) as mock_allocate,
            patch("vllm.v1.worker.utils.bind_kv_cache") as mock_bind,
        ):
            tensors = runner.initialize_kv_cache_tensors(cfg, [16, 16])

        mock_allocate.assert_called_once()
        mock_bind.assert_called_once()
        self.assertEqual(tensors["attn.0"].numel(), 0)
        self.assertIs(tensors["ssm.0"], mamba_tensor)

    @unittest.skipIf(
        hasattr(CPUModelRunner, "_allocate_kv_cache_tensors"),
        "unified allocation hook unavailable",
    )
    def test_shared_attention_follower_does_not_allocate_storage(self) -> None:
        group = _uniform_group({"attn.0": _attention_spec()})
        group.layer_names.append("attn.shared")
        cfg = _kv_cache_config(group, num_blocks=2)
        runner = _unified_allocator_runner("gemma4", {"attn.shared": "attn.0"})

        with (
            patch("vllm.v1.worker.utils.allocate_kv_cache") as mock_allocate,
            patch("vllm.v1.worker.utils.bind_kv_cache"),
        ):
            tensors = runner.initialize_kv_cache_tensors(cfg, [16])

        mock_allocate.assert_not_called()
        self.assertEqual(tensors["attn.0"].numel(), 0)
        self.assertIs(tensors["attn.shared"], tensors["attn.0"])

    def test_from_kv_cache_config_skips_mamba_group(self) -> None:
        cfg = _kv_cache_config(
            _group(["attn.0", "attn.1"], _attention_spec()),
            _group(["ssm.0"], _mamba_spec()),
        )
        kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.layer_names, ["attn.0", "attn.1"])

    def test_hybrid_connector_fails_before_partial_state_restoration(self) -> None:
        cfg = _kv_cache_config(
            _group(["attn.0"], _attention_spec()),
            _group(["ssm.0"], _mamba_spec()),
        )
        with self.assertRaisesRegex(ValueError, "hybrid Mamba/SSM"):
            PaceSlabConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(
                        kv_connector_extra_config={"slab_block_size": 16}
                    )
                ),
                KVConnectorRole.SCHEDULER,
                cfg,
            )


class TestMLARejected(unittest.TestCase):
    """Item 2: MLAAttentionSpec must raise NotImplementedError before
    SlabPool gets constructed against the wrong K/V geometry."""

    def test_from_kv_cache_config_rejects_mla_bare(self) -> None:
        cfg = _kv_cache_config(_group(["attn.0", "attn.1"], _mla_spec()))
        with self.assertRaises(NotImplementedError) as cm:
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertIn("MLA", str(cm.exception))

    def test_from_kv_cache_config_rejects_mla_in_uniform(self) -> None:
        cfg = _kv_cache_config(
            _uniform_group({"attn.0": _mla_spec(), "attn.1": _mla_spec()})
        )
        with self.assertRaises(NotImplementedError) as cm:
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertIn("MLA", str(cm.exception))


class TestHeterogeneousGeometry(unittest.TestCase):
    """Item 3: PaceKVCache must accept layers with different
    `(num_kv_heads, head_dim)` and build one independently-sized
    SlabPool per layer. Sizing keeps `num_blocks` equal across layers
    using `num_blocks = budget // sum(page_size_bytes)`."""

    def test_heterogeneous_geometry_accepted(self) -> None:
        # Two layers with different num_kv_heads (8 vs 4); both bf16.
        cfg = _kv_cache_config(
            _group(["attn.wide"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["attn.narrow"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        # Force-deterministic block_size so sizing math is checkable.
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.layer_names, ["attn.wide", "attn.narrow"])
        self.assertEqual(kv.spec.per_layer[0].num_kv_heads, 8)
        self.assertEqual(kv.spec.per_layer[1].num_kv_heads, 4)
        # Both layers got SlabPool instances (distinct objects).
        self.assertIsNot(kv.pool_for_layer(0), kv.pool_for_layer(1))

    def test_heterogeneous_sizing_matches_formula(self) -> None:
        # 2 layers, geometry (num_kv_heads=8, head_dim=128) and
        # (num_kv_heads=4, head_dim=128); env-pinned block_size=64.
        # bf16 -> 2 bytes/elem; K+V -> *2.
        # page0 = 2 * 8 * 128 * 64 * 2 = 262144
        # page1 = 2 * 4 * 128 * 64 * 2 = 131072
        # sum = 393216
        # num_blocks = floor(2 GiB / 393216) = 5461
        cfg = _kv_cache_config(
            _group(["attn.wide"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["attn.narrow"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        page0 = 2 * 8 * 128 * 64 * 2
        page1 = 2 * 4 * 128 * 64 * 2
        expected_num_blocks = _BUDGET_BYTES // (page0 + page1)
        self.assertEqual(kv.spec.per_layer[0].num_blocks, expected_num_blocks)
        self.assertEqual(kv.spec.per_layer[1].num_blocks, expected_num_blocks)
        self.assertLessEqual(kv.spec.total_bytes, _BUDGET_BYTES)

    def test_prefix_connector_uses_uniform_block_size(self) -> None:
        cfg = _kv_cache_config(
            _group(["attn.wide"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["attn.narrow"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size"
        ) as mock_autotune:
            kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config(64))

        mock_autotune.assert_not_called()
        self.assertEqual(
            [layer.block_size for layer in kv.spec.per_layer],
            [64, 64],
        )


class TestSlabBlockSizeAutotune(unittest.TestCase):
    """Item 4: `block_size` per layer must come from the C++ L2
    autotuner (or `PACE_VLLM_SLAB_BLOCK_SIZE` override), NOT from
    vLLM's `AttentionSpec.block_size`."""

    def test_autotune_overrides_vllm_block_size(self) -> None:
        # vLLM's spec.block_size is 16; the autotuner says 128. The
        # SlabPool must be sized with 128.
        cfg = _kv_cache_config(
            _group(["attn.0"], _attention_spec(block_size=16)),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size", return_value=128
        ):
            with patch.dict(os.environ, {}, clear=False):
                # Make sure the env override is unset for this test.
                os.environ.pop("PACE_VLLM_SLAB_BLOCK_SIZE", None)
                kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.spec.per_layer[0].block_size, 128)

    def test_env_override_skips_autotune(self) -> None:
        # When the env is set, autotune must NOT be called.
        cfg = _kv_cache_config(
            _group(["attn.0", "attn.1"], _attention_spec()),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size"
        ) as mock_autotune:
            with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
                kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        mock_autotune.assert_not_called()
        for layer in kv.spec.per_layer:
            self.assertEqual(layer.block_size, 64)

    def test_autotune_per_unique_geometry_only(self) -> None:
        # Three layers, only two distinct (num_kv_heads, head_dim)
        # tuples -> autotune called exactly twice.
        cfg = _kv_cache_config(
            _group(["a"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["b"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["c"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size", return_value=64
        ) as mock_autotune:
            os.environ.pop("PACE_VLLM_SLAB_BLOCK_SIZE", None)
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        # Two unique (num_kv_heads, head_dim) pairs -> two calls.
        self.assertEqual(mock_autotune.call_count, 2)
        called_keys = {tuple(call.args) for call in mock_autotune.call_args_list}
        self.assertEqual(called_keys, {(8, 128), (4, 128)})


class TestKvCacheBudgetReader(unittest.TestCase):
    """`_pace_kvcache_budget_bytes` is a thin reader for
    `cache_config.kv_cache_memory_bytes`. It returns the int when set
    positive, None otherwise. The actual auto-budget computation lives
    in `PaceWorker.determine_available_memory`, which writes the
    auto-computed value back onto `cache_config` so this reader sees it
    regardless of whether the user set the env var / CLI flag."""

    def test_positive_value_is_returned(self) -> None:
        cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES)
        )
        self.assertEqual(_pace_kvcache_budget_bytes(cfg), _BUDGET_BYTES)

    def test_unset_returns_none(self) -> None:
        cfg = SimpleNamespace(cache_config=SimpleNamespace())
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_zero_budget_returns_none(self) -> None:
        cfg = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=0))
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_no_cache_config_returns_none(self) -> None:
        cfg = SimpleNamespace()
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_factory_succeeds_with_kv_cache_memory_bytes(self) -> None:
        # End-to-end: from_kv_cache_config reaches the sizing path when
        # the budget field is populated (whether by user or by
        # PaceWorker.determine_available_memory mirror-write).
        cfg = _kv_cache_config(_group(["attn.0"], _attention_spec()))
        vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES)
        )
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, vllm_cfg)
        self.assertEqual(kv.layer_names, ["attn.0"])


if __name__ == "__main__":
    unittest.main()
