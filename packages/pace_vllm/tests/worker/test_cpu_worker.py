# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `PaceWorker.determine_available_memory` and `_split_kv_budget`.

`determine_available_memory` splits vLLM's CPU KV cache budget between
vLLM's own block pool and the slab. Pure-attention models leave both at the
full budget (vLLM allocates no pool, so no double-count); hybrid models
split so the vLLM Mamba pool + slab sum to the budget.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import pace_vllm

pace_vllm._load_pace_native()

from pace_vllm.v1.worker.cpu_worker import PaceWorker  # noqa: E402

_GIB = 1024**3


def _stub_worker(
    kv_cache_memory_bytes: "int | None" = None,
    kv_cache_spec: "dict | None" = None,
) -> PaceWorker:
    """Build a barely-initialised PaceWorker via __new__ so the test
    avoids CPUWorker's NUMA / OMP / model-load init chain. Only the
    fields the override touches are populated. `model_runner.get_kv_cache_spec`
    returns `kv_cache_spec` (empty by default -> pure-attention split path)."""
    worker = PaceWorker.__new__(PaceWorker)
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    )
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: dict(kv_cache_spec or {})
    )
    return worker


def _attn_group(layer_names):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )
    return KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)


def _mamba_group(layer_names):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

    spec = MambaSpec(
        block_size=16,
        shapes=((128, 64),),
        dtypes=(torch.bfloat16,),
        mamba_type="mamba2",
        mamba_cache_mode="mamba2",
        num_speculative_blocks=0,
    )
    return KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)


class TestPaceWorkerDetermineAvailableMemory(unittest.TestCase):
    def test_pure_attention_keeps_full_budget(self) -> None:
        worker = _stub_worker()
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=8 * _GIB,
        ):
            ret = worker.determine_available_memory()
        self.assertEqual(ret, 8 * _GIB)
        self.assertEqual(
            worker.vllm_config.cache_config.kv_cache_memory_bytes, 8 * _GIB
        )

    def test_explicit_budget_is_passed_through(self) -> None:
        worker = _stub_worker(kv_cache_memory_bytes=4 * _GIB)
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=4 * _GIB,
        ):
            ret = worker.determine_available_memory()
        self.assertEqual(ret, 4 * _GIB)
        self.assertEqual(
            worker.vllm_config.cache_config.kv_cache_memory_bytes, 4 * _GIB
        )

    def test_non_positive_budget_raises(self) -> None:
        worker = _stub_worker()
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=0,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-positive"):
                worker.determine_available_memory()


class TestPaceWorkerSplitKvBudget(unittest.TestCase):
    """`_split_kv_budget` reverse-engineers the vLLM/slab split from the
    KV cache groups so the two shares never double-book memory."""

    def test_pure_attention_no_mamba_returns_full_both(self) -> None:
        worker = _stub_worker(kv_cache_spec={"a": object()})
        groups = [_attn_group(["a0", "a1"])]
        with patch(
            "vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups
        ):
            vllm_b, slab_b = worker._split_kv_budget(10 * _GIB)
        self.assertEqual((vllm_b, slab_b), (10 * _GIB, 10 * _GIB))

    def test_pure_mamba_no_attention_gives_all_to_vllm(self) -> None:
        worker = _stub_worker(kv_cache_spec={"m": object()})
        groups = [_mamba_group(["m0", "m1"])]
        with patch(
            "vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups
        ):
            vllm_b, slab_b = worker._split_kv_budget(10 * _GIB)
        self.assertEqual((vllm_b, slab_b), (10 * _GIB, 0))

    def test_hybrid_split_attn_equals_group_size_is_half(self) -> None:
        worker = _stub_worker(kv_cache_spec={"x": object()})
        groups = [
            _mamba_group([f"m{i}" for i in range(10)]),
            _attn_group([f"a{i}" for i in range(10)]),
        ]
        budget = 40 * _GIB
        with patch(
            "vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups
        ):
            vllm_b, slab_b = worker._split_kv_budget(budget)
        self.assertEqual(vllm_b, budget * 10 // 20)
        self.assertEqual(slab_b, budget - budget * 10 // 20)
        self.assertEqual(vllm_b + slab_b, budget)

    def test_hybrid_split_ratio_follows_attn_over_group_size(self) -> None:
        worker = _stub_worker(kv_cache_spec={"x": object()})
        groups = [
            _mamba_group([f"m{i}" for i in range(10)]),
            _attn_group([f"a{i}" for i in range(10)]),
            _attn_group([f"b{i}" for i in range(10)]),
        ]
        budget = 30 * _GIB
        with patch(
            "vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups
        ):
            vllm_b, slab_b = worker._split_kv_budget(budget)
        self.assertEqual(vllm_b, budget * 10 // 30)
        self.assertEqual(vllm_b + slab_b, budget)

    def test_empty_spec_returns_full_both(self) -> None:
        worker = _stub_worker()
        vllm_b, slab_b = worker._split_kv_budget(7 * _GIB)
        self.assertEqual((vllm_b, slab_b), (7 * _GIB, 7 * _GIB))


if __name__ == "__main__":
    unittest.main()
