# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for KV sharing: a layer whose spec sets ``kv_sharing_target`` reuses an
earlier layer's KV. Such a layer owns no k/v weights and passes K=V=None to the
attention backend, so no dummy tensor is allocated and no cache write happens.
Uses Gemma4 (the model that needs this) over a tiny config.

The model-level tests cover the contiguous path; the backend-level tests below
pin the same None contract on the paged and slab backends, which handle it in
their own forward().
"""

import unittest

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401

from pace.llm.attention import KVCacheManager, KVCacheType
from pace.llm.attention.base import AttentionBackendType
from pace.llm.attention.paged.backend import PagedAttentionBackend
from pace.llm.attention.paged.utils import (
    build_paged_attention_metadata,
    create_paged_kv_cache_manager,
)
from pace.llm.attention.slab.backend import SlabAttentionBackend
from pace.llm.attention.slab.cache import SlabPoolLayerView, SlabPoolManager
from pace.llm.attention.specs import compute_layer_kv_specs
from pace.llm.configs import BackendType, LLMOperatorType, OperatorConfig
from pace.llm.models.gemma4 import Gemma4ForCausalLM

try:
    from transformers.models.gemma4 import Gemma4TextConfig

    transformers_gemma4 = Gemma4TextConfig
except ImportError:  # pragma: no cover - depends on transformers version
    transformers_gemma4 = None


def _config():
    """4 layers: sliding/global alternating, trailing 2 layers share KV."""
    return transformers_gemma4(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=128,
        sliding_window=16,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        max_position_embeddings=256,
        rope_parameters={
            "full_attention": {"rope_type": "proportional", "rope_theta": 10000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )


def _opconfig(attn=AttentionBackendType.NATIVE):
    opconfig = OperatorConfig()
    for op in LLMOperatorType:
        opconfig[op] = attn if op == LLMOperatorType.Attention else BackendType.NATIVE
    opconfig.finalize()
    return opconfig


@unittest.skipIf(transformers_gemma4 is None, "transformers lacks Gemma4")
class TestGemma4KVSharing(TestCase):
    def setUp(self):
        self.config = _config()
        self.specs = compute_layer_kv_specs(self.config)

    def test_specs_mark_trailing_layers_as_shared(self):
        self.assertEqual([s.is_shared for s in self.specs], [False, False, True, True])
        self.assertEqual([s.kv_sharing_target for s in self.specs], [None, None, 0, 1])

    def test_shared_layers_own_no_kv_projections(self):
        model = Gemma4ForCausalLM(self.config, _opconfig()).to(torch.bfloat16).eval()
        for idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            expected = idx < 2
            self.assertEqual(hasattr(attn, "k_proj"), expected, f"layer {idx} k_proj")
            self.assertEqual(hasattr(attn, "v_proj"), expected, f"layer {idx} v_proj")
            self.assertEqual(attn.is_kv_shared_layer, not expected)

    def test_shared_layers_pass_none_kv_to_the_backend(self):
        """No dummy K/V is built for a layer that stores nothing."""
        model = Gemma4ForCausalLM(self.config, _opconfig()).to(torch.bfloat16).eval()

        got_none = {}
        for idx, layer in enumerate(model.model.layers):
            attn_mod = layer.self_attn.attn
            original = attn_mod.forward

            def spy(Q, K, V, kv_cache, positions, _i=idx, _f=original, **kwargs):
                got_none[_i] = (K is None, V is None)
                return _f(Q, K, V, kv_cache, positions, **kwargs)

            attn_mod.forward = spy

        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        cache = KVCacheManager(
            self.config, max_seq_length=32, cache_type=KVCacheType.DYNAMIC
        )
        with torch.no_grad():
            model(input_ids=ids, positions=positions, kv_cache=cache)

        self.assertEqual(
            got_none,
            {0: (False, False), 1: (False, False), 2: (True, True), 3: (True, True)},
        )

    def test_shared_layer_cache_aliases_its_target_and_is_not_written(self):
        cache = KVCacheManager(
            self.config, max_seq_length=32, cache_type=KVCacheType.DYNAMIC
        )
        shared_view = cache.cache_objects[2]
        target = cache.cache_objects[0]

        k = torch.randn(1, 2, 5, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 5, 64, dtype=torch.bfloat16)
        target.update_cache(k, v, concat_dim=2)
        self.assertEqual(target.seq_len, 5)

        # Reading through the shared view yields the target's KV ...
        got_k, got_v = shared_view.update_cache(None, None, concat_dim=2)
        self.assertEqual(got_k, target.key)
        self.assertEqual(got_v, target.value)
        # ... and the write attempt changed nothing.
        self.assertEqual(target.seq_len, 5)
        self.assertEqual(shared_view.seq_len, 5)


Q_LEN = 6


@unittest.skipIf(transformers_gemma4 is None, "transformers lacks Gemma4")
class TestSlabBackendSharedKV(TestCase):
    """The slab backend must accept K=V=None without touching the pool."""

    def setUp(self):
        self.config = _config()
        self.specs = compute_layer_kv_specs(self.config)
        self.num_heads = self.config.num_attention_heads
        self.manager = SlabPoolManager(self.config, max_total_tokens=256)
        self.token = self.manager.create_sequence(token="seq-0")

    def test_none_kv_attends_the_target_pool_without_writing(self):
        spec = self.specs[2]
        self.assertEqual(spec.kv_sharing_target, 0)

        target = SlabPoolLayerView(self.manager, [self.token], layer_idx=0)
        shape = (1, Q_LEN, spec.num_kv_heads, spec.head_dim)
        target.update(
            torch.randn(*shape, dtype=torch.bfloat16),
            torch.randn(*shape, dtype=torch.bfloat16),
        )
        self.assertEqual(self.manager.get_sequence_length(self.token), Q_LEN)

        backend = SlabAttentionBackend(
            self.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            None,
            sliding_window=spec.sliding_window,
        )
        shared = SlabPoolLayerView(self.manager, [self.token], layer_idx=2)
        q = torch.randn(1, Q_LEN, self.num_heads, spec.head_dim, dtype=torch.bfloat16)
        out = backend(q, None, None, shared, torch.arange(Q_LEN).unsqueeze(0))

        self.assertEqual(out.shape, (1, Q_LEN, self.num_heads, spec.head_dim))
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(self.manager.get_sequence_length(self.token), Q_LEN)

    def test_only_one_of_k_and_v_being_none_is_rejected(self):
        spec = self.specs[2]
        backend = SlabAttentionBackend(
            self.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            None,
            sliding_window=spec.sliding_window,
        )
        view = SlabPoolLayerView(self.manager, [self.token], layer_idx=2)
        q = torch.randn(1, Q_LEN, self.num_heads, spec.head_dim, dtype=torch.bfloat16)
        kv = torch.randn(
            1, Q_LEN, spec.num_kv_heads, spec.head_dim, dtype=torch.bfloat16
        )
        positions = torch.arange(Q_LEN).unsqueeze(0)

        for K, V in ((kv, None), (None, kv)):
            with self.subTest(K_is_none=K is None):
                with self.assertRaises(AssertionError):
                    backend(q, K, V, view, positions)


@unittest.skipIf(transformers_gemma4 is None, "transformers lacks Gemma4")
class TestPagedBackendSharedKV(TestCase):
    """The paged backend must accept K=V=None and skip the block write."""

    BLOCK_SIZE = 16

    def setUp(self):
        self.config = _config()
        self.specs = compute_layer_kv_specs(self.config)
        self.num_heads = self.config.num_attention_heads
        self.manager = create_paged_kv_cache_manager(
            self.config,
            max_seq_length=32,
            block_size=self.BLOCK_SIZE,
            batch_size=1,
        )
        self.meta = build_paged_attention_metadata(
            self.manager,
            self.config,
            torch.tensor([Q_LEN]),
            torch.bfloat16,
            self.BLOCK_SIZE,
        )

    def test_none_kv_leaves_the_target_blocks_unchanged(self):
        spec = self.specs[2]
        self.assertEqual(spec.kv_sharing_target, 0)

        target = self.manager.cache_objects[0]
        shared = self.manager.cache_objects[2]
        self.assertTrue(getattr(shared, "kv_sharing_readonly", False))

        backend = PagedAttentionBackend(
            self.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            sliding_window=spec.sliding_window,
        )
        q = torch.randn(1, Q_LEN, self.num_heads, spec.head_dim, dtype=torch.bfloat16)
        kv_shape = (1, Q_LEN, spec.num_kv_heads, spec.head_dim)
        backend(
            q,
            torch.randn(*kv_shape, dtype=torch.bfloat16),
            torch.randn(*kv_shape, dtype=torch.bfloat16),
            target,
            None,
            paged_attn_metadata=self.meta,
        )

        before = [t.clone() for t in target.get_cache_tensors()]
        out = backend(q, None, None, shared, None, paged_attn_metadata=self.meta)

        self.assertEqual(out.shape, (1, Q_LEN, self.num_heads, spec.head_dim))
        for expected, actual in zip(before, target.get_cache_tensors()):
            self.assertEqual(actual, expected)

    def test_none_kv_is_tolerated_on_a_writable_cache(self):
        """None must be handled on its own, not only via kv_sharing_readonly."""
        spec = self.specs[0]
        target = self.manager.cache_objects[0]
        self.assertFalse(getattr(target, "kv_sharing_readonly", False))

        backend = PagedAttentionBackend(
            self.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            sliding_window=spec.sliding_window,
        )
        q = torch.randn(1, Q_LEN, self.num_heads, spec.head_dim, dtype=torch.bfloat16)
        before = [t.clone() for t in target.get_cache_tensors()]
        backend(q, None, None, target, None, paged_attn_metadata=self.meta)

        for expected, actual in zip(before, target.get_cache_tensors()):
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
