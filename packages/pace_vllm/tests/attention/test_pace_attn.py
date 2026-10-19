# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `PaceAttentionBackend` static configuration, `PaceAttentionImpl`
constructor validation, and `PaceAttentionMetadata` dataclass shape."""

from __future__ import annotations

import unittest

import torch
from vllm.v1.attention.backend import AttentionType

from pace_vllm.v1.attention.backends.pace_attn import (
    PaceAttentionBackend,
    PaceAttentionImpl,
    PaceAttentionMetadata,
    PaceAttentionMetadataBuilder,
)


class TestPaceAttentionBackend(unittest.TestCase):
    def test_get_name_shadows_cpu_attn(self) -> None:
        # Lets vLLM's `AttentionBackendEnum[...]` lookup resolve without
        # adding a new PACE_SLAB enum value upstream.
        self.assertEqual(PaceAttentionBackend.get_name(), "CPU_ATTN")

    def test_get_kv_cache_shape_is_zero_byte(self) -> None:
        # Returning (0,) makes vLLM ship empty placeholders; real K/V lives
        # in the SlabPool owned by PaceKVCache.
        self.assertEqual(PaceAttentionBackend.get_kv_cache_shape(0, 0, 0, 0), (0,))

    def test_placeholder_block_dimension_is_zero(self) -> None:
        self.assertEqual(
            PaceAttentionBackend.get_kv_cache_block_dim(16, 8, 128),
            0,
        )

    def test_does_not_accept_output_buffer(self) -> None:
        self.assertFalse(PaceAttentionBackend.accept_output_buffer)

    def test_supported_dtypes_is_bf16(self) -> None:
        self.assertEqual(PaceAttentionBackend.supported_dtypes, [torch.bfloat16])

    def test_supports_decoder_only(self) -> None:
        self.assertTrue(PaceAttentionBackend.supports_attn_type(AttentionType.DECODER))
        # Encoder / cross-attention not in scope.
        self.assertFalse(PaceAttentionBackend.supports_attn_type(AttentionType.ENCODER))

    def test_supports_sink(self) -> None:
        self.assertTrue(PaceAttentionBackend.supports_sink())

    def test_supports_sliding_window(self) -> None:
        self.assertTrue(PaceAttentionBackend.supports_sliding_window())

    def test_get_supported_head_sizes(self) -> None:
        sizes = PaceAttentionBackend.get_supported_head_sizes()
        self.assertIn(64, sizes)
        self.assertIn(128, sizes)
        self.assertIn(256, sizes)

    def test_get_impl_cls(self) -> None:
        self.assertIs(PaceAttentionBackend.get_impl_cls(), PaceAttentionImpl)

    def test_get_builder_cls(self) -> None:
        self.assertIs(
            PaceAttentionBackend.get_builder_cls(), PaceAttentionMetadataBuilder
        )

    def test_use_cascade_attention_disabled(self) -> None:
        self.assertFalse(PaceAttentionBackend.use_cascade_attention())


class TestPaceAttentionImplInit(unittest.TestCase):
    def _make(self, **overrides) -> PaceAttentionImpl:
        kwargs = dict(num_heads=8, head_size=128, scale=0.1)
        kwargs.update(overrides)
        return PaceAttentionImpl(**kwargs)

    def test_num_kv_heads_defaults_to_num_heads(self) -> None:
        impl = self._make()
        self.assertEqual(impl.num_kv_heads, 8)
        self.assertEqual(impl.num_queries_per_kv, 1)

    def test_explicit_num_kv_heads_for_gqa(self) -> None:
        impl = self._make(num_kv_heads=2)
        self.assertEqual(impl.num_kv_heads, 2)
        self.assertEqual(impl.num_queries_per_kv, 4)

    def test_sliding_window_none_normalised_to_zero(self) -> None:
        impl = self._make(sliding_window=None)
        self.assertEqual(impl.sliding_window, 0)

    def test_sliding_window_int_passes_through(self) -> None:
        impl = self._make(sliding_window=512)
        self.assertEqual(impl.sliding_window, 512)

    def test_sinks_shape_must_match_num_heads(self) -> None:
        # 4 != num_heads=8 → ValueError.
        sinks = torch.zeros(4, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            self._make(sinks=sinks)

    def test_num_kv_heads_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_kv_heads must be positive"):
            self._make(num_kv_heads=0)
        with self.assertRaisesRegex(ValueError, "num_kv_heads must be positive"):
            self._make(num_kv_heads=-1)

    def test_num_kv_heads_must_divide_num_heads(self) -> None:
        # num_heads=8, num_kv_heads=3 -> 8 % 3 != 0 -> ValueError.
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            self._make(num_kv_heads=3)

    def test_sinks_correct_shape_accepted(self) -> None:
        sinks = torch.zeros(8, dtype=torch.bfloat16)
        impl = self._make(sinks=sinks)
        self.assertIs(impl.sinks, sinks)

    def test_no_sinks_default(self) -> None:
        self.assertIsNone(self._make().sinks)

    def test_logits_soft_cap_default_zero(self) -> None:
        self.assertEqual(self._make().logits_soft_cap, 0.0)

    def test_pool_starts_unbound(self) -> None:
        # _pace_pool is resolved on first forward; before that, None.
        self.assertIsNone(self._make()._pace_pool)


class TestPaceAttentionMetadata(unittest.TestCase):
    def test_full_construction(self) -> None:
        md = PaceAttentionMetadata(
            num_actual_tokens=10,
            query_start_loc=torch.tensor([0, 5, 10]),
            seq_lens=torch.tensor([5, 5]),
            num_reqs=2,
            query_lens=[5, 5],
            seq_ids=[1, 2],
        )
        self.assertEqual(md.num_actual_tokens, 10)
        self.assertEqual(md.num_reqs, 2)
        self.assertEqual(md.query_lens, [5, 5])
        self.assertEqual(md.seq_ids, [1, 2])

    def test_seq_ids_none_signals_warmup(self) -> None:
        md = PaceAttentionMetadata(
            num_actual_tokens=0,
            query_start_loc=torch.tensor([0]),
            seq_lens=torch.tensor([]),
            num_reqs=0,
            query_lens=[],
            seq_ids=None,
        )
        self.assertIsNone(md.seq_ids)


class TestPaceAttentionImplOutputBuffer(unittest.TestCase):
    """vLLM 0.20+ pre-allocates `output` and discards the return value;
    PaceAttentionImpl.forward must write into it in-place. Stub the
    SlabPool dependency so this is a pure shape/buffer test."""

    def _make_impl(self) -> PaceAttentionImpl:
        return PaceAttentionImpl(num_heads=4, head_size=8, scale=0.1)

    def _make_layer(self):
        class _Layer:
            layer_name = "test_layer"

        return _Layer()

    def _make_pool(self, attn_value: torch.Tensor):
        # Minimal stand-in for SlabPool: records cache_update calls and
        # returns a fixed (T, H, D) tensor from attention(...).
        class _Pool:
            def __init__(self) -> None:
                self.cache_update_called = False

            def cache_update(self, *_args, **_kwargs) -> None:
                self.cache_update_called = True

            def attention(self, *_args, **_kwargs) -> torch.Tensor:
                return attn_value

        return _Pool()

    def _make_metadata(self, num_tokens: int) -> PaceAttentionMetadata:
        return PaceAttentionMetadata(
            num_actual_tokens=num_tokens,
            query_start_loc=torch.tensor([0, num_tokens]),
            seq_lens=torch.tensor([num_tokens]),
            num_reqs=1,
            query_lens=[num_tokens],
            seq_ids=[0],
        )

    def test_output_buffer_is_filled_in_place_when_provided(self) -> None:
        impl = self._make_impl()
        attn_value = torch.full((3, 4, 8), 7.0, dtype=torch.bfloat16)
        impl._pace_pool = self._make_pool(attn_value)

        query = torch.zeros((3, 32), dtype=torch.bfloat16)
        key = torch.zeros((3, 32), dtype=torch.bfloat16)
        value = torch.zeros((3, 32), dtype=torch.bfloat16)
        kv_cache = torch.empty(0)
        # Sentinel value so we can prove the buffer was overwritten (not
        # just returned untouched as garbage).
        output = torch.full_like(query, float("nan"))

        ret = impl.forward(
            self._make_layer(),
            query,
            key,
            value,
            kv_cache,
            self._make_metadata(3),
            output=output,
        )

        self.assertIs(ret, output)
        self.assertTrue(torch.equal(output, attn_value.view_as(query)))

    def test_returns_fresh_tensor_when_output_is_none(self) -> None:
        # vLLM 0.19 path: output=None, caller consumes the return value.
        impl = self._make_impl()
        attn_value = torch.full((2, 4, 8), 3.0, dtype=torch.bfloat16)
        impl._pace_pool = self._make_pool(attn_value)

        query = torch.zeros((2, 32), dtype=torch.bfloat16)
        key = torch.zeros((2, 32), dtype=torch.bfloat16)
        value = torch.zeros((2, 32), dtype=torch.bfloat16)
        kv_cache = torch.empty(0)

        ret = impl.forward(
            self._make_layer(),
            query,
            key,
            value,
            kv_cache,
            self._make_metadata(2),
            output=None,
        )

        self.assertTrue(torch.equal(ret, attn_value.view_as(query)))

    def test_shared_kv_layer_skips_cache_update(self) -> None:
        impl = self._make_impl()
        attn_value = torch.full((2, 4, 8), 5.0, dtype=torch.bfloat16)
        pool = self._make_pool(attn_value)
        impl._pace_pool = pool
        query = torch.zeros((2, 32), dtype=torch.bfloat16)
        output = torch.empty_like(query)

        ret = impl.forward(
            self._make_layer(),
            query,
            None,
            None,
            torch.empty(0),
            self._make_metadata(2),
            output=output,
        )

        self.assertFalse(pool.cache_update_called)
        self.assertIs(ret, output)
        self.assertTrue(torch.equal(output, attn_value.view_as(query)))

    def test_warmup_returns_output_buffer_when_provided(self) -> None:
        # During dummy_run / warmup, attn_metadata is None. We must not
        # allocate a fresh tensor on the hot path and must return the
        # caller-provided buffer as-is.
        impl = self._make_impl()
        query = torch.zeros((2, 32), dtype=torch.bfloat16)
        key = torch.zeros((2, 32), dtype=torch.bfloat16)
        value = torch.zeros((2, 32), dtype=torch.bfloat16)
        kv_cache = torch.empty(0)
        output = torch.full_like(query, 9.0)

        ret = impl.forward(
            self._make_layer(),
            query,
            key,
            value,
            kv_cache,
            attn_metadata=None,
            output=output,
        )

        self.assertIs(ret, output)


class TestPaceAttentionImplLayerRegistryMissRaises(unittest.TestCase):
    """C1 regression: a layer-registry miss in steady state is a config
    error (PaceModelRunner.initialize_kv_cache forgot a layer or KV-sharing
    follower). forward() must raise rather than return uninitialised memory
    or an untouched output buffer."""

    def setUp(self) -> None:
        from pace_vllm.v1.attention.layer_registry import clear_layer_registry

        clear_layer_registry()

    def tearDown(self) -> None:
        from pace_vllm.v1.attention.layer_registry import clear_layer_registry

        clear_layer_registry()

    def test_forward_raises_when_layer_not_registered(self) -> None:
        impl = PaceAttentionImpl(num_heads=4, head_size=8, scale=0.1)

        class _Layer:
            layer_name = "unregistered_layer"

        query = torch.zeros((3, 32), dtype=torch.bfloat16)
        key = torch.zeros((3, 32), dtype=torch.bfloat16)
        value = torch.zeros((3, 32), dtype=torch.bfloat16)
        kv_cache = torch.empty(0)
        md = PaceAttentionMetadata(
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 3]),
            seq_lens=torch.tensor([3]),
            num_reqs=1,
            query_lens=[3],
            seq_ids=[0],
        )

        with self.assertRaisesRegex(RuntimeError, "unregistered_layer"):
            impl.forward(_Layer(), query, key, value, kv_cache, md, output=None)


class TestPaceAttentionMetadataBuilderUnresolvedReqIdRaises(unittest.TestCase):
    """C1b regression: an unresolved req_id mid-batch must raise rather than
    silently demote seq_ids to None (which would emit garbage attention for
    the entire batch, including the requests that DID resolve)."""

    def setUp(self) -> None:
        from pace_vllm.v1.attention.layer_registry import (
            get_current_kv_cache,
            set_current_kv_cache,
        )

        self._saved_kv_cache = get_current_kv_cache()
        set_current_kv_cache(None)

    def tearDown(self) -> None:
        from pace_vllm.v1.attention.layer_registry import set_current_kv_cache

        set_current_kv_cache(self._saved_kv_cache)

    def test_build_raises_when_req_id_unresolved(self) -> None:
        from pace_vllm.v1.attention.layer_registry import set_current_kv_cache

        class _InputBatch:
            req_ids = ["known-req", "unknown-req"]

        class _Owner:
            input_batch = _InputBatch()

        class _StubKVCache:
            def get_owner(self):
                return _Owner()

            def get_sequence_id(self, rid):
                return 0 if rid == "known-req" else None

        set_current_kv_cache(_StubKVCache())

        # build() only reads common_attn_metadata + the module-level current
        # kv_cache; it touches no builder attributes, so __new__ skips the
        # heavy AttentionMetadataBuilder.__init__ chain.
        builder = PaceAttentionMetadataBuilder.__new__(PaceAttentionMetadataBuilder)

        common_md = type("_CM", (), {})()
        common_md.num_reqs = 2
        common_md.num_actual_tokens = 5
        common_md.query_start_loc = torch.tensor([0, 3, 5])
        common_md.seq_lens = torch.tensor([3, 2])

        with self.assertRaisesRegex(RuntimeError, "unknown-req"):
            builder.build(common_prefix_len=0, common_attn_metadata=common_md)

    def test_build_raises_when_req_ids_is_none(self) -> None:
        # input_batch exists (not warmup) but req_ids is None -> state
        # divergence; must raise rather than silently fall through to
        # seq_ids=None and emit garbage attention.
        from pace_vllm.v1.attention.layer_registry import set_current_kv_cache

        class _InputBatch:
            req_ids = None

        class _Owner:
            input_batch = _InputBatch()

        class _StubKVCache:
            def get_owner(self):
                return _Owner()

        set_current_kv_cache(_StubKVCache())

        builder = PaceAttentionMetadataBuilder.__new__(PaceAttentionMetadataBuilder)
        common_md = type("_CM", (), {})()
        common_md.num_reqs = 2
        common_md.num_actual_tokens = 5
        common_md.query_start_loc = torch.tensor([0, 3, 5])
        common_md.seq_lens = torch.tensor([3, 2])

        with self.assertRaisesRegex(RuntimeError, "missing or shorter"):
            builder.build(common_prefix_len=0, common_attn_metadata=common_md)

    def test_build_raises_when_req_ids_shorter_than_num_reqs(self) -> None:
        from pace_vllm.v1.attention.layer_registry import set_current_kv_cache

        class _InputBatch:
            req_ids = ["only-one"]

        class _Owner:
            input_batch = _InputBatch()

        class _StubKVCache:
            def get_owner(self):
                return _Owner()

        set_current_kv_cache(_StubKVCache())

        builder = PaceAttentionMetadataBuilder.__new__(PaceAttentionMetadataBuilder)
        common_md = type("_CM", (), {})()
        common_md.num_reqs = 2
        common_md.num_actual_tokens = 5
        common_md.query_start_loc = torch.tensor([0, 3, 5])
        common_md.seq_lens = torch.tensor([3, 2])

        with self.assertRaisesRegex(RuntimeError, "missing or shorter"):
            builder.build(common_prefix_len=0, common_attn_metadata=common_md)

    def test_build_returns_seq_ids_none_when_input_batch_missing(self) -> None:
        # `input_batch is None` is the legitimate warmup window before
        # PaceModelRunner wires its InputBatch onto the worker; must NOT raise.
        from pace_vllm.v1.attention.layer_registry import set_current_kv_cache

        class _Owner:
            input_batch = None

        class _StubKVCache:
            def get_owner(self):
                return _Owner()

        set_current_kv_cache(_StubKVCache())

        builder = PaceAttentionMetadataBuilder.__new__(PaceAttentionMetadataBuilder)
        common_md = type("_CM", (), {})()
        common_md.num_reqs = 1
        common_md.num_actual_tokens = 3
        common_md.query_start_loc = torch.tensor([0, 3])
        common_md.seq_lens = torch.tensor([3])

        md = builder.build(common_prefix_len=0, common_attn_metadata=common_md)
        self.assertIsNone(md.seq_ids)


if __name__ == "__main__":
    unittest.main()
