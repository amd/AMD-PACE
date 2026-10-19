# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.testing._internal.common_utils import TestCase

from pace.llm.attention import Attention, AttentionBackendType, KVCacheType
from pace.llm.attention.eviction.cache import (
    EvictingCache,
    EvictingCacheContext,
    EvictionState,
)
from pace.llm.attention.eviction.config import EvictionConfig, RoPEMode, UpdateMode
from pace.llm.attention.eviction.guards import validate_eviction_setup
from pace.llm.attention.eviction.rope import EvictionRoPE
from pace.ops.rotary_embedding import RotaryEmbedding

HEADS, HEAD_DIM, MAX_POS = 4, 16, 512


def rotary(max_pos=MAX_POS):
    return RotaryEmbedding(
        rope_scaling=None,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=max_pos,
        rope_theta=10000,
    )


def rotate(rope, tensor, positions):
    cos, sin = rope(tensor, positions, return_cos_sin=True)
    return rope.apply_rotary_emb(tensor, tensor, cos=cos, sin=sin, unsqueeze_dim=1)[0]


def context(config, rope=None, layers=1, kv_cache_type=KVCacheType.DYNAMIC):
    return EvictingCacheContext(
        SimpleNamespace(num_hidden_layers=layers),
        MAX_POS,
        config,
        rotary=rope,
        kv_cache_type=kv_cache_type,
    )


def attention():
    config = MagicMock()
    config.attention = AttentionBackendType.NATIVE
    return Attention(HEADS, HEADS, HEAD_DIM, config)


def tokens(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, HEADS, 1, HEAD_DIM)
    return [
        tuple(torch.randn(shape, generator=generator) for _ in range(3))
        for _ in range(count)
    ]


def reference_outputs(config, samples):
    rope = rotary()
    keys, values, outputs = [], [], []
    for query, key, value in samples:
        keys.append(key)
        values.append(value)
        if len(keys) > config.budget:
            keys = keys[: config.sink_size] + keys[-config.recent_size :]
            values = values[: config.sink_size] + values[-config.recent_size :]
        retained_keys, retained_values = torch.cat(keys, 2), torch.cat(values, 2)
        positions = torch.arange(retained_keys.shape[2]).unsqueeze(0)
        query = rotate(rope, query, positions[:, -1:])
        key = rotate(rope, retained_keys, positions)
        outputs.append(
            (query @ key.transpose(-1, -2) / math.sqrt(HEAD_DIM)).softmax(-1)
            @ retained_values
        )
    return outputs


def run_contiguous(config, samples):
    rope, cache, attn = rotary(), context(config, rotary())[0], attention()
    outputs = []
    for step, (query, key, value) in enumerate(samples):
        position = torch.tensor([[step]])
        if config.rope_mode == RoPEMode.SINK_ONLY:
            query, key = rotate(rope, query, position), rotate(rope, key, position)
        output = attn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            cache,
            position,
        )
        outputs.append(output.transpose(1, 2))
    return cache, outputs


def model_stub(supports_rope=True, has_rotary=True, **config):
    return SimpleNamespace(
        config=SimpleNamespace(**config),
        supports_eviction_rope=supports_rope,
        model=SimpleNamespace(rotary_emb=object() if has_rotary else None),
    )


class TestEvictionState(TestCase):
    def test_ring_positions_and_independent_rows(self):
        state = EvictionState(EvictionConfig(2, 4))
        slots = [int(state.admit(1)[0][0, 0]) for _ in range(14)]
        self.assertEqual(slots, list(range(6)) + [2, 3, 4, 5] * 2)
        positions = state.logical_positions()[0]
        self.assertEqual(positions[:2].tolist(), [0, 1])
        self.assertEqual(sorted(positions.tolist()), list(range(6)))

        state = EvictionState(EvictionConfig(1, 3), batch_size=2)
        state.filled, state.write_idx = torch.tensor([4, 1]), torch.tensor([2, 1])
        written, _ = state.admit(1)
        self.assertEqual(written.tolist(), [[2], [1]])
        self.assertEqual(state.filled.tolist(), [4, 2])


class TestContiguousEviction(TestCase):
    def test_matches_reference(self):
        samples = tokens(30)
        for mode in (RoPEMode.REINDEX, RoPEMode.SINK_ONLY):
            with self.subTest(mode=mode):
                config = EvictionConfig(2, 6, rope_mode=mode)
                cache, actual = run_contiguous(config, samples)
                for expected, got in zip(reference_outputs(config, samples), actual):
                    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
                self.assertEqual(cache.seq_len, config.budget)
                self.assertEqual(
                    cache.context.state.total_seen.tolist(), [len(samples)]
                )
                if mode == RoPEMode.REINDEX:
                    shared, attn = context(config, rotary(), layers=2), attention()
                    for step, (query, key, value) in enumerate(samples[:10]):
                        position = torch.tensor([[step]])
                        outputs = [
                            attn(
                                query.transpose(1, 2),
                                key.transpose(1, 2),
                                value.transpose(1, 2),
                                shared[layer],
                                position,
                            )
                            for layer in range(2)
                        ]
                        torch.testing.assert_close(outputs[0], outputs[1])
                else:
                    rope = rotary()
                    expected_sinks = EvictionRoPE(config, rope).reindex_sinks(
                        cache.sink_pristine,
                        cache.context.state.total_seen,
                        config.budget,
                        1,
                        config.sink_size,
                    )
                    torch.testing.assert_close(
                        cache.key.narrow(2, 0, config.sink_size), expected_sinks
                    )

    def test_shift_matches_inplace_reindex(self):
        samples = tokens(24)
        inplace = EvictionConfig(
            2, 6, rope_mode=RoPEMode.REINDEX, update=UpdateMode.INPLACE
        )
        shift = EvictionConfig(
            2, 6, rope_mode=RoPEMode.REINDEX, update=UpdateMode.SHIFT
        )
        _, got_inplace = run_contiguous(inplace, samples)
        _, got_shift = run_contiguous(shift, samples)
        for a, b in zip(got_inplace, got_shift):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)

    def test_bmc_matches_dynamic_and_rounds_capacity(self):
        samples = tokens(20)
        config = EvictionConfig(2, 6, rope_mode=RoPEMode.REINDEX)
        _, got_dynamic = run_contiguous(config, samples)

        _, cache, attn = (
            rotary(),
            context(config, rotary(), kv_cache_type=KVCacheType.BMC)[0],
            attention(),
        )
        self.assertEqual(cache.context.kv_cache_type, KVCacheType.BMC)
        self.assertIsNotNone(cache.context.tokens_per_split)
        outputs = []
        for step, (query, key, value) in enumerate(samples):
            position = torch.tensor([[step]])
            output = attn(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                cache,
                position,
            )
            outputs.append(output.transpose(1, 2))
        for a, b in zip(got_dynamic, outputs):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        self.assertEqual(cache.context.capacity % cache.context.tokens_per_split, 0)
        self.assertGreaterEqual(cache.context.capacity, config.budget)

    def test_bmc_compact_keeps_segment_capacity(self):
        config = EvictionConfig(2, 4, rope_mode=RoPEMode.REINDEX)
        cache, attn = (
            context(config, rotary(), kv_cache_type=KVCacheType.BMC)[0],
            attention(),
        )
        length = 12
        keys = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        keys = keys.expand(1, HEADS, length, HEAD_DIM).contiguous()
        attn(
            torch.zeros(1, length, HEADS, HEAD_DIM),
            keys.transpose(1, 2),
            torch.zeros(1, length, HEADS, HEAD_DIM),
            cache,
            torch.arange(length).unsqueeze(0),
        )
        attn(
            torch.zeros(1, 1, HEADS, HEAD_DIM),
            torch.full((1, 1, HEADS, HEAD_DIM), 99.0),
            torch.zeros(1, 1, HEADS, HEAD_DIM),
            cache,
            torch.tensor([[length]]),
        )
        self.assertEqual(cache.context.capacity % cache.context.tokens_per_split, 0)
        self.assertGreaterEqual(cache.context.capacity, config.budget)
        self.assertEqual(cache.key.size(2), cache.context.capacity)

    def test_overlong_prefill_compacts(self):
        config = EvictionConfig(2, 4, rope_mode=RoPEMode.REINDEX)
        cache, attn = context(config, rotary())[0], attention()
        length = 12
        keys = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        keys = keys.expand(1, HEADS, length, HEAD_DIM).contiguous()
        attn(
            torch.zeros(1, length, HEADS, HEAD_DIM),
            keys.transpose(1, 2),
            torch.zeros(1, length, HEADS, HEAD_DIM),
            cache,
            torch.arange(length).unsqueeze(0),
        )
        self.assertEqual(cache.seq_len, length)
        attn(
            torch.zeros(1, 1, HEADS, HEAD_DIM),
            torch.full((1, 1, HEADS, HEAD_DIM), 99.0),
            torch.zeros(1, 1, HEADS, HEAD_DIM),
            cache,
            torch.tensor([[length]]),
        )
        retained = cache.key[0, 0, :, 0].tolist()
        self.assertEqual(retained[:2], [0.0, 1.0])
        self.assertEqual(sorted(retained[2:]), [9.0, 10.0, 11.0, 99.0])

    def test_left_padding_matches_unpadded(self):
        config, real_len = EvictionConfig(2, 4, rope_mode=RoPEMode.REINDEX), 9
        generator = torch.Generator().manual_seed(5)
        samples = [
            torch.randn(1, HEADS, real_len, HEAD_DIM, generator=generator)
            for _ in range(3)
        ]

        def run(pad):
            cache, attn = context(config, rotary())[0], attention()
            positions = torch.cat([torch.zeros(pad), torch.arange(real_len)]).long()[
                None
            ]

            def padded(tensor):
                filler = torch.randn(1, HEADS, pad, HEAD_DIM, generator=generator)
                return torch.cat([filler, tensor], 2)

            prefill = attn(
                *(padded(tensor).transpose(1, 2) for tensor in samples),
                cache,
                positions,
            )
            decode = []
            for step in range(3):
                torch.manual_seed(100 + step)
                decode.append(
                    attn(
                        *[torch.randn(1, 1, HEADS, HEAD_DIM) for _ in range(3)],
                        cache,
                        torch.tensor([[real_len + step]]),
                    )
                )
            return prefill[:, pad:], decode

        expected, expected_decode = run(0)
        actual, actual_decode = run(3)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        for got, want in zip(actual_decode, expected_decode):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


class TestEvictionGuards(TestCase):
    def test_unsupported_combinations(self):
        cases = [
            (
                EvictionConfig(),
                KVCacheType.DYNAMIC,
                model_stub(),
                object(),
                "speculative",
            ),
            (
                EvictionConfig(),
                KVCacheType.DYNAMIC,
                model_stub(sliding_window=128),
                None,
                "sliding_window",
            ),
            (
                EvictionConfig(),
                KVCacheType.DYNAMIC,
                model_stub(has_rotary=False),
                None,
                "rotary position",
            ),
            (
                EvictionConfig(rope_mode=RoPEMode.REINDEX),
                KVCacheType.DYNAMIC,
                model_stub(supports_rope=False),
                None,
                "Llama-style",
            ),
            (
                EvictionConfig(rope_mode=RoPEMode.SINK_ONLY),
                KVCacheType.DYNAMIC,
                model_stub(supports_rope=False),
                None,
                "Llama-style",
            ),
            (EvictionConfig(), KVCacheType.PAGED, model_stub(), None, "contiguous"),
            (EvictionConfig(), KVCacheType.SLAB_POOL, model_stub(), None, "contiguous"),
            (
                EvictionConfig(),
                KVCacheType.DYNAMIC,
                model_stub(num_kv_shared_layers=2),
                None,
                "KV-sharing",
            ),
        ]
        for config, cache_type, model, spec, message in cases:
            with self.subTest(message=message, cache_type=cache_type):
                with self.assertRaisesRegex(NotImplementedError, message):
                    validate_eviction_setup(config, cache_type, model, spec)

    def test_sink_only_limits(self):
        long_rope = rotary(16)
        long_rope.rope_type = "longrope"
        with self.assertRaisesRegex(ValueError, "linear in position"):
            EvictionRoPE(EvictionConfig(rope_mode=RoPEMode.SINK_ONLY), long_rope)

        config = EvictionConfig(1, 3, rope_mode=RoPEMode.SINK_ONLY)
        cache, attn = context(config, rotary(8))[0], attention()
        with self.assertRaisesRegex(ValueError, "cannot generate past the RoPE"):
            for step in range(10):
                attn(
                    *[torch.randn(1, 1, HEADS, HEAD_DIM) for _ in range(3)],
                    cache,
                    torch.tensor([[step]]),
                )

    def test_merge_contexts_rejects_batches(self):
        backend = EvictingCache(EvictionConfig())
        single = context(EvictionConfig())
        self.assertIs(backend.merge_contexts([single]), single)
        with self.assertRaisesRegex(NotImplementedError, "batch independent"):
            backend.merge_contexts([single, context(EvictionConfig())])

    def test_rejects_multi_token_update_after_ring_permutes(self):
        config = EvictionConfig(2, 4, rope_mode=RoPEMode.REINDEX)
        cache, attn = context(config, rotary())[0], attention()
        for step in range(config.budget):
            attn(
                *[torch.randn(1, 1, HEADS, HEAD_DIM) for _ in range(3)],
                cache,
                torch.tensor([[step]]),
            )
        with self.assertRaisesRegex(NotImplementedError, "multi-token updates"):
            attn(
                *[torch.randn(1, 2, HEADS, HEAD_DIM) for _ in range(3)],
                cache,
                torch.tensor([[config.budget, config.budget + 1]]),
            )

    def test_rejects_overlong_chunk_after_partial_fill(self):
        config = EvictionConfig(2, 4, rope_mode=RoPEMode.REINDEX)
        cache, attn = context(config, rotary())[0], attention()
        for step in range(3):
            attn(
                *[torch.randn(1, 1, HEADS, HEAD_DIM) for _ in range(3)],
                cache,
                torch.tensor([[step]]),
            )
        with self.assertRaisesRegex(NotImplementedError, "multi-token updates"):
            attn(
                *[torch.randn(1, config.budget + 1, HEADS, HEAD_DIM) for _ in range(3)],
                cache,
                torch.arange(3, 3 + config.budget + 1).unsqueeze(0),
            )
