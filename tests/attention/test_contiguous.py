# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

import torch
import torch.nn.functional as F

from pace.llm.attention import Attention, AttentionBackendType
from pace.llm.attention.contiguous.cache import DynamicKVCache


def _make_opconfig(backend: AttentionBackendType):
    """Create a mock opconfig with the given attention backend."""
    cfg = MagicMock()
    cfg.attention = backend
    return cfg


def _make_positions(batch, seq_len):
    """Create sequential positions [0, 1, ..., seq_len-1] for each batch."""
    return torch.arange(seq_len).unsqueeze(0).expand(batch, -1)


def _reset_encoder_caches():
    from pace.llm.attention.contiguous.backend import (
        reset_encoder_mask_cache,
        reset_encoder_window_cache,
    )

    reset_encoder_window_cache()
    reset_encoder_mask_cache()


class TestContiguousAttention(TestCase):

    @given(
        backend=st.sampled_from(
            [AttentionBackendType.JIT, AttentionBackendType.NATIVE]
        ),
    )
    def test_mha(self, backend):
        """Test MHA via Attention wrapper."""
        opconfig = _make_opconfig(backend)
        attn = Attention(num_heads=8, num_kv_heads=8, head_dim=16, opconfig=opconfig)

        batch, seq, head, dim = 2, 4, 8, 16
        Q = torch.randn(batch, seq, head, dim)
        K = torch.randn(batch, seq, head, dim)
        V = torch.randn(batch, seq, head, dim)
        kv_cache = DynamicKVCache(max_seq_length=64)
        positions = _make_positions(batch, seq)

        out = attn(Q, K, V, kv_cache, positions)
        self.assertEqual(out.shape, (batch, seq, head, dim))

    @given(
        backend=st.sampled_from(
            [AttentionBackendType.JIT, AttentionBackendType.NATIVE]
        ),
    )
    def test_gqa(self, backend):
        """Test GQA via Attention wrapper (num_kv_heads < num_heads)."""
        opconfig = _make_opconfig(backend)
        attn = Attention(num_heads=8, num_kv_heads=2, head_dim=16, opconfig=opconfig)

        batch, seq = 2, 4
        Q = torch.randn(batch, seq, 8, 16)
        K = torch.randn(batch, seq, 2, 16)
        V = torch.randn(batch, seq, 2, 16)
        kv_cache = DynamicKVCache(max_seq_length=64)
        positions = _make_positions(batch, seq)

        out = attn(Q, K, V, kv_cache, positions)
        self.assertEqual(out.shape, (batch, seq, 8, 16))

    @given(
        backend=st.sampled_from(
            [AttentionBackendType.JIT, AttentionBackendType.NATIVE]
        ),
    )
    def test_mha_correctness(self, backend):
        """Verify MHA output matches PyTorch SDPA numerically."""
        opconfig = _make_opconfig(backend)
        attn = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)

        batch, seq, head, dim = 1, 6, 4, 16
        Q = torch.randn(batch, seq, head, dim)
        K = torch.randn(batch, seq, head, dim)
        V = torch.randn(batch, seq, head, dim)
        kv_cache = DynamicKVCache(max_seq_length=64)
        positions = _make_positions(batch, seq)

        out = attn(Q, K, V, kv_cache, positions)

        ref = F.scaled_dot_product_attention(
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), is_causal=True
        ).transpose(1, 2)

        threshold = 1e-4 if backend == AttentionBackendType.NATIVE else 1e-1
        self.assertTrue(
            torch.allclose(out, ref, atol=threshold, rtol=threshold),
            f"MHA output differs from PyTorch SDPA (backend={backend})",
        )

    @given(
        backend=st.sampled_from(
            [AttentionBackendType.JIT, AttentionBackendType.NATIVE]
        ),
    )
    def test_gqa_correctness(self, backend):
        """Verify GQA output matches PyTorch SDPA numerically."""
        opconfig = _make_opconfig(backend)
        attn = Attention(num_heads=8, num_kv_heads=2, head_dim=16, opconfig=opconfig)

        batch, seq = 1, 6
        Q = torch.randn(batch, seq, 8, 16)
        K = torch.randn(batch, seq, 2, 16)
        V = torch.randn(batch, seq, 2, 16)
        kv_cache = DynamicKVCache(max_seq_length=64)
        positions = _make_positions(batch, seq)

        out = attn(Q, K, V, kv_cache, positions)

        ref = F.scaled_dot_product_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)

        threshold = 1e-4 if backend == AttentionBackendType.NATIVE else 1e-1
        self.assertTrue(
            torch.allclose(out, ref, atol=threshold, rtol=threshold),
            f"GQA output differs from PyTorch SDPA (backend={backend})",
        )

    def test_list_kv_cache(self):
        """Test attention with list of kv_caches (per-sequence, serving mode)."""
        opconfig = _make_opconfig(AttentionBackendType.JIT)
        attn = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)

        batch, seq, head, dim = 3, 2, 4, 16
        Q = torch.randn(batch, seq, head, dim)
        K = torch.randn(batch, seq, head, dim)
        V = torch.randn(batch, seq, head, dim)
        kv_caches = [DynamicKVCache(max_seq_length=64) for _ in range(batch)]
        positions = _make_positions(batch, seq)

        out = attn(Q, K, V, kv_caches, positions)
        self.assertEqual(out.shape[0], batch)
        self.assertEqual(out.shape[-1], dim)

    def test_output_bshd_layout(self):
        """Verify output is BSHD, same layout as input."""
        opconfig = _make_opconfig(AttentionBackendType.NATIVE)
        attn = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)

        Q = torch.randn(1, 8, 4, 16)
        K = torch.randn(1, 8, 4, 16)
        V = torch.randn(1, 8, 4, 16)
        kv_cache = DynamicKVCache(max_seq_length=64)
        positions = _make_positions(1, 8)

        out = attn(Q, K, V, kv_cache, positions)
        self.assertEqual(out.shape, (1, 8, 4, 16))

    def test_padding_masks_out_pad_positions(self):
        """Changing K/V at padded positions must NOT affect output at real positions."""
        opconfig = _make_opconfig(AttentionBackendType.NATIVE)
        attn = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)

        torch.manual_seed(42)
        Q = torch.randn(1, 5, 4, 16)
        K = torch.randn(1, 5, 4, 16)
        V = torch.randn(1, 5, 4, 16)
        pos = torch.tensor([[0, 0, 0, 0, 1]])

        kv1 = DynamicKVCache(max_seq_length=64)
        out1 = attn(Q, K, V, kv1, pos)

        K2 = K.clone()
        V2 = V.clone()
        K2[:, :3] = torch.randn_like(K2[:, :3])
        V2[:, :3] = torch.randn_like(V2[:, :3])

        attn2 = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)
        kv2 = DynamicKVCache(max_seq_length=64)
        out2 = attn2(Q, K2, V2, kv2, pos)

        self.assertTrue(
            torch.allclose(out1[0, 3:], out2[0, 3:], atol=1e-5),
            "Output at real positions must not change when padded K/V values change",
        )

    def test_padding_masks_out_pad_positions_batched(self):
        """Multi-batch with BMC: different padding per sequence must not leak."""
        from pace.llm.attention.contiguous.cache import BMCKVCache

        opconfig = _make_opconfig(AttentionBackendType.NATIVE)
        attn = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)

        torch.manual_seed(42)
        batch, seq, head, dim = 2, 6, 4, 16
        Q = torch.randn(batch, seq, head, dim)
        K = torch.randn(batch, seq, head, dim)
        V = torch.randn(batch, seq, head, dim)
        pos = torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 1]])

        kv1 = BMCKVCache(max_seq_length=64)
        out1 = attn(Q, K, V, kv1, pos)

        K2 = K.clone()
        V2 = V.clone()
        K2[0, :2] = torch.randn_like(K2[0, :2])
        V2[0, :2] = torch.randn_like(V2[0, :2])
        K2[1, :4] = torch.randn_like(K2[1, :4])
        V2[1, :4] = torch.randn_like(V2[1, :4])

        attn2 = Attention(num_heads=4, num_kv_heads=4, head_dim=16, opconfig=opconfig)
        kv2 = BMCKVCache(max_seq_length=64)
        out2 = attn2(Q, K2, V2, kv2, pos)

        self.assertTrue(
            torch.allclose(out1[0, 2:], out2[0, 2:], atol=1e-5),
            "Seq0: output at real positions must not change when padded K/V change",
        )
        self.assertTrue(
            torch.allclose(out1[1, 4:], out2[1, 4:], atol=1e-5),
            "Seq1: output at real positions must not change when padded K/V change",
        )

    def test_mask_cache_no_aliasing_same_shape_different_owners(self):
        """Concurrent requests with the same (shape, dtype) but different owners
        must receive independent mask buffers and must not alias each other.

        Regression test: without the ``owner`` key in the pool, the second
        call overwrites the shared buffer in-place so the first sequence sees
        the wrong mask for the rest of its forward pass.
        """
        from pace.llm.attention.contiguous.backend import MaskCache

        MaskCache.reset()
        cache = MaskCache()

        bs, q_len, kv_len, seq_len = 1, 1, 8, 6
        dtype = torch.float32
        min_val = torch.finfo(dtype).min

        leading_pad_a = torch.tensor([2])
        leading_pad_b = torch.tensor([4])

        mask_a = cache.get(
            bs,
            q_len,
            kv_len,
            seq_len,
            dtype,
            leading_pad=leading_pad_a,
            owner="req-uuid-a",
        )
        mask_b = cache.get(
            bs,
            q_len,
            kv_len,
            seq_len,
            dtype,
            leading_pad=leading_pad_b,
            owner="req-uuid-b",
        )

        self.assertNotEqual(
            mask_a.data_ptr(),
            mask_b.data_ptr(),
            "Different owners must receive separate mask buffers (no aliasing)",
        )

        self.assertTrue(
            torch.all(mask_a[0, 0, :, :2] == min_val),
            "Mask A: columns 0-1 must be masked (leading_pad=2)",
        )
        self.assertFalse(
            torch.all(mask_a[0, 0, :, 2:] == min_val),
            "Mask A: columns beyond pad must not all be masked",
        )
        self.assertTrue(
            torch.all(mask_b[0, 0, :, :4] == min_val),
            "Mask B: columns 0-3 must be masked (leading_pad=4)",
        )

        mask_a_again = cache.get(
            bs,
            q_len,
            kv_len,
            seq_len,
            dtype,
            leading_pad=leading_pad_a,
            owner="req-uuid-a",
        )
        self.assertEqual(
            mask_a.data_ptr(),
            mask_a_again.data_ptr(),
            "Cache hit for owner A must return the same buffer",
        )
        self.assertTrue(
            torch.all(mask_a_again[0, 0, :, :2] == min_val),
            "Owner A mask must still reflect its own padding after owner B wrote",
        )

        MaskCache.reset()

    def test_chunked_mask_uses_model_positions_with_leading_pad(self):
        """Chunk boundaries follow model positions, not physical cache indices:
        with two left pads and chunk_size=4, physical indices 2-5 are positions
        0-3 and must stay in one chunk rather than being split at physical 4.
        """
        from pace.llm.attention.contiguous.backend import MaskCache

        MaskCache.reset()
        cache = MaskCache()
        dtype = torch.float32
        min_val = torch.finfo(dtype).min
        chunk_size = 4

        mask = cache.get(
            1,
            6,
            6,
            6,
            dtype,
            leading_pad=torch.tensor([2]),
            chunk_size=chunk_size,
        )
        allowed = mask[0, 0] > (min_val / 2)

        got = [j for j in range(6) if bool(allowed[5, j])]
        self.assertEqual(
            got,
            [2, 3, 4, 5],
            "Left pad must not split real positions 0-3 across chunk boundary",
        )

        unpadded = cache.get(
            1,
            4,
            4,
            4,
            dtype,
            leading_pad=None,
            chunk_size=chunk_size,
        )[0, 0] > (min_val / 2)
        self.assertTrue(
            torch.equal(allowed[2:, 2:], unpadded),
            "Padded real-token attention pattern must equal the unpadded one",
        )

        batched = cache.get(
            2,
            7,
            7,
            7,
            dtype,
            leading_pad=torch.tensor([0, 3]),
            chunk_size=chunk_size,
        )
        ref_a = cache.get(
            1,
            7,
            7,
            7,
            dtype,
            leading_pad=torch.tensor([0]),
            owner="a",
            chunk_size=chunk_size,
        )
        ref_b = cache.get(
            1,
            7,
            7,
            7,
            dtype,
            leading_pad=torch.tensor([3]),
            owner="b",
            chunk_size=chunk_size,
        )
        self.assertTrue(torch.equal(batched[0], ref_a[0]))
        self.assertTrue(torch.equal(batched[1], ref_b[0]))

        MaskCache.reset()


class TestEncoderMaskDevice(TestCase):
    """The encoder mask must be built on the device of the tensors it masks.

    PACE serves on CPU, so ``meta`` stands in for a second device here: it is
    the only one available everywhere the suite runs, and it is enough to see
    whether the requested device is honoured or silently replaced by the
    default one.
    """

    @staticmethod
    def _encoder_backend(sliding_window):
        from pace.llm.attention.contiguous.backend import (
            ContiguousAttentionBackend,
        )

        return ContiguousAttentionBackend(
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
            attn_backend=AttentionBackendType.NATIVE,
            sliding_window=sliding_window,
            is_causal=False,
        )

    def test_window_mask_is_built_on_the_requested_device_without_a_mask(self):
        """A window mask is built even when there is no attention_mask, so the
        device cannot come from the mask. This is the ModernBERT local-attention
        path, where forward() leaves attention_mask at None.
        """
        backend = self._encoder_backend(sliding_window=4)

        mask = backend._build_encoder_mask(
            None, 1, 8, 8, torch.float32, torch.device("meta")
        )

        self.assertIsNotNone(mask, "a sliding window must produce a mask")
        self.assertEqual(
            mask.device.type,
            "meta",
            "window mask ignored the requested device and fell back to the default",
        )

    def test_padding_mask_is_built_on_the_requested_device(self):
        backend = self._encoder_backend(sliding_window=0)

        mask = backend._build_encoder_mask(
            torch.tensor([[1, 1, 0, 0]]), 1, 4, 4, torch.float32, torch.device("meta")
        )

        self.assertIsNotNone(mask, "padding must produce a mask")
        self.assertEqual(mask.device.type, "meta")

    def test_forward_encoder_builds_the_mask_on_the_qkv_device(self):
        """The device the mask is built on has to be the one Q/K/V live on,
        otherwise masked_fill_ would be mixing devices.
        """
        backend = self._encoder_backend(sliding_window=4)
        requested = []
        build = backend._build_encoder_mask

        def record_device(*args, **kwargs):
            requested.append(args[-1])
            return build(*args, **kwargs)

        backend._build_encoder_mask = record_device

        Q = torch.randn(1, 8, 4, 16)
        backend._forward_encoder(Q, Q.clone(), Q.clone())

        self.assertEqual(requested, [Q.device])

    def test_window_pattern_is_reused_for_short_sequences(self):
        from pace.llm.attention.contiguous.backend import (
            _encoder_windows,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=4)
        first = backend._build_encoder_mask(
            None, 1, 8, 8, torch.float32, torch.device("cpu")
        )
        self.assertEqual(len(_encoder_windows), 1)
        cached = next(iter(_encoder_windows.values()))
        second = backend._build_encoder_mask(
            None, 2, 8, 8, torch.float32, torch.device("cpu")
        )
        self.assertIs(next(iter(_encoder_windows.values())), cached)
        self.assertEqual(first.shape, (1, 1, 8, 8))
        self.assertEqual(second.shape, (2, 1, 8, 8))

    def test_window_pattern_is_shared_across_layers(self):
        """Every sliding layer of one encoder wants the same [q, kv] boolean.

        mmBERT-32K has 14 of them, so a per-backend cache rebuilt the pattern
        once per layer on every forward.
        """
        from pace.llm.attention.contiguous.backend import (
            _encoder_windows,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        layers = [self._encoder_backend(sliding_window=4) for _ in range(3)]
        for layer in layers:
            layer._build_encoder_mask(None, 1, 8, 8, torch.float32, torch.device("cpu"))

        self.assertEqual(
            len(_encoder_windows),
            1,
            "each layer built its own copy of the same window pattern",
        )

    def test_long_window_pattern_is_not_retained(self):
        from pace.llm.attention.contiguous.backend import (
            _ENCODER_WINDOW_CACHE_CELLS,
            _encoder_windows,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=4)
        n = int(_ENCODER_WINDOW_CACHE_CELLS**0.5) + 1
        backend._build_encoder_mask(None, 1, n, n, torch.float32, torch.device("cpu"))
        self.assertEqual(len(_encoder_windows), 0)

    def test_window_cache_is_bounded(self):
        from pace.llm.attention.contiguous.backend import (
            _ENCODER_WINDOW_CACHE_ENTRIES,
            _encoder_windows,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=4)
        for n in range(8, 8 + _ENCODER_WINDOW_CACHE_ENTRIES + 4):
            backend._build_encoder_mask(
                None, 1, n, n, torch.float32, torch.device("cpu")
            )

        self.assertEqual(len(_encoder_windows), _ENCODER_WINDOW_CACHE_ENTRIES)

    def test_additive_mask_is_reused_across_layers(self):
        from pace.llm.attention.contiguous.backend import (
            _encoder_masks,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        layers = [self._encoder_backend(sliding_window=4) for _ in range(3)]
        first = layers[0]._build_encoder_mask(
            None, 1, 8, 8, torch.float32, torch.device("cpu")
        )
        for layer in layers[1:]:
            other = layer._build_encoder_mask(
                None, 1, 8, 8, torch.float32, torch.device("cpu")
            )
            self.assertIs(other, first)
        self.assertEqual(len(_encoder_masks), 1)

    def test_additive_mask_keeps_separate_entries_per_padding(self):
        from pace.llm.attention.contiguous.backend import (
            _encoder_masks,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=0)
        pad_a = torch.tensor([[1, 1, 0, 0]])
        pad_b = torch.tensor([[1, 0, 0, 0]])
        first = backend._build_encoder_mask(
            pad_a, 1, 4, 4, torch.float32, torch.device("cpu")
        )
        second = backend._build_encoder_mask(
            pad_b, 1, 4, 4, torch.float32, torch.device("cpu")
        )
        self.assertIsNot(second, first)
        self.assertEqual(len(_encoder_masks), 2)
        self.assertLess(second[0, 0, 0, 1].item(), -1e30)
        self.assertLess(second[0, 0, 0, 2].item(), -1e30)

    def test_long_additive_mask_is_not_retained(self):
        from pace.llm.attention.contiguous.backend import (
            _ENCODER_WINDOW_CACHE_CELLS,
            _encoder_masks,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=0)
        n = int(_ENCODER_WINDOW_CACHE_CELLS**0.5) + 1
        pad = torch.ones(1, n)
        pad[0, -1] = 0
        backend._build_encoder_mask(pad, 1, n, n, torch.float32, torch.device("cpu"))
        self.assertEqual(len(_encoder_masks), 0)

    def test_additive_mask_cache_is_bounded(self):
        from pace.llm.attention.contiguous.backend import (
            _ENCODER_MASK_CACHE_ENTRIES,
            _encoder_masks,
        )

        _reset_encoder_caches()
        self.addCleanup(_reset_encoder_caches)

        backend = self._encoder_backend(sliding_window=4)
        for n in range(8, 8 + _ENCODER_MASK_CACHE_ENTRIES + 4):
            backend._build_encoder_mask(
                None, 1, n, n, torch.float32, torch.device("cpu")
            )

        self.assertEqual(len(_encoder_masks), _ENCODER_MASK_CACHE_ENTRIES)
