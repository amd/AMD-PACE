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
        # 3 pad positions + 2 real positions
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
        # seq0: 2 pad + 4 real, seq1: 4 pad + 2 real
        pos = torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 1]])

        kv1 = BMCKVCache(max_seq_length=64)
        out1 = attn(Q, K, V, kv1, pos)

        K2 = K.clone()
        V2 = V.clone()
        K2[0, :2] = torch.randn_like(K2[0, :2])  # perturb seq0 pad
        V2[0, :2] = torch.randn_like(V2[0, :2])
        K2[1, :4] = torch.randn_like(K2[1, :4])  # perturb seq1 pad
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

        # Two concurrent requests share the same (bs, q_len, kv_len) but
        # have different amounts of leading padding.
        leading_pad_a = torch.tensor([2])  # request A: 2 leading-pad tokens
        leading_pad_b = torch.tensor([4])  # request B: 4 leading-pad tokens

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

        # Buffers must be distinct allocations.
        self.assertNotEqual(
            mask_a.data_ptr(),
            mask_b.data_ptr(),
            "Different owners must receive separate mask buffers (no aliasing)",
        )

        # Each mask must reflect its own leading-pad, not the other's.
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

        # After getting mask_b, mask_a must be unchanged (no in-place overwrite).
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
