# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401
from pace.ops.rotary_embedding import RotaryEmbedding


def _python_rope_reference(x, cos, sin, unsqueeze_dim):
    """Pure-Python neox-style RoPE reference implementation."""
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out_first = x1 * cos_u - x2 * sin_u
    out_second = x2 * cos_u + x1 * sin_u
    return torch.cat([out_first, out_second], dim=-1)


class TestFusedRoPE(TestCase):

    def _run_fused_rope_test(self, BS, seq_len, num_heads, head_dim, unsqueeze_dim):
        """Helper: compare C++ fused_rope against Python reference."""
        if unsqueeze_dim == 1:
            q = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
            k = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        else:
            q = torch.randn(BS, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
            k = torch.randn(BS, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

        cos = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.bfloat16)
        sin = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.bfloat16)

        q_ref = _python_rope_reference(q, cos, sin, unsqueeze_dim)
        k_ref = _python_rope_reference(k, cos, sin, unsqueeze_dim)

        q_fused, k_fused = torch.ops.pace.fused_rope_apply(
            q, k, cos, sin, unsqueeze_dim
        )

        self.assertEqual(q_fused, q_ref, atol=2e-2, rtol=2e-2)
        self.assertEqual(k_fused, k_ref, atol=2e-2, rtol=2e-2)

    def test_fused_rope_bnsh(self):
        """Test fused RoPE with BNSH layout (unsqueeze_dim=1)."""
        shapes = [
            (1, 4, 8, 64),
            (2, 16, 4, 128),
            (1, 1, 32, 32),
            (4, 8, 2, 96),
        ]
        for BS, seq_len, num_heads, head_dim in shapes:
            with self.subTest(
                BS=BS, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
            ):
                self._run_fused_rope_test(BS, seq_len, num_heads, head_dim, 1)

    def test_fused_rope_bsnh(self):
        """Test fused RoPE with BSNH layout (unsqueeze_dim=2)."""
        shapes = [
            (1, 4, 8, 64),
            (2, 16, 4, 128),
            (1, 1, 32, 32),
            (4, 8, 2, 96),
        ]
        for BS, seq_len, num_heads, head_dim in shapes:
            with self.subTest(
                BS=BS, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
            ):
                self._run_fused_rope_test(BS, seq_len, num_heads, head_dim, 2)

    def _make_rope(self, head_dim):
        return RotaryEmbedding(
            rope_scaling=None,
            rotary_dim=head_dim,
            max_position_embeddings=128,
            rope_theta=10000,
        )

    def test_fused_rope_apply_rotary_emb(self):
        """End-to-end: RotaryEmbedding.apply_rotary_emb fast-path matches fallback."""
        BS, seq_len, num_heads, head_dim = 2, 8, 4, 64
        rope = self._make_rope(head_dim)

        q = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        cos = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.bfloat16)
        sin = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.bfloat16)

        q_ref = _python_rope_reference(q, cos, sin, unsqueeze_dim=1)
        k_ref = _python_rope_reference(k, cos, sin, unsqueeze_dim=1)

        q_out, k_out = rope.apply_rotary_emb(
            q, k, cos, sin, unsqueeze_dim=1, is_neox_style=True
        )

        self.assertEqual(q_out, q_ref, atol=2e-2, rtol=2e-2)
        self.assertEqual(k_out, k_ref, atol=2e-2, rtol=2e-2)

    def test_fused_rope_fallback_non_bf16(self):
        """FP32 input should fall through to the Python path without error."""
        BS, seq_len, num_heads, head_dim = 1, 4, 2, 64
        rope = self._make_rope(head_dim)

        q = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(BS, num_heads, seq_len, head_dim, dtype=torch.float32)
        cos = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.float32)
        sin = torch.randn(BS, seq_len, head_dim // 2, dtype=torch.float32)

        q_out, k_out = rope.apply_rotary_emb(
            q, k, cos, sin, unsqueeze_dim=1, is_neox_style=True
        )

        q_ref = _python_rope_reference(q, cos, sin, unsqueeze_dim=1)
        k_ref = _python_rope_reference(k, cos, sin, unsqueeze_dim=1)

        self.assertEqual(q_out, q_ref, atol=1e-5, rtol=1e-5)
        self.assertEqual(k_out, k_ref, atol=1e-5, rtol=1e-5)
