# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Numerical parity between the JIT (AVX-512 C++) and NATIVE Gated DeltaNet
backends.

The model frontend is backend-agnostic, so the only contract the C++ kernels
have to honour is the one the NATIVE backend defines: same outputs, same
updated conv/recurrent state, and the same in-place semantics for the decode
paths. Prefill is compared with a looser tolerance because the NATIVE scan is
chunked (chunk_size=64, zero-padded) while the kernel walks the sequence token
by token.

Covered:
  - decode recurrence, with and without grouped-query heads
  - prefill scan, within and beyond one NATIVE chunk
  - conv1d decode update (kernel_size 4 fast path and the generic path)
  - conv1d prefill, including continuation from a cached state and the
    model's strided input layout
  - gated RMSNorm
  - non-contiguous conv/recurrent state (the kernels write via raw pointers)
  - fake (torch.compile) registrations of every op, via torch.library.opcheck
"""

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from pace.ops import RMSNormGated
from pace.ops.enum import BackendType, DataType
from pace.ops.gated_deltanet import GatedDeltaAttention

_HAS_JIT_OPS = hasattr(torch.ops.pace, "recurrent_gated_delta_rule")

# bf16 in / fp32 accumulate: the two backends reassociate differently, so
# compare with bf16-level tolerances rather than exact equality.
_ATOL = 5e-2
_RTOL = 5e-2


def _gdn(backend: BackendType) -> GatedDeltaAttention:
    return GatedDeltaAttention(backend_impl=backend, dtype=DataType.BFLOAT16)


def _inputs(batch, seq_len, num_k_heads, num_groups, head_k_dim, head_v_dim):
    num_v_heads = num_k_heads * num_groups
    return dict(
        query=torch.randn(
            batch, seq_len, num_k_heads, head_k_dim, dtype=torch.bfloat16
        ),
        key=torch.randn(batch, seq_len, num_k_heads, head_k_dim, dtype=torch.bfloat16),
        value=torch.randn(
            batch, seq_len, num_v_heads, head_v_dim, dtype=torch.bfloat16
        ),
        # g is a log-decay (<= 0) and beta a sigmoid output, matching the model.
        g=-torch.rand(batch, seq_len, num_v_heads) * 0.5,
        beta=torch.rand(batch, seq_len, num_v_heads, dtype=torch.bfloat16),
    )


@unittest.skipUnless(_HAS_JIT_OPS, "pace Gated DeltaNet C++ ops not built")
class TestGatedDeltaRecurrenceParity(TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def _decode_case(self, batch, num_k_heads, num_groups, head_k_dim, head_v_dim):
        num_v_heads = num_k_heads * num_groups
        args = _inputs(batch, 1, num_k_heads, num_groups, head_k_dim, head_v_dim)
        state = (
            torch.randn(batch, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
            * 0.1
        )

        out_native, state_native = _gdn(BackendType.NATIVE).recurrence(
            **args, state=state.clone(), num_groups=num_groups
        )
        out_jit, state_jit = _gdn(BackendType.JIT).recurrence(
            **args, state=state.clone(), num_groups=num_groups
        )

        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        self.assertEqual(
            state_jit.float(), state_native.float(), atol=_ATOL, rtol=_RTOL
        )

    def test_decode_without_gqa(self):
        self._decode_case(
            batch=2, num_k_heads=4, num_groups=1, head_k_dim=128, head_v_dim=128
        )

    def test_decode_with_gqa(self):
        # 16 key/query heads broadcast over 32 value heads (Qwen3.5 shape).
        self._decode_case(
            batch=2, num_k_heads=16, num_groups=2, head_k_dim=128, head_v_dim=128
        )

    def test_decode_head_dims_not_multiple_of_16(self):
        # Exercises the scalar tails of the AVX-512 loops.
        self._decode_case(
            batch=2, num_k_heads=8, num_groups=2, head_k_dim=24, head_v_dim=40
        )

    def _prefill_case(self, seq_len):
        args = _inputs(
            batch=2,
            seq_len=seq_len,
            num_k_heads=4,
            num_groups=2,
            head_k_dim=64,
            head_v_dim=64,
        )
        out_native, state_native = _gdn(BackendType.NATIVE).recurrence(
            **args, state=None, num_groups=2
        )
        out_jit, state_jit = _gdn(BackendType.JIT).recurrence(
            **args, state=None, num_groups=2
        )

        self.assertEqual(out_jit.shape, out_native.shape)
        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        self.assertEqual(
            state_jit.float(), state_native.float(), atol=_ATOL, rtol=_RTOL
        )

    def test_prefill_within_one_chunk(self):
        self._prefill_case(seq_len=8)

    def test_prefill_across_chunks(self):
        # Longer than the NATIVE chunk_size of 64.
        self._prefill_case(seq_len=70)

    def test_decode_state_is_updated_in_place(self):
        args = _inputs(2, 1, 4, 2, 64, 64)
        state = torch.zeros(2, 8, 64, 64, dtype=torch.float32)
        before = state.clone()

        _, new_state = _gdn(BackendType.JIT).recurrence(
            **args, state=state, num_groups=2
        )

        self.assertIs(new_state, state)
        self.assertFalse(torch.equal(state, before))

    def test_decode_accepts_non_contiguous_state(self):
        args = _inputs(1, 1, 4, 1, 64, 64)
        # A transposed view has the right shape but the wrong strides; the
        # kernel writes through a raw pointer, so the backend must fix this up.
        packed = (torch.randn(1, 4, 64, 64, dtype=torch.float32) * 0.1).transpose(
            -1, -2
        )
        self.assertFalse(packed.is_contiguous())

        out_native, state_native = _gdn(BackendType.NATIVE).recurrence(
            **args, state=packed.clone(), num_groups=1
        )
        out_jit, state_jit = _gdn(BackendType.JIT).recurrence(
            **args, state=packed, num_groups=1
        )

        self.assertTrue(state_jit.is_contiguous())
        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        self.assertEqual(
            state_jit.float(), state_native.float(), atol=_ATOL, rtol=_RTOL
        )


@unittest.skipUnless(_HAS_JIT_OPS, "pace Gated DeltaNet C++ ops not built")
class TestCausalConv1dParity(TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def _decode_case(self, batch, channels, kernel_size):
        x = torch.randn(batch, channels, 1, dtype=torch.bfloat16)
        weight = torch.randn(channels, kernel_size, dtype=torch.bfloat16)
        conv_state = torch.randn(batch, channels, kernel_size - 1, dtype=torch.bfloat16)

        out_native, state_native = _gdn(BackendType.NATIVE).conv1d_update(
            x, conv_state.clone(), weight
        )
        out_jit, state_jit = _gdn(BackendType.JIT).conv1d_update(
            x, conv_state.clone(), weight
        )

        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        self.assertEqual(
            state_jit.float(), state_native.float(), atol=_ATOL, rtol=_RTOL
        )

    def test_decode_kernel_size_4_fast_path(self):
        self._decode_case(batch=2, channels=64, kernel_size=4)

    def test_decode_kernel_size_4_scalar_tail(self):
        # batch * channels is not a multiple of 16.
        self._decode_case(batch=3, channels=10, kernel_size=4)

    def test_decode_generic_kernel_size(self):
        self._decode_case(batch=2, channels=20, kernel_size=3)

    def _prefill_case(self, x, kernel_size, cached):
        batch, channels, _ = x.shape
        weight = torch.randn(channels, kernel_size, dtype=torch.bfloat16)
        conv_state = (
            torch.randn(batch, channels, kernel_size - 1, dtype=torch.bfloat16)
            if cached
            else None
        )

        out_native, state_native = _gdn(BackendType.NATIVE).conv1d_update(
            x, conv_state, weight
        )
        out_jit, state_jit = _gdn(BackendType.JIT).conv1d_update(x, conv_state, weight)

        self.assertEqual(out_jit.shape, out_native.shape)
        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        # The next cache is a copy of input tokens, so it must match exactly.
        self.assertTrue(torch.equal(state_jit, state_native))
        # Decode steps write through the state's raw pointer.
        self.assertTrue(state_jit.is_contiguous())

    def test_prefill_matches_native(self):
        self._prefill_case(
            torch.randn(2, 16, 5, dtype=torch.bfloat16), kernel_size=4, cached=False
        )

    def test_prefill_continuation_from_cached_state(self):
        self._prefill_case(
            torch.randn(2, 40, 7, dtype=torch.bfloat16), kernel_size=4, cached=True
        )

    def test_prefill_model_layout(self):
        # The model passes a [B, C, T] transpose of a wider token-major
        # projection. 40 channels exercise the masked channel tail, K=3 the
        # generic kernel, and 70 tokens more than one token block.
        proj = torch.randn(2, 70, 40 + 24, dtype=torch.bfloat16)
        self._prefill_case(proj[..., :40].transpose(1, 2), kernel_size=3, cached=False)

    def test_prefill_shorter_than_kernel(self):
        # The next cache is left-padded with zeros.
        self._prefill_case(
            torch.randn(1, 16, 2, dtype=torch.bfloat16), kernel_size=4, cached=False
        )

    def test_prefill_very_negative_input(self):
        # SiLU of conv sums far below -88, where exp(-x) is not representable.
        x = torch.tensor([-1e3, -1e30, -3e38], dtype=torch.bfloat16)[:, None, None]
        x = x.expand(-1, 40, 6).contiguous()
        weight = torch.full((40, 4), 0.25, dtype=torch.bfloat16)

        out_native, _ = _gdn(BackendType.NATIVE).conv1d_update(x, None, weight)
        out_jit, _ = _gdn(BackendType.JIT).conv1d_update(x, None, weight)

        self.assertEqual(out_jit.float(), out_native.float(), atol=1e-6, rtol=0)

    def test_decode_state_is_updated_in_place(self):
        x = torch.randn(1, 32, 1, dtype=torch.bfloat16)
        weight = torch.randn(32, 4, dtype=torch.bfloat16)
        conv_state = torch.randn(1, 32, 3, dtype=torch.bfloat16)
        before = conv_state.clone()

        _, new_state = _gdn(BackendType.JIT).conv1d_update(x, conv_state, weight)

        self.assertIs(new_state, conv_state)
        self.assertFalse(torch.equal(conv_state, before))

    def test_decode_accepts_non_contiguous_state(self):
        x = torch.randn(1, 8, 1, dtype=torch.bfloat16)
        weight = torch.randn(8, 4, dtype=torch.bfloat16)
        packed = torch.randn(1, 3, 8, dtype=torch.bfloat16).transpose(-1, -2)
        self.assertFalse(packed.is_contiguous())

        out_native, state_native = _gdn(BackendType.NATIVE).conv1d_update(
            x, packed.clone(), weight
        )
        out_jit, state_jit = _gdn(BackendType.JIT).conv1d_update(x, packed, weight)

        self.assertTrue(state_jit.is_contiguous())
        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)
        self.assertEqual(
            state_jit.float(), state_native.float(), atol=_ATOL, rtol=_RTOL
        )


@unittest.skipUnless(
    _HAS_JIT_OPS and hasattr(torch.ops.pace, "rmsnorm_gated"),
    "pace rmsnorm_gated C++ op not built",
)
class TestRMSNormGatedParity(TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def _norm(self, backend: BackendType, dim: int) -> RMSNormGated:
        norm = RMSNormGated(
            dim, eps=1e-6, dtype=DataType.BFLOAT16, backend_impl=backend
        )
        with torch.no_grad():
            norm.weight.copy_(torch.randn(dim, dtype=torch.bfloat16))
        return norm

    def _run_case(self, rows, dim):
        x = torch.randn(rows, dim, dtype=torch.bfloat16)
        gate = torch.randn(rows, dim, dtype=torch.bfloat16)

        native = self._norm(BackendType.NATIVE, dim)
        jit = self._norm(BackendType.JIT, dim)
        with torch.no_grad():
            jit.weight.copy_(native.weight)

        out_native = native(x, gate)
        out_jit = jit(x, gate)

        self.assertEqual(out_jit.float(), out_native.float(), atol=_ATOL, rtol=_RTOL)

    def test_aligned_dim(self):
        self._run_case(rows=32, dim=128)

    def test_single_row(self):
        self._run_case(rows=1, dim=128)

    def test_dim_not_multiple_of_16(self):
        self._run_case(rows=8, dim=40)

    def test_gate_is_required(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        for backend in (BackendType.NATIVE, BackendType.JIT):
            with self.assertRaises(TypeError):
                self._norm(backend, 64)(x)

    def test_very_negative_gate(self):
        # exp(-gate) is not representable below gate = -88; SiLU must still go
        # to zero there. 40 columns cover both the vector lanes and the tail.
        x = torch.randn(6, 40, dtype=torch.bfloat16)
        gate = torch.tensor(
            [-90.0, -150.0, -1e3, -1e5, -1e30, -3e38], dtype=torch.bfloat16
        )[:, None].expand(-1, 40)

        native = self._norm(BackendType.NATIVE, 40)
        jit = self._norm(BackendType.JIT, 40)
        with torch.no_grad():
            jit.weight.copy_(native.weight)

        self.assertEqual(
            jit(x, gate).float(), native(x, gate).float(), atol=1e-6, rtol=0
        )


@unittest.skipUnless(_HAS_JIT_OPS, "pace Gated DeltaNet C++ ops not built")
class TestFakeRegistrations(TestCase):
    """torch.compile support: each op's fake (pace/_register_fake.py) must
    agree with the real kernel on output shapes, dtypes and strides, and its
    schema must declare exactly the arguments it mutates."""

    def setUp(self):
        torch.manual_seed(0)

    def _opcheck(self, op, args):
        results = torch.library.opcheck(op, args)
        failed = {name: r for name, r in results.items() if r != "SUCCESS"}
        self.assertFalse(failed, failed)

    def _recurrence_args(self, seq_len):
        args = _inputs(
            2, seq_len, num_k_heads=2, num_groups=2, head_k_dim=32, head_v_dim=16
        )
        state = torch.zeros(2, 4, 32, 16, dtype=torch.float32)
        return (
            args["query"],
            args["key"],
            args["value"],
            args["g"],
            args["beta"],
            state,
        )

    def test_recurrent_gated_delta_rule(self):
        self._opcheck(
            torch.ops.pace.recurrent_gated_delta_rule, self._recurrence_args(1)
        )

    def test_chunk_gated_delta_rule(self):
        self._opcheck(torch.ops.pace.chunk_gated_delta_rule, self._recurrence_args(5))

    def test_causal_conv1d_update(self):
        self._opcheck(
            torch.ops.pace.causal_conv1d_update,
            (
                torch.randn(2, 40, 1, dtype=torch.bfloat16),
                torch.randn(2, 40, 3, dtype=torch.bfloat16),
                torch.randn(40, 4, dtype=torch.bfloat16),
            ),
        )

    def test_causal_conv1d_prefill(self):
        weight = torch.randn(40, 4, dtype=torch.bfloat16)
        proj = torch.randn(2, 5, 48, dtype=torch.bfloat16)
        self._opcheck(
            torch.ops.pace.causal_conv1d_prefill,
            (proj[..., :40].transpose(1, 2), None, weight),
        )
        self._opcheck(
            torch.ops.pace.causal_conv1d_prefill,
            (
                torch.randn(2, 40, 5, dtype=torch.bfloat16),
                torch.randn(2, 40, 3, dtype=torch.bfloat16),
                weight,
            ),
        )

    def test_rmsnorm_gated(self):
        self._opcheck(
            torch.ops.pace.rmsnorm_gated,
            (
                torch.randn(6, 40, dtype=torch.bfloat16),
                torch.randn(6, 40, dtype=torch.bfloat16),
                torch.randn(40, dtype=torch.bfloat16),
                1e-6,
            ),
        )


if __name__ == "__main__":
    run_tests()
