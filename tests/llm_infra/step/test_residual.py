# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

from pace.ops.backends.tpp import _pack_weight_for_block_size
from pace.llm.step.residual import ResidualFFN, unpack_weight
from pace.utils.logging import suppress_logging_cls

D_MODEL = 256
CHANNELS = 128


def weights(channels=CHANNELS, d_model=D_MODEL, dtype=torch.bfloat16):
    torch.manual_seed(0)
    gate = torch.randn(channels, d_model, dtype=dtype) * 0.02
    up = torch.randn(channels, d_model, dtype=dtype) * 0.02
    down = torch.randn(d_model, channels, dtype=dtype) * 0.02
    return gate, up, down


def reference(hidden, gate, up, down, activation=F.silu):
    hidden = hidden.float()
    inter = activation(F.linear(hidden, gate.float())) * F.linear(hidden, up.float())
    return F.linear(inter, down.float())


def relative_error(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@suppress_logging_cls()
class TestWeightPacking(TestCase):

    def test_round_trip_is_exact(self):
        gate, _, down = weights()
        for weight in (gate, down):
            packed = _pack_weight_for_block_size(weight, 32)
            self.assertIsNotNone(packed)
            self.assertEqual(packed.dim(), 5)
            self.assertTrue(torch.equal(unpack_weight(packed), weight))

    def test_unpack_rejects_unpacked_input(self):
        gate, _, _ = weights()
        with self.assertRaises(AssertionError):
            unpack_weight(gate)


@suppress_logging_cls()
class TestResidualFFN(TestCase):

    def test_pace_and_torch_agree_with_reference(self):
        gate, up, down = weights()
        hidden = torch.randn(1, 7, D_MODEL, dtype=torch.bfloat16) * 0.5
        want = reference(hidden, gate, up, down)
        for backend in ("pace", "torch"):
            ffn = ResidualFFN(gate, up, down, "silu", backend=backend)
            self.assertLess(relative_error(ffn.forward(hidden), want), 5e-2)

    def test_aligned_widths_take_the_fused_path(self):
        gate, up, down = weights()
        self.assertTrue(ResidualFFN(gate, up, down, "silu", backend="pace").fused)

    def test_non_bfloat16_falls_back_to_torch(self):
        gate, up, down = weights(dtype=torch.float32)
        ffn = ResidualFFN(gate, up, down, "silu", backend="auto")
        self.assertFalse(ffn.use_kernels)
        hidden = torch.randn(1, 4, D_MODEL, dtype=torch.float32)
        want = reference(hidden, gate, up, down)
        self.assertLess(relative_error(ffn.forward(hidden), want), 1e-5)

    def test_pace_backend_rejects_non_bfloat16(self):
        gate, up, down = weights(dtype=torch.float32)
        with self.assertRaises(AssertionError):
            ResidualFFN(gate, up, down, "silu", backend="pace")

    def test_mismatched_shapes_are_rejected(self):
        gate, up, down = weights()
        with self.assertRaises(AssertionError):
            ResidualFFN(gate, up, down[:, :64], "silu")
        with self.assertRaises(AssertionError):
            ResidualFFN(gate[:64], up, down, "silu")

    def test_an_activation_with_no_implementation_is_rejected_at_construction(self):
        gate, up, down = weights()
        for backend in ("auto", "torch", "pace"):
            with self.assertRaises(AssertionError):
                ResidualFFN(gate, up, down, "quick_gelu", backend=backend)

    def test_empty_residual_contributes_nothing(self):
        gate, up, down = weights(channels=0)
        ffn = ResidualFFN(gate, up, down, "silu", backend="torch")
        hidden = torch.randn(1, 3, D_MODEL, dtype=torch.bfloat16)
        self.assertEqual(ffn.channels, 0)
        self.assertTrue(torch.equal(ffn.forward(hidden), torch.zeros_like(hidden)))

    def test_gelu_activation_is_supported(self):
        gate, up, down = weights()
        hidden = torch.randn(1, 5, D_MODEL, dtype=torch.bfloat16) * 0.5
        want = reference(hidden, gate, up, down, activation=F.gelu)
        ffn = ResidualFFN(gate, up, down, "gelu_pytorch_tanh", backend="pace")
        self.assertEqual(ffn.activation, "gelu")
        self.assertLess(relative_error(ffn.forward(hidden), want), 5e-2)


@suppress_logging_cls()
class TestResidualEviction(TestCase):

    def test_absorbing_channels_reconstructs_the_whole_residual(self):
        gate, up, down = weights()
        hidden = torch.randn(1, 6, D_MODEL, dtype=torch.bfloat16) * 0.5
        want = reference(hidden, gate, up, down)

        split = 64
        ffn = ResidualFFN(gate[split:], up[split:], down[:, split:], "silu", "pace")
        self.assertEqual(ffn.channels, CHANNELS - split)
        ffn.absorb(gate[:split], up[:split], down[:, :split])
        self.assertEqual(ffn.channels, CHANNELS)
        self.assertLess(relative_error(ffn.forward(hidden), want), 5e-2)

    def test_absorbing_keeps_weights_recoverable(self):
        gate, up, down = weights()
        ffn = ResidualFFN(gate[64:], up[64:], down[:, 64:], "silu", "pace")
        ffn.absorb(gate[:64], up[:64], down[:, :64])
        got_gate, got_up, got_down = ffn.weights_2d()
        self.assertTrue(torch.equal(got_gate, gate))
        self.assertTrue(torch.equal(got_up, up))
        self.assertTrue(torch.equal(got_down, down))

    def test_absorb_requires_a_gate_when_one_exists(self):
        gate, up, down = weights()
        ffn = ResidualFFN(gate[64:], up[64:], down[:, 64:], "silu", "pace")
        with self.assertRaises(AssertionError):
            ffn.absorb(None, up[:64], down[:, :64])


@suppress_logging_cls()
class TestResidualRelease(TestCase):

    def test_release_hands_back_the_leading_channels(self):
        gate, up, down = weights()
        ffn = ResidualFFN(gate, up, down, "silu", "pace")
        got_gate, got_up, got_down = ffn.release(64)
        self.assertTrue(torch.equal(got_gate, gate[:64]))
        self.assertTrue(torch.equal(got_up, up[:64]))
        self.assertTrue(torch.equal(got_down, down[:, :64]))
        self.assertEqual(ffn.channels, CHANNELS - 64)
        kept_gate, kept_up, kept_down = ffn.weights_2d()
        self.assertTrue(torch.equal(kept_gate, gate[64:]))
        self.assertTrue(torch.equal(kept_up, up[64:]))
        self.assertTrue(torch.equal(kept_down, down[:, 64:]))

    def test_release_undoes_absorb(self):
        gate, up, down = weights()
        ffn = ResidualFFN(gate[64:], up[64:], down[:, 64:], "silu", "pace")
        ffn.absorb(gate[:64], up[:64], down[:, :64])
        ffn.release(64)
        kept_gate, _, _ = ffn.weights_2d()
        self.assertTrue(torch.equal(kept_gate, gate[64:]))
        self.assertTrue(ffn.fused)

    def test_the_remainder_still_computes_its_share(self):
        gate, up, down = weights()
        hidden = torch.randn(1, 6, D_MODEL, dtype=torch.bfloat16) * 0.5
        ffn = ResidualFFN(gate, up, down, "silu", "pace")
        ffn.release(64)
        want = reference(hidden, gate[64:], up[64:], down[:, 64:])
        self.assertLess(relative_error(ffn.forward(hidden), want), 5e-2)

    def test_the_kept_rows_do_not_hold_the_released_ones(self):
        gate, up, down = weights(dtype=torch.float32)
        ffn = ResidualFFN(gate, up, down, "silu", "torch")
        ffn.release(64)
        for weight in ffn.weights_2d():
            self.assertEqual(
                weight.untyped_storage().nbytes(),
                weight.numel() * weight.element_size(),
            )

    def test_releasing_more_than_is_held_is_rejected(self):
        gate, up, down = weights()
        ffn = ResidualFFN(gate, up, down, "silu", "pace")
        with self.assertRaises(AssertionError):
            ffn.release(CHANNELS + 1)
