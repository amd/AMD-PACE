# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

from pace.llm.step.budget import CHANNEL_ALIGNMENT
from pace.llm.step.split import SplitFFN, SplitState
from pace.utils.device import gpu_available
from pace.utils.logging import suppress_logging_cls

D_MODEL = 128
CHANNELS = 256


class GatedMLP(nn.Module):
    def __init__(self, d_model=D_MODEL, channels=CHANNELS):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, channels, bias=False)
        self.up_proj = nn.Linear(d_model, channels, bias=False)
        self.down_proj = nn.Linear(channels, d_model, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def make_split(gpu_channels):
    block = GatedMLP().to(torch.bfloat16)
    return SplitFFN(
        block,
        gpu_channels,
        SplitState(),
        torch.device("cpu"),
        "silu",
        residual_backend="pace",
    )


@suppress_logging_cls()
class TestEvictionReachesCompletion(TestCase):
    """
    Within a request eviction is one-way, driven by a KV cache that only grows,
    so it has to be able to run all the way to attention-only on the GPU rather
    than stalling.
    """

    def test_repeated_small_eviction_reaches_zero(self):
        split = make_split(CHANNELS)
        seen = set()
        steps = 0
        while split.gpu_channels > 0:
            self.assertNotIn(split.gpu_channels, seen, "eviction stalled")
            seen.add(split.gpu_channels)
            moved = split.evict(max(1, int(split.gpu_channels * 0.10)))
            self.assertGreater(moved, 0, "evict moved nothing but channels remain")
            steps += 1
            self.assertLess(steps, 100, "eviction is not converging")
        self.assertEqual(split.gpu_channels, 0)
        self.assertEqual(split.residual.channels, CHANNELS)

    def test_residual_stays_packable_throughout(self):
        split = make_split(CHANNELS)
        while split.gpu_channels > 0:
            split.evict(max(1, int(split.gpu_channels * 0.10)))
            self.assertEqual(split.residual.channels % CHANNEL_ALIGNMENT, 0)
            self.assertTrue(split.residual.fused)

    def test_a_request_below_one_block_still_makes_progress(self):
        split = make_split(CHANNELS)
        moved = split.evict(1)
        self.assertEqual(moved, CHANNEL_ALIGNMENT)
        self.assertEqual(split.gpu_channels, CHANNELS - CHANNEL_ALIGNMENT)

    def test_evicting_everything_at_once_is_allowed(self):
        split = make_split(CHANNELS)
        self.assertEqual(split.evict(CHANNELS), CHANNELS)
        self.assertEqual(split.gpu_channels, 0)
        self.assertEqual(split.residual.channels, CHANNELS)

    def test_an_unaligned_remainder_can_still_be_cleared(self):
        # 200 is not a multiple of the alignment, so the last move must take all of it.
        split = make_split(200)
        while split.gpu_channels > 0:
            self.assertGreater(split.evict(max(1, split.gpu_channels // 10)), 0)
        self.assertEqual(split.gpu_channels, 0)

    def test_evicting_an_empty_prefix_moves_nothing(self):
        split = make_split(CHANNELS)
        split.evict(CHANNELS)
        self.assertEqual(split.evict(64), 0)

    def test_oversized_request_is_clamped(self):
        split = make_split(CHANNELS)
        self.assertEqual(split.evict(CHANNELS * 10), CHANNELS)


def owns_only_itself(tensor):
    return tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


@suppress_logging_cls()
class TestMemoryIsReleased(TestCase):
    """Eviction exists to free VRAM, so no part of a split may hold the full weight behind a view."""

    def test_eviction_releases_the_evicted_rows(self):
        split = make_split(CHANNELS)
        split.evict(128)
        for name in ("gate_proj", "up_proj", "down_proj"):
            self.assertTrue(owns_only_itself(getattr(split.block, name).weight), name)

    def test_the_split_keeps_no_view_of_the_original_weights(self):
        split = SplitFFN(
            GatedMLP().to(torch.bfloat16),
            192,
            SplitState(),
            torch.device("cpu"),
            "silu",
            residual_backend="torch",
        )
        prefix = [
            getattr(split.block, name).weight
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        for weight in prefix + list(split.residual.weights_2d()):
            self.assertTrue(owns_only_itself(weight))


def full_output(split, hidden):
    return split.block(hidden).cpu() + split.residual.forward(hidden.cpu())


@suppress_logging_cls()
class TestRestore(TestCase):
    """Restoring runs between requests, once the KV cache that forced an eviction is gone."""

    def test_restoring_undoes_an_eviction(self):
        split = make_split(192)
        weights = [
            split.block.gate_proj.weight.clone(),
            split.block.up_proj.weight.clone(),
            split.block.down_proj.weight.clone(),
        ]
        split.evict(64)
        self.assertEqual(split.restore(64), 64)
        self.assertEqual(split.gpu_channels, 192)
        self.assertEqual(split.residual.channels, CHANNELS - 192)
        self.assertEqual(split.block.gate_proj.weight, weights[0])
        self.assertEqual(split.block.up_proj.weight, weights[1])
        self.assertEqual(split.block.down_proj.weight, weights[2])

    def test_the_function_is_unchanged_throughout(self):
        split = make_split(192)
        hidden = torch.randn(1, 3, D_MODEL, dtype=torch.bfloat16)
        before = full_output(split, hidden)
        split.evict(128)
        split.restore(128)
        self.assertEqual(full_output(split, hidden), before)

    def test_the_residual_stays_aligned(self):
        split = make_split(128)
        self.assertEqual(split.restore(100), CHANNEL_ALIGNMENT)
        self.assertEqual(split.residual.channels % CHANNEL_ALIGNMENT, 0)
        self.assertTrue(split.residual.fused)

    def test_restoring_everything_clears_the_residual(self):
        split = make_split(128)
        self.assertEqual(split.restore(CHANNELS), CHANNELS - 128)
        self.assertEqual(split.gpu_channels, CHANNELS)
        self.assertEqual(split.residual.channels, 0)

    def test_nothing_on_the_cpu_means_nothing_moves(self):
        split = make_split(CHANNELS)
        self.assertEqual(split.restore(CHANNEL_ALIGNMENT), 0)
        self.assertEqual(split.gpu_channels, CHANNELS)

    def test_a_torch_residual_restores_exactly(self):
        # 56 residual channels is not a multiple of the alignment, which only the packed kernel needs.
        split = SplitFFN(
            GatedMLP().to(torch.bfloat16),
            200,
            SplitState(),
            torch.device("cpu"),
            "silu",
            residual_backend="torch",
        )
        split.evict(64)
        self.assertEqual(split.restore(64), 64)
        self.assertEqual(split.gpu_channels, 200)


@suppress_logging_cls()
class TestSplitConstruction(TestCase):

    def test_prefix_and_residual_partition_the_channels(self):
        split = make_split(192)
        self.assertEqual(split.gpu_channels, 192)
        self.assertEqual(split.residual.channels, CHANNELS - 192)
        self.assertEqual(split.total_channels, CHANNELS)

    def test_a_ratio_outside_the_range_is_rejected(self):
        block = GatedMLP().to(torch.bfloat16)
        with self.assertRaises(AssertionError):
            SplitFFN(block, CHANNELS + 1, SplitState(), torch.device("cpu"), "silu")

    def test_a_biased_projection_is_rejected(self):
        for name, shape in (
            ("gate_proj", (D_MODEL, CHANNELS)),
            ("up_proj", (D_MODEL, CHANNELS)),
            ("down_proj", (CHANNELS, D_MODEL)),
        ):
            block = GatedMLP()
            setattr(block, name, nn.Linear(*shape, bias=True))
            with self.assertRaises(AssertionError):
                SplitFFN(
                    block.to(torch.bfloat16),
                    128,
                    SplitState(),
                    torch.device("cpu"),
                    "silu",
                )


@unittest.skipUnless(gpu_available(), "co-execution needs a GPU")
@suppress_logging_cls()
class TestSharedStaging(TestCase):

    def test_layers_stage_through_one_pair_of_pinned_buffers(self):
        device = torch.device("cuda")
        state = SplitState()
        splits = []
        for _ in range(3):
            split = SplitFFN(
                GatedMLP().to(torch.bfloat16),
                192,
                state,
                device,
                "silu",
                residual_backend="torch",
            )
            split.block.to(device)
            splits.append(split)

        hidden = torch.randn(1, 40, D_MODEL, dtype=torch.bfloat16, device=device)
        for split in splits:
            expected = full_output(split, hidden)
            hidden = split(hidden)
            self.assertEqual(hidden.cpu(), expected)
        self.assertEqual(state.staging._storage.numel(), 40 * D_MODEL)
        self.assertEqual(state.result_staging._storage.numel(), 40 * D_MODEL)


@unittest.skipUnless(gpu_available(), "the prefix goes to a GPU")
@suppress_logging_cls()
class TestPlacement(TestCase):

    def test_splitting_sends_the_prefix_to_the_gpu_and_keeps_the_residual(self):
        block = GatedMLP().to(torch.bfloat16)
        gate, up, down = (
            block.gate_proj.weight.clone(),
            block.up_proj.weight.clone(),
            block.down_proj.weight.clone(),
        )
        split = SplitFFN(
            block,
            192,
            SplitState(),
            torch.device("cuda"),
            "silu",
            residual_backend="torch",
        )

        prefix = split.block
        self.assertTrue(
            all(
                getattr(prefix, name).weight.is_cuda
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        )
        residual = split.residual
        self.assertTrue(
            all(
                w.device.type == "cpu"
                for w in (residual.gate, residual.up, residual.down)
            )
        )
        self.assertEqual(
            torch.cat([prefix.gate_proj.weight.cpu(), residual.gate]), gate
        )
        self.assertEqual(torch.cat([prefix.up_proj.weight.cpu(), residual.up]), up)
        self.assertEqual(
            torch.cat([prefix.down_proj.weight.cpu(), residual.down], dim=1), down
        )
