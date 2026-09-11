# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the per-expert MoE TPP weight-packing helper `tpp_prepack_moe`."""

from __future__ import annotations

import unittest

import torch

from pace_vllm.model_executor.layers.utils import tpp_prepack, tpp_prepack_moe


class TestTppPrepackMoe(unittest.TestCase):
    def test_packs_valid_3d_shape(self) -> None:
        weight = torch.zeros(8, 256, 128, dtype=torch.bfloat16)
        packed = tpp_prepack_moe(weight, block_size=32)
        self.assertIsNotNone(packed)
        self.assertEqual(packed.shape, (8, 256 // 32, 128 // 64, 32, 32, 2))
        self.assertEqual(packed.dtype, torch.bfloat16)
        self.assertTrue(packed.is_contiguous())
        self.assertEqual(packed.numel(), weight.numel())

    def test_matches_stacked_2d_prepack(self) -> None:
        weight = torch.randn(4, 128, 192, dtype=torch.bfloat16)
        packed = tpp_prepack_moe(weight, block_size=32)
        expected = torch.stack(
            [tpp_prepack(weight[i], block_size=32) for i in range(weight.shape[0])]
        )
        self.assertEqual(packed.shape, expected.shape)
        self.assertTrue(torch.equal(packed, expected))

    def test_returns_none_for_non_bf16(self) -> None:
        weight = torch.zeros(8, 256, 128, dtype=torch.float32)
        self.assertIsNone(tpp_prepack_moe(weight, block_size=32))

    def test_returns_none_for_non_3d(self) -> None:
        weight = torch.zeros(256, 128, dtype=torch.bfloat16)
        self.assertIsNone(tpp_prepack_moe(weight, block_size=32))

    def test_returns_none_for_meta_tensor(self) -> None:
        weight = torch.empty(8, 256, 128, dtype=torch.bfloat16, device="meta")
        self.assertIsNone(tpp_prepack_moe(weight, block_size=32))

    def test_rejects_out_features_not_divisible_by_block_size(self) -> None:
        weight = torch.zeros(8, 100, 128, dtype=torch.bfloat16)
        self.assertIsNone(tpp_prepack_moe(weight, block_size=32))

    def test_explicit_block_size_argument(self) -> None:
        weight = torch.zeros(4, 128, 128, dtype=torch.bfloat16)
        packed = tpp_prepack_moe(weight, block_size=64)
        self.assertIsNotNone(packed)
        self.assertEqual(packed.shape, (4, 128 // 64, 128 // 64, 32, 64, 2))


if __name__ == "__main__":
    unittest.main()
