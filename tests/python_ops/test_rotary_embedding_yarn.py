# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""YaRN cos/sin cache tests pinned against transformers.

The scaling factor has two possible sources: the checkpoint's explicit
`factor`, and the ratio between the extended and pretrained context lengths.
When those two disagree, only the explicit `factor` is correct -- that is what
transformers uses, and the ratio is a fallback for checkpoints that omit
`factor` altogether (DeepSeek-V3 expresses the scaling that way).

Published checkpoints exist for both shapes, so both are covered here:
  * Yarn-Llama-2-13b-128k: factor 32, 4096 -> 131072, ratio 32   (agree)
  * Yarn-Mistral-7b-64k:   factor 8,  8192 ->  32768, ratio 4    (disagree)
"""

import math

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_yarn_parameters

import pace  # noqa: F401
from pace.ops.rotary_embedding import _compute_yarn_cos_sin_cache

HEAD_DIM = 128
BASE = 10000.0


def _transformers_cos_sin(max_position_embeddings, rope_scaling):
    """Ground-truth cos/sin cache built from transformers' own YaRN parameters."""
    config = LlamaConfig(
        hidden_size=HEAD_DIM * 8,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=max_position_embeddings,
    )
    config.head_dim = HEAD_DIM
    config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": BASE,
        **rope_scaling,
    }
    inv_freq, attention_factor = _compute_yarn_parameters(config)

    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.float())
    return freqs.cos() * attention_factor, freqs.sin() * attention_factor


def _pace_cos_sin(max_position_embeddings, rope_scaling):
    return _compute_yarn_cos_sin_cache(
        BASE, HEAD_DIM, max_position_embeddings, dict(rope_scaling)
    )


class TestYarnCosSinCache(TestCase):
    """PACE's YaRN cache must agree with transformers for every factor source."""

    def _assert_matches_transformers(self, max_position_embeddings, rope_scaling):
        expected_cos, expected_sin = _transformers_cos_sin(
            max_position_embeddings, rope_scaling
        )
        actual_cos, actual_sin = _pace_cos_sin(max_position_embeddings, rope_scaling)

        rows = min(expected_cos.shape[0], actual_cos.shape[0])
        cols = min(expected_cos.shape[-1], actual_cos.shape[-1])
        self.assertEqual(actual_cos[:rows, :cols].float(), expected_cos[:rows, :cols])
        self.assertEqual(actual_sin[:rows, :cols].float(), expected_sin[:rows, :cols])

    def test_factor_agrees_with_context_ratio(self):
        """Yarn-Llama-2-13b-128k shape: both sources give 32, so either works."""
        self._assert_matches_transformers(
            131072, {"factor": 32.0, "original_max_position_embeddings": 4096}
        )

    def test_factor_disagrees_with_context_ratio(self):
        """Yarn-Mistral-7b-64k shape: factor is 8 while the ratio is only 4."""
        self._assert_matches_transformers(
            32768, {"factor": 8.0, "original_max_position_embeddings": 8192}
        )

    def test_factor_larger_than_context_ratio(self):
        """Yarn-Mistral-7b-128k shape: factor is 16 while the ratio is still 4."""
        self._assert_matches_transformers(
            32768, {"factor": 16.0, "original_max_position_embeddings": 8192}
        )

    def test_factor_null_falls_back_to_context_ratio(self):
        """DeepSeek-V3 shape: no factor declared, so the ratio is the only source."""
        self._assert_matches_transformers(
            163840, {"factor": None, "original_max_position_embeddings": 4096}
        )

    def test_missing_original_max_position_embeddings(self):
        """With no pretrained length to compare against, `factor` stands alone."""
        self._assert_matches_transformers(
            32768, {"factor": 8.0, "original_max_position_embeddings": 32768}
        )

    def test_explicit_factor_is_not_overridden_by_ratio(self):
        """Regression guard: recomputing `factor` from the ratio changes numerics.

        This is the failure mode the cache had before -- the ratio silently won
        whenever `original_max_position_embeddings` was present, so a
        Yarn-Mistral checkpoint was scaled by 4 instead of its declared 8.
        """
        max_position_embeddings = 32768
        declared = {"factor": 8.0, "original_max_position_embeddings": 8192}
        ratio = max_position_embeddings / declared["original_max_position_embeddings"]
        self.assertNotEqual(declared["factor"], ratio)

        cos_declared, _ = _pace_cos_sin(max_position_embeddings, declared)
        cos_from_ratio, _ = _pace_cos_sin(
            max_position_embeddings,
            {"factor": ratio, "original_max_position_embeddings": 8192},
        )

        # The two must differ, otherwise `factor` is being ignored.
        self.assertFalse(torch.allclose(cos_declared.float(), cos_from_ratio.float()))

    def test_attention_factor_tracks_declared_factor(self):
        """The mscale attention factor is derived from `factor`, not the ratio."""
        max_position_embeddings = 32768
        rope_scaling = {"factor": 8.0, "original_max_position_embeddings": 8192}
        cos, _ = _pace_cos_sin(max_position_embeddings, rope_scaling)

        # get_mscale(8.0) applied at position 0, where every cos term is 1.
        expected = 0.1 * math.log(8.0) + 1.0
        self.assertAlmostEqual(cos[0].float().max().item(), expected, places=5)


if __name__ == "__main__":
    run_tests()
