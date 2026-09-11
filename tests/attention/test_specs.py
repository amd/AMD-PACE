# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
Tests for per-layer KV cache specs: compute_layer_kv_specs, LayerKVSpec and
resolve_text_config. These specs are the single source of truth for per-layer
KV geometry and KV sharing, consumed by every cache backend and by models
(e.g. Gemma4), so the config -> spec derivation is pinned down here.
"""

from types import SimpleNamespace

from torch.testing._internal.common_utils import TestCase

import pace  # noqa: F401

from pace.llm.attention.specs import (
    LayerKVSpec,
    compute_layer_kv_specs,
    resolve_text_config,
)


def _config(**kwargs):
    """Minimal HF-like text config: 4 layers, 8 q_heads, 2 kv_heads, dim 64."""
    base = dict(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=512,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestResolveTextConfig(TestCase):
    def test_plain_config_returned_unchanged(self):
        cfg = _config()
        self.assertIs(resolve_text_config(cfg), cfg)

    def test_multimodal_wrapper_unwrapped(self):
        text = _config(num_hidden_layers=3)
        wrapper = SimpleNamespace(text_config=text, num_hidden_layers=99)
        self.assertIs(resolve_text_config(wrapper), text)
        # Specs must follow the text config, not the wrapper's layer count.
        self.assertEqual(len(compute_layer_kv_specs(wrapper)), 3)


class TestUniformModels(TestCase):
    def test_uniform_model_has_identical_specs_and_no_sharing(self):
        specs = compute_layer_kv_specs(_config())
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs, [LayerKVSpec(2, 64, 0, None)] * 4)
        self.assertFalse(any(s.is_shared for s in specs))

    def test_head_dim_derived_from_hidden_size_when_absent(self):
        cfg = _config(hidden_size=512, num_attention_heads=8)
        del cfg.head_dim
        self.assertTrue(all(s.head_dim == 64 for s in compute_layer_kv_specs(cfg)))

    def test_kv_heads_default_to_query_heads_when_absent(self):
        cfg = _config()
        del cfg.num_key_value_heads
        self.assertTrue(all(s.num_kv_heads == 8 for s in compute_layer_kv_specs(cfg)))

    def test_sliding_window_ignored_without_layer_types(self):
        # No layer_types => every layer is full attention, so no window applies.
        specs = compute_layer_kv_specs(_config(sliding_window=128))
        self.assertTrue(all(s.sliding_window == 0 for s in specs))


class TestPerLayerVariation(TestCase):
    def _mixed_config(self, **kwargs):
        return _config(
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=128,
            **kwargs,
        )

    def test_sliding_window_applies_only_to_sliding_layers(self):
        specs = compute_layer_kv_specs(self._mixed_config())
        self.assertEqual([s.sliding_window for s in specs], [128, 0, 128, 0])

    def test_global_head_dim_applies_only_to_full_attention_layers(self):
        specs = compute_layer_kv_specs(self._mixed_config(global_head_dim=256))
        self.assertEqual([s.head_dim for s in specs], [64, 256, 64, 256])

    def test_k_eq_v_kv_heads_apply_only_to_full_attention_layers(self):
        specs = compute_layer_kv_specs(
            self._mixed_config(attention_k_eq_v=True, num_global_key_value_heads=4)
        )
        self.assertEqual([s.num_kv_heads for s in specs], [2, 4, 2, 4])

    def test_k_eq_v_without_global_kv_heads_falls_back_to_default(self):
        specs = compute_layer_kv_specs(self._mixed_config(attention_k_eq_v=True))
        self.assertEqual([s.num_kv_heads for s in specs], [2, 2, 2, 2])


class TestKVSharing(TestCase):
    def _shared_config(self, num_shared, num_layers=4):
        types = ["sliding_attention", "full_attention"] * (num_layers // 2)
        return _config(
            num_hidden_layers=num_layers,
            layer_types=types,
            sliding_window=128,
            num_kv_shared_layers=num_shared,
        )

    def test_trailing_layers_share_last_layer_of_same_type(self):
        specs = compute_layer_kv_specs(self._shared_config(2))
        self.assertEqual([s.kv_sharing_target for s in specs], [None, None, 0, 1])
        self.assertEqual([s.is_shared for s in specs], [False, False, True, True])

    def test_sharing_target_is_the_last_non_shared_layer_of_that_type(self):
        # 6 layers, 2 shared: layers 4,5 reuse the newest sliding/full pair (2,3).
        specs = compute_layer_kv_specs(self._shared_config(2, num_layers=6))
        self.assertEqual(
            [s.kv_sharing_target for s in specs], [None, None, None, None, 2, 3]
        )

    def test_shared_layers_keep_the_geometry_of_their_type(self):
        cfg = self._shared_config(2)
        cfg.global_head_dim = 256
        specs = compute_layer_kv_specs(cfg)
        # A shared layer reports the dims of the layer it reads from.
        self.assertEqual([s.head_dim for s in specs], [64, 256, 64, 256])
        for spec in specs:
            if spec.is_shared:
                self.assertEqual(spec.head_dim, specs[spec.kv_sharing_target].head_dim)
                self.assertEqual(
                    spec.num_kv_heads, specs[spec.kv_sharing_target].num_kv_heads
                )

    def test_zero_or_missing_shared_layers_means_no_sharing(self):
        for cfg in (self._shared_config(0), self._config_without_shared_field()):
            specs = compute_layer_kv_specs(cfg)
            self.assertTrue(all(s.kv_sharing_target is None for s in specs))

    def _config_without_shared_field(self):
        cfg = self._shared_config(2)
        del cfg.num_kv_shared_layers
        return cfg

    def test_all_layers_shared_is_rejected(self):
        # num_kv_shared_layers == num_hidden_layers leaves no source layer.
        with self.assertRaisesRegex(ValueError, "num_kv_shared_layers"):
            compute_layer_kv_specs(self._shared_config(4))

    def test_shared_layer_without_an_earlier_layer_of_its_type_is_rejected(self):
        cfg = _config(
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            sliding_window=128,
            num_kv_shared_layers=1,
        )
        with self.assertRaisesRegex(ValueError, "full_attention"):
            compute_layer_kv_specs(cfg)


class TestGemma4Shape(TestCase):
    """The exact per-layer shape this feature was added for."""

    def test_gemma4_like_config(self):
        cfg = _config(
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            global_head_dim=128,
            sliding_window=16,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
        )
        self.assertEqual(
            compute_layer_kv_specs(cfg),
            [
                LayerKVSpec(2, 64, 16, None),
                LayerKVSpec(2, 128, 0, None),
                LayerKVSpec(2, 64, 16, 0),
                LayerKVSpec(2, 128, 0, 1),
            ],
        )


class TestLayerKVSpec(TestCase):
    def test_defaults_and_is_shared(self):
        spec = LayerKVSpec(num_kv_heads=4, head_dim=128)
        self.assertEqual((spec.sliding_window, spec.kv_sharing_target), (0, None))
        self.assertFalse(spec.is_shared)
        self.assertTrue(LayerKVSpec(4, 128, 0, 2).is_shared)

    def test_spec_is_hashable_and_frozen(self):
        spec = LayerKVSpec(4, 128)
        self.assertEqual(len({spec, LayerKVSpec(4, 128)}), 1)
        with self.assertRaises(Exception):
            spec.head_dim = 64


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
