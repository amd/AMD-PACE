# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import TestCase
from transformers import (
    DynamicCache,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from pace.llm.step.budget import (
    CHANNEL_ALIGNMENT,
    GIB,
    VRAM_RESERVE,
    align_channels,
    available_vram_bytes,
    kv_bytes_per_token,
    model_memory,
    plan_split,
    prefill_activation_bytes,
    ratio_from_budget,
    resolve_split,
)
from pace.utils.logging import suppress_logging_cls

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    max_position_embeddings=64,
)


def small_config(**overrides):
    settings = dict(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
        tie_word_embeddings=True,
    )
    settings.update(overrides)
    return LlamaConfig(**settings)


def gemma4_config(**overrides):
    """Full-attention layers with twice the head width and as many KV heads as the sliding ones."""
    settings = dict(
        TINY,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        num_global_key_value_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"] * 2,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
    )
    settings.update(overrides)
    return Gemma4TextConfig(**settings)


@suppress_logging_cls()
class TestModelMemory(TestCase):

    def test_components_sum_to_total(self):
        memory = model_memory(small_config(), max_seq_length=4096, workspace=GIB)
        self.assertEqual(
            memory.total,
            memory.attention
            + memory.embedding
            + memory.norm
            + memory.kv_cache
            + memory.activations
            + memory.workspace
            + memory.ffn
            + memory.ffn_activations,
        )
        self.assertEqual(memory.fixed, memory.total - memory.ffn_cost)

    def test_ffn_matches_three_projections(self):
        config = small_config()
        memory = model_memory(config, max_seq_length=1024)
        expected = (
            config.num_hidden_layers
            * 3
            * config.hidden_size
            * config.intermediate_size
            * 2
        )
        self.assertEqual(memory.ffn, expected)

    def test_untied_embeddings_cost_double(self):
        tied = model_memory(small_config(tie_word_embeddings=True), 1024)
        untied = model_memory(small_config(tie_word_embeddings=False), 1024)
        self.assertEqual(untied.embedding, 2 * tied.embedding)

    def test_kv_cache_scales_with_sequence_length(self):
        short = model_memory(small_config(), 1024)
        long = model_memory(small_config(), 4096)
        self.assertEqual(long.kv_cache, 4 * short.kv_cache)
        self.assertEqual(long.ffn, short.ffn)

    def test_float32_doubles_bfloat16(self):
        half = model_memory(small_config(), 1024, dtype=torch.bfloat16)
        full = model_memory(small_config(), 1024, dtype=torch.float32)
        self.assertEqual(full.total, 2 * half.total)

    def test_a_draft_cache_doubles_the_reservation(self):
        one = model_memory(small_config(), 1024)
        two = model_memory(small_config(), 1024, kv_caches=2)
        self.assertEqual(two.kv_cache, 2 * one.kv_cache)
        self.assertEqual(two.ffn, one.ffn)

    def test_unnamed_weights_count_as_fixed(self):
        memory = model_memory(small_config(), 1024)
        named = memory.fixed
        memory.other = 3 * GIB
        self.assertEqual(memory.fixed, named + 3 * GIB)
        self.assertEqual(memory.total, memory.fixed + memory.ffn_cost)

    def test_prefill_activations_follow_the_sequence_and_the_gpu_share(self):
        config = small_config()
        short, long = model_memory(config, 1024), model_memory(config, 4096)
        self.assertEqual(long.activations, 4 * short.activations)
        self.assertEqual(long.ffn_activations, 4 * short.ffn_activations)
        # The FFN's part is the gate, up and product over every channel.
        self.assertEqual(short.ffn_activations, 1024 * 3 * 8192 * 2)
        self.assertEqual(
            prefill_activation_bytes(config, torch.bfloat16, 1024, 4096),
            short.activations + short.ffn_activations // 2,
        )

    def test_describe_names_unnamed_weights_only_when_present(self):
        memory = model_memory(small_config(), 1024)
        self.assertNotIn("other", memory.describe())
        memory.other = GIB
        self.assertIn("other 1.00 GiB", memory.describe())

    @torch.no_grad()
    def test_kv_bytes_match_what_a_cache_holds(self):
        tokens = 6
        for name, model in (
            ("llama", LlamaForCausalLM(LlamaConfig(**TINY))),
            ("gemma4", Gemma4ForCausalLM(gemma4_config())),
            (
                "gemma4 sharing keys and values",
                Gemma4ForCausalLM(gemma4_config(num_kv_shared_layers=2)),
            ),
        ):
            with self.subTest(model=name):
                cache = DynamicCache()
                model(
                    input_ids=torch.arange(3, 3 + tokens).unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                )
                held = sum(
                    tensor.numel() * tensor.element_size()
                    for layer in cache.layers
                    for tensor in (layer.keys, layer.values)
                )
                self.assertEqual(
                    kv_bytes_per_token(model.config, torch.float32), held // tokens
                )

    def test_the_widest_attention_layer_sets_the_prefill_peak(self):
        # One token: the hidden states, then 4 query and 2 KV heads at the full-attention layers' width of 32.
        self.assertEqual(
            prefill_activation_bytes(gemma4_config(), torch.bfloat16, 1, 0),
            (7 * 64 + 3 * (4 + 2) * 32) * 2,
        )


@suppress_logging_cls()
class TestRatioFromBudget(TestCase):

    def test_budget_covering_everything_gives_one(self):
        memory = model_memory(small_config(), 4096)
        self.assertEqual(ratio_from_budget(memory, memory.total * 2), 1.0)

    def test_budget_below_fixed_cost_gives_zero(self):
        memory = model_memory(small_config(), 4096)
        self.assertEqual(ratio_from_budget(memory, memory.fixed // 2), 0.0)

    def test_half_the_ffn_gives_half(self):
        memory = model_memory(small_config(), 4096)
        budget = memory.fixed + memory.ffn_cost // 2
        self.assertAlmostEqual(ratio_from_budget(memory, budget), 0.5, places=5)

    def test_unnamed_weights_take_room_from_the_prefix(self):
        memory = model_memory(small_config(), 4096)
        budget = memory.total
        self.assertEqual(ratio_from_budget(memory, budget), 1.0)
        memory.other = memory.ffn_cost // 2
        self.assertAlmostEqual(ratio_from_budget(memory, budget), 0.5, places=5)


@suppress_logging_cls()
class TestAlignChannels(TestCase):

    def test_residual_is_aligned(self):
        total = 8192
        for desired in (1, 63, 100, 4095, 6000, 8191):
            residual = total - align_channels(desired, total)
            self.assertEqual(residual % CHANNEL_ALIGNMENT, 0)

    def test_rounds_down_to_respect_the_budget(self):
        self.assertLessEqual(align_channels(6000, 8192), 6000)

    def test_full_residency_is_preserved(self):
        self.assertEqual(align_channels(8192, 8192), 8192)

    def test_clamps_out_of_range(self):
        self.assertEqual(align_channels(-5, 8192), 0)
        self.assertEqual(align_channels(99999, 8192), 8192)

    def test_tiny_ffn_falls_back_to_no_prefix(self):
        self.assertEqual(align_channels(16, 32), 0)


@suppress_logging_cls()
class TestResolveSplit(TestCase):

    def test_explicit_ratio_is_honoured_after_alignment(self):
        split = resolve_split(8192, 0.75)
        self.assertEqual(split.gpu_channels, 6144)
        self.assertEqual(split.cpu_channels, 2048)
        self.assertTrue(split.packable)
        self.assertTrue(split.aligned)

    def test_derived_ratio_fits_the_budget(self):
        config = small_config()
        memory = model_memory(config, 4096)
        budget = memory.fixed + memory.ffn // 2
        split = resolve_split(8192, None, memory=memory, budget_bytes=budget)
        self.assertLessEqual(split.ratio, 0.5)
        self.assertTrue(split.packable)

    def test_budget_below_fixed_cost_is_rejected(self):
        config = small_config()
        memory = model_memory(config, 4096)
        with self.assertRaises(AssertionError):
            resolve_split(8192, None, memory=memory, budget_bytes=memory.fixed // 2)

    def test_an_ffn_too_narrow_to_align_reports_unpackable(self):
        split = resolve_split(32, 0.5)
        self.assertEqual(split.gpu_channels, 0)
        self.assertFalse(split.packable)

    def test_an_awkward_ratio_still_aligns_the_residual(self):
        split = resolve_split(8192, 0.7312)
        self.assertTrue(split.packable)
        self.assertLessEqual(split.ratio, 0.7312)

    def test_alignment_off_honours_the_exact_ratio(self):
        split = resolve_split(8192, 0.7312, align=False)
        self.assertEqual(split.gpu_channels, int(0.7312 * 8192))
        self.assertFalse(split.aligned)
        self.assertFalse(split.packable)

    def test_alignment_off_still_reports_the_requested_ratio(self):
        split = resolve_split(8192, 0.5, align=False)
        self.assertEqual(split.gpu_channels, 4096)
        self.assertTrue(split.packable)

    def test_an_explicit_ratio_outside_the_range_is_rejected(self):
        for ratio in (-0.1, 0.0, 1.5):
            for align in (True, False):
                with self.subTest(ratio=ratio, align=align):
                    with self.assertRaises(AssertionError):
                        resolve_split(8192, ratio, align=align)

    def test_describe_mentions_both_sides(self):
        text = resolve_split(8192, 0.75).describe()
        self.assertIn("6144", text)
        self.assertIn("2048", text)

    def test_generous_budget_reports_gibibytes(self):
        memory = model_memory(small_config(), 4096)
        self.assertGreater(memory.total / GIB, 0.0)


@suppress_logging_cls()
class TestPlanSplit(TestCase):
    """The single planning path the runtime and the planning tool share."""

    def _plan(self, **overrides):
        # About 2.6 GiB in full, so 1.5 GiB leaves part of the FFN on the CPU.
        settings = dict(
            config=small_config(),
            budget_bytes=int(1.5 * GIB),
            max_seq_length=4096,
            dtype=torch.bfloat16,
            speculation="auto",
        )
        settings.update(overrides)
        return plan_split(**settings)

    def test_the_split_is_solved_against_the_whole_budget(self):
        plan = self._plan()
        expected = resolve_split(
            8192, None, memory=plan.memory, budget_bytes=plan.budget_bytes
        )
        self.assertEqual(plan.split.gpu_channels, expected.gpu_channels)
        self.assertLessEqual(
            plan.memory.fixed + plan.split.ratio * plan.memory.ffn_cost,
            plan.budget_bytes,
        )

    def test_the_workspace_takes_room_from_the_prefix(self):
        measured = self._plan(workspace=GIB // 4)
        self.assertEqual(measured.memory.workspace, GIB // 4)
        self.assertLess(measured.split.gpu_channels, self._plan().split.gpu_channels)

    def test_speculation_reserves_a_draft_cache(self):
        drafting = self._plan(speculation="on")
        plain = self._plan(speculation="off")
        self.assertEqual(drafting.memory.kv_cache, 2 * plain.memory.kv_cache)
        self.assertGreater(plain.split.gpu_channels, drafting.split.gpu_channels)

    def test_loaded_weights_beyond_the_named_components_count_as_fixed(self):
        named = self._plan()
        itemsize = 2
        parameters = (
            named.memory.attention
            + named.memory.embedding
            + named.memory.norm
            + named.memory.ffn
        ) // itemsize + int(0.25 * GIB) // itemsize
        counted = self._plan(parameter_count=parameters)
        self.assertEqual(counted.memory.other, int(0.25 * GIB))
        self.assertLess(counted.split.gpu_channels, named.split.gpu_channels)

    def test_an_explicit_ratio_past_the_budget_is_kept_when_its_weights_fit(self):
        plan = self._plan(draft_ratio=0.25, align=False)
        self.assertGreater(
            plan.memory.fixed + 0.25 * plan.memory.ffn_cost, plan.budget_bytes
        )
        self.assertEqual(plan.split.gpu_channels, 2048)

    def test_an_explicit_ratio_whose_weights_do_not_fit_is_derived_instead(self):
        with patch("pace.llm.step.budget.PACE_LLM_WARNING") as warning:
            plan = self._plan(draft_ratio=0.5)
        self.assertGreater(plan.memory.installed(0.5), plan.budget_bytes)
        self.assertEqual(plan.split.gpu_channels, self._plan().split.gpu_channels)
        self.assertIn("derived from the budget", warning.call_args_list[0].args[0])

    def test_a_budget_below_the_fixed_cost_is_refused_for_any_ratio(self):
        fixed = self._plan().memory.fixed
        for draft_ratio in (None, 0.5):
            with self.subTest(draft_ratio=draft_ratio):
                with self.assertRaises(AssertionError):
                    self._plan(budget_bytes=fixed // 2, draft_ratio=draft_ratio)


@suppress_logging_cls()
class TestAvailableVram(TestCase):

    def test_the_default_budget_is_what_the_process_can_use_minus_the_reserve(self):
        with (
            patch("torch.cuda.mem_get_info", return_value=(20 * GIB, 32 * GIB)),
            patch("torch.cuda.memory_reserved", return_value=2 * GIB),
        ):
            self.assertEqual(available_vram_bytes(0), 22 * GIB - VRAM_RESERVE)
