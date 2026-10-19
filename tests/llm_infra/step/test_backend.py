# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import itertools
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import torch
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.testing._internal.common_utils import TestCase
from transformers import (
    AutoModelForCausalLM,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from pace.llm import LLMModel, SamplingConfig
from pace.llm.attention import AttentionBackendType, KVCacheType
from pace.llm.configs import LLMOperatorType, OperatorConfig, PardSpecDecodeConfig
from pace.llm.generation import GenerationRequest, TerminationReason
from pace.llm.generation.config import resolve_generation_backend_config
from pace.llm.models import hf_utils
from pace.llm.ops import LLMBackendType
from pace.llm.step import StepGenerationBackend, StepGenerationBackendConfig, StepStats
from pace.utils.device import gpu_available
from pace.utils.logging import suppress_logging_cls

VOCAB = 128
PROMPT = torch.tensor([[2, 5, 9, 17, 33, 64, 3, 71]])


@suppress_logging_cls()
class TestStepStats(TestCase):

    def test_tokens_per_step_divides_committed_rather_than_returned_tokens(self):
        stats = StepStats(steps=4, committed=12, generated=10)
        self.assertEqual(stats.tokens_per_step, 3.0)

    def test_an_empty_run_reports_zero_rather_than_dividing(self):
        stats = StepStats()
        self.assertEqual(stats.tokens_per_step, 0.0)
        self.assertEqual(stats.decode_tokens_per_second, 0.0)

    def test_decode_throughput_leaves_out_the_prefill_token(self):
        self.assertEqual(
            StepStats(generated=11, decode_seconds=2.0).decode_tokens_per_second, 5.0
        )

    def test_describe_reports_fallback_and_eviction_only_when_they_happened(self):
        quiet = StepStats(steps=2, committed=6, generated=6).describe()
        self.assertNotIn("plain steps", quiet)
        self.assertNotIn("evictions", quiet)
        busy = StepStats(steps=3, committed=7, plain_steps=1, evictions=2).describe()
        self.assertIn("1 plain steps", busy)
        self.assertIn("2 evictions", busy)


@suppress_logging_cls()
class TestStepGenerationBackendConfig(TestCase):

    def test_nonsense_is_rejected(self):
        for override in (
            {"draft_ratio": 0.0},
            {"draft_ratio": 1.5},
            {"tree_width": 0},
            {"tree_depth": 0},
            {"tree_mode": "bushy"},
            {"residual_backend": "cuda"},
            {"speculation": "maybe"},
            {"min_tokens_per_step": 0.5},
            {"speculation_window": 0},
            {"draft_kv_window": 2},
            {"vram_budget_gb": 0.0},
            {"vram_budget_gb": -1.0},
            {"max_seq_length": 0},
            {"cpu_threads": 0},
            {"gpu_index": -1},
            {"temperature": 0.6},
        ):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                StepGenerationBackendConfig(**override)

    def test_pace_kernels_drive_alignment(self):
        config = StepGenerationBackendConfig()
        self.assertTrue(config.uses_pace_kernels(torch.bfloat16))
        self.assertFalse(config.uses_pace_kernels(torch.float32))
        self.assertFalse(
            StepGenerationBackendConfig(residual_backend="torch").uses_pace_kernels(
                torch.bfloat16
            )
        )

    def test_a_pace_residual_needs_bfloat16(self):
        pace = StepGenerationBackendConfig(residual_backend="pace")
        pace.check_residual_backend(torch.bfloat16)
        StepGenerationBackendConfig(residual_backend="auto").check_residual_backend(
            torch.float32
        )
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype), self.assertRaises(AssertionError):
                pace.check_residual_backend(dtype)

    def test_server_json_resolves_through_the_shared_registry(self):
        config = resolve_generation_backend_config(
            {"type": "step", "draft_ratio": 0.6, "tree_mode": "full"}
        )
        self.assertIsInstance(config, StepGenerationBackendConfig)
        self.assertEqual(config.draft_ratio, 0.6)
        self.assertEqual(config.model_dump(mode="json")["type"], "step")


def server_operator_config():
    """The operator config the engine builds when no op_config is given."""
    return OperatorConfig(
        **{
            LLMOperatorType.Norm: LLMBackendType.NATIVE,
            LLMOperatorType.QKVProjection: LLMBackendType.TPP,
            LLMOperatorType.Attention: AttentionBackendType.SLAB,
            LLMOperatorType.OutProjection: LLMBackendType.TPP,
            LLMOperatorType.MLP: LLMBackendType.TPP,
            LLMOperatorType.LMHead: LLMBackendType.NATIVE,
        }
    )


@suppress_logging_cls()
class TestSharedSettings(TestCase):
    """Every setting `LLMModel` and the server pass is either accepted or refused, never ignored."""

    def _check(self, **overrides):
        settings = dict(
            serving=False,
            dtype=torch.bfloat16,
            kv_cache_type=KVCacheType.SLAB_POOL,
            spec_config=None,
            opconfig=None,
            cache_kwargs=None,
        )
        settings.update(overrides)
        StepGenerationBackendConfig.check_shared_settings(**settings)

    def test_the_offline_defaults_are_accepted(self):
        self._check()

    def test_the_default_layout_for_each_dtype_is_accepted(self):
        for dtype, layout in (
            (torch.bfloat16, KVCacheType.SLAB_POOL),
            (torch.float32, KVCacheType.PAGED),
            (torch.float32, None),
        ):
            with self.subTest(dtype=dtype, layout=layout):
                self._check(dtype=dtype, kv_cache_type=layout)

    def test_what_the_server_passes_is_accepted(self):
        self._check(
            serving=True,
            opconfig=server_operator_config(),
            cache_kwargs={"dtype": torch.bfloat16},
        )

    def test_settings_with_nothing_to_act_on_are_refused(self):
        for override in (
            {"spec_config": PardSpecDecodeConfig(model_name_or_path="draft")},
            {"kv_cache_type": KVCacheType.PAGED},
            {"kv_cache_type": KVCacheType.DYNAMIC},
            {"dtype": torch.float32, "kv_cache_type": KVCacheType.SLAB_POOL},
            {"opconfig": OperatorConfig()},
            {"cache_kwargs": {"kv_cache_memory_gb": 4.0}},
            {"cache_kwargs": {"dtype": torch.float16}},
            {"eviction_config": object()},
        ):
            with self.subTest(override=override), self.assertRaises(AssertionError):
                self._check(**override)


def finalized(**settings):
    config = SamplingConfig(max_new_tokens=8, eos_token_id=[1], **settings)
    config.finalize()
    return config


@suppress_logging_cls()
class TestSamplingChecks(TestCase):

    def test_settings_applied_per_position_are_accepted(self):
        for settings in (
            {"do_sample": False},
            {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05},
            {"temperature": 0.6, "seed": 3},
            {"do_sample": False, "return_logprobs": True, "return_probs": True},
        ):
            with self.subTest(settings=settings):
                StepGenerationBackend._check_sampling(finalized(**settings))

    def test_settings_that_depend_on_each_candidates_history_are_refused(self):
        for settings in (
            {"repetition_penalty": 1.05},
            {"frequency_penalty": 0.5},
            {"min_new_tokens": 4},
            {"return_input_logprobs": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(AssertionError):
                StepGenerationBackend._check_sampling(finalized(**settings))

    def test_only_settings_the_request_passed_survive(self):
        config = SamplingConfig(
            repetition_penalty=1.1, frequency_penalty=0.5, min_new_tokens=4
        )
        dropped = StepGenerationBackend._drop_inherited_settings(
            config, frozenset({"frequency_penalty"})
        )
        self.assertEqual(
            dropped, [("repetition_penalty", 1.1, 1.0), ("min_new_tokens", 4, 0)]
        )
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertEqual(config.min_new_tokens, 0)
        self.assertEqual(config.frequency_penalty, 0.5)


def greedy(max_new_tokens=32, **settings):
    return SamplingConfig(max_new_tokens=max_new_tokens, do_sample=False, **settings)


def word_tokenizer():
    vocab = {f"t{index}": index for index in range(VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="t0",
        pad_token="t0",
        eos_token="t1",
        bos_token="t2",
    )


def huggingface_greedy(directory, max_new_tokens=32):
    reference = AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float32
    ).cuda()
    generated = reference.generate(
        PROMPT.cuda(), max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=0
    )
    del reference
    torch.cuda.empty_cache()
    return generated[0, PROMPT.shape[1] :].cpu().tolist()


def sliding_window_models():
    """Random models whose sliding layers see 8 tokens, so generating after the 8-token prompt soon runs past the window."""
    small = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=512,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    alternating = ["sliding_attention", "full_attention"] * 2
    return {
        "multimodal gemma4": lambda: Gemma4ForConditionalGeneration(
            Gemma4Config(
                text_config=Gemma4TextConfig(
                    **small,
                    head_dim=16,
                    global_head_dim=32,
                    attention_k_eq_v=True,
                    num_global_key_value_heads=1,
                    sliding_window=8,
                    layer_types=alternating,
                    num_kv_shared_layers=0,
                    hidden_size_per_layer_input=0,
                    use_bidirectional_attention="vision",
                )
            )
        ),
        "qwen3": lambda: Qwen3ForCausalLM(
            Qwen3Config(
                **small,
                head_dim=16,
                use_sliding_window=True,
                sliding_window=8,
                layer_types=alternating,
            )
        ),
        "mistral": lambda: MistralForCausalLM(MistralConfig(**small, sliding_window=8)),
    }


class RecordingStreamer:
    def __init__(self):
        self.puts = []

    def put(self, value):
        self.puts.append(value.reshape(-1).tolist())

    def end(self):
        pass


@unittest.skipUnless(gpu_available(), "STeP needs a GPU")
@suppress_logging_cls()
class TestStepGeneration(TestCase):
    """
    End to end on a tiny random Llama, in float32 so the split's reassociation cannot flip a greedy token.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.directory = tempfile.mkdtemp()
        torch.manual_seed(0)
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=VOCAB,
            max_position_embeddings=512,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        LlamaForCausalLM(config).save_pretrained(cls.directory)
        word_tokenizer().save_pretrained(cls.directory)
        cls.expected = huggingface_greedy(cls.directory)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)
        super().tearDownClass()

    def _model(self, **settings):
        settings.setdefault("residual_backend", "torch")
        settings.setdefault("max_seq_length", 256)
        return LLMModel(
            self.directory,
            dtype=torch.float32,
            generation_backend_config=StepGenerationBackendConfig(**settings),
        )

    def _generated(self, model, sampling=None, streamer=None):
        output = model.generate(PROMPT, sampling or greedy(), text_streamer=streamer)
        return output, output.output_token_ids[0, PROMPT.shape[1] :].tolist()

    def test_greedy_output_matches_huggingface(self):
        for settings in (
            {"draft_ratio": 0.5, "speculation": "on"},
            {"draft_ratio": 0.5, "speculation": "off"},
            {"draft_ratio": 1.0, "speculation": "on", "tree_mode": "full"},
        ):
            with self.subTest(settings=settings):
                _, generated = self._generated(self._model(**settings))
                self.assertEqual(generated, self.expected)

    def test_a_hub_id_loads_without_the_native_model_check(self):
        with (
            tempfile.TemporaryDirectory() as cache,
            patch.multiple(
                hf_utils, snapshot_download=DEFAULT, hf_hub_download=DEFAULT
            ) as hub,
            patch.multiple(hf_utils.constants, HF_HUB_CACHE=cache, HF_HUB_OFFLINE=True),
        ):
            hub["snapshot_download"].return_value = self.directory
            model = LLMModel(
                "someone/any-architecture",
                dtype=torch.float32,
                generation_backend_config=StepGenerationBackendConfig(
                    draft_ratio=0.5, residual_backend="torch", max_seq_length=256
                ),
            )
        hub["hf_hub_download"].assert_not_called()
        _, generated = self._generated(model)
        self.assertEqual(generated, self.expected)

    def test_the_whole_ffn_on_the_gpu_accepts_every_draft_token(self):
        output, _ = self._generated(
            self._model(draft_ratio=1.0, speculation="on", tree_depth=3)
        )
        stats = output.speculative_stats
        self.assertGreater(stats.steps, 0)
        self.assertEqual(stats.accepted_per_step, 3.0)
        self.assertEqual(stats.tokens_per_step, 4.0)
        self.assertEqual(stats.mean_accepted_tokens, 4.0)
        self.assertEqual(stats.generated, len(self.expected))

    def test_each_round_reaches_the_streamer_in_one_put(self):
        streamer = RecordingStreamer()
        output, generated = self._generated(
            self._model(draft_ratio=1.0, speculation="on", tree_depth=3),
            streamer=streamer,
        )
        prompt, first, *rounds = streamer.puts
        self.assertEqual(prompt, PROMPT.reshape(-1).tolist())
        self.assertEqual(first + [t for tokens in rounds for t in tokens], generated)
        self.assertEqual(len(rounds), output.speculative_stats.steps)
        # With the whole FFN on the GPU the draft is the full model, so every round before the last, which the length limit may cut, is three accepted tokens and the bonus.
        for tokens in rounds[:-1]:
            self.assertEqual(len(tokens), 4)

    def test_eviction_keeps_the_output_lossless(self):
        model = self._model(draft_ratio=0.5, speculation="on")
        backend = model._backend
        # Prefill reads as empty, the first round one byte over the budget, and the rest as empty again.
        readings = itertools.chain(
            [0, backend.plan.budget_bytes + 1], itertools.repeat(0)
        )
        with patch(
            "torch.cuda.memory_allocated", side_effect=lambda *_: next(readings)
        ):
            output, generated = self._generated(model)
        self.assertEqual(output.speculative_stats.evictions, 1)
        self.assertLess(backend.blocks[0].gpu_channels, backend.split.gpu_channels)
        self.assertEqual(generated, self.expected)

    def test_a_prompt_makes_room_before_its_caches_are_built(self):
        model = self._model(draft_ratio=0.5, speculation="on")
        backend = model._backend
        # One byte under the budget, so only the prompt's caches can push it over.
        with patch(
            "torch.cuda.memory_allocated", return_value=backend.plan.budget_bytes - 1
        ):
            output, generated = self._generated(model)
        self.assertEqual(output.speculative_stats.evictions, 1)
        self.assertLess(backend.blocks[0].gpu_channels, backend.split.gpu_channels)
        self.assertEqual(generated, self.expected)

    def test_an_eviction_makes_room_for_the_sequence_to_double(self):
        backend = self._model(
            draft_ratio=0.5, speculation="off", max_seq_length=64
        )._backend
        block_bytes = backend._channel_bytes * 64
        # Short of max_seq_length the room is for max_seq_length; past it, for twice the length rather than all of max_new_tokens.
        for cached, max_new_tokens, room in ((40, 32, 64), (100, 1000, 200)):
            with self.subTest(cached=cached):
                backend.reset()
                state = SimpleNamespace(
                    prompt_length=8,
                    sampling_config=SimpleNamespace(max_new_tokens=max_new_tokens),
                    backend_state=SimpleNamespace(speculating=False, stats=StepStats()),
                )
                growth = (room - cached) * backend._kv_bytes_per_token
                moved = -(-(growth + 1) // block_bytes) * 64
                self.assertLess(moved, backend.split.gpu_channels)
                with patch(
                    "torch.cuda.memory_allocated",
                    return_value=backend.plan.budget_bytes + 1,
                ):
                    backend._maybe_evict(state, cached)
                self.assertEqual(state.backend_state.stats.evictions, 1)
                self.assertEqual(
                    backend.blocks[0].gpu_channels, backend.split.gpu_channels - moved
                )

    def test_a_prefill_eviction_counts_the_activations_each_block_frees(self):
        backend = self._model(draft_ratio=0.5, speculation="off")._backend
        block_bytes = backend._channel_bytes * 64
        state = SimpleNamespace(
            prompt_length=256,
            sampling_config=SimpleNamespace(max_new_tokens=1),
            backend_state=SimpleNamespace(speculating=False, stats=StepStats()),
        )
        # In this model a block at 256 tokens frees as many activation bytes as weight bytes, so one block clears an excess of one and a half.
        with patch(
            "torch.cuda.memory_allocated", return_value=backend.plan.budget_bytes
        ):
            backend._maybe_evict(state, 256, transient_bytes=block_bytes * 3 // 2)
        self.assertEqual(
            backend.blocks[0].gpu_channels, backend.split.gpu_channels - 64
        )

    def test_an_explicit_budget_is_capped_at_what_the_device_can_give(self):
        backend = self._model(draft_ratio=0.5, vram_budget_gb=1e6)._backend
        self.assertLess(backend.plan.budget_bytes, torch.cuda.mem_get_info()[1])

    def test_reset_restores_the_planned_split(self):
        model = self._model(draft_ratio=0.5, speculation="on")
        backend = model._backend
        planned = backend.split.gpu_channels
        for block in backend.blocks:
            block.evict(64)
        self.assertEqual(backend.blocks[0].gpu_channels, planned - 64)
        backend.reset()
        self.assertEqual(
            [block.gpu_channels for block in backend.blocks],
            [planned] * len(backend.blocks),
        )
        _, generated = self._generated(model)
        self.assertEqual(generated, self.expected)

    def test_reset_between_requests_starts_each_from_the_plan(self):
        model = self._model(
            draft_ratio=0.5, speculation="on", reset_between_requests=True
        )
        backend = model._backend
        for block in backend.blocks:
            block.evict(64)
        self._generated(model)
        self.assertEqual(backend.blocks[0].gpu_channels, backend.split.gpu_channels)

    def test_the_auto_fallback_persists_until_reset(self):
        # No round can reach five tokens with a depth of three, so auto always gives up.
        model = self._model(
            draft_ratio=0.5,
            speculation="auto",
            min_tokens_per_step=5.0,
            speculation_window=1,
        )
        backend = model._backend
        first, generated = self._generated(model)
        self.assertIsNotNone(first.speculative_stats.speculation_stopped_at)
        self.assertTrue(backend.speculation_abandoned)
        self.assertEqual(generated, self.expected)

        second, _ = self._generated(model)
        self.assertEqual(
            second.speculative_stats.plain_steps, second.speculative_stats.steps
        )

        backend.reset()
        third, _ = self._generated(model)
        self.assertIsNotNone(third.speculative_stats.speculation_stopped_at)

    def test_a_batch_is_refused(self):
        with self.assertRaises(AssertionError):
            self._model(draft_ratio=0.5).generate(torch.cat([PROMPT, PROMPT]), greedy())

    def test_a_tree_wider_than_the_vocabulary_is_refused(self):
        with self.assertRaises(AssertionError):
            self._model(draft_ratio=0.5, tree_width=VOCAB + 1)

    def test_a_pace_residual_in_float32_is_refused_before_loading(self):
        with patch("pace.llm.step.backend.load_model") as load:
            with self.assertRaises(AssertionError):
                self._model(residual_backend="pace")
        load.assert_not_called()

    def test_a_second_request_waits_for_the_first(self):
        backend = self._model(draft_ratio=0.5)._backend

        def request(name):
            return backend.create_request(
                GenerationRequest(
                    request_id=name, input_ids=PROMPT, sampling_config=greedy()
                )
            )

        first = request("first")
        with self.assertRaises(AssertionError):
            request("second")
        backend.release(first, TerminationReason.CANCELLED)
        backend.release(request("second"), TerminationReason.CANCELLED)

    def test_unsupported_sampling_is_refused_before_anything_runs(self):
        model = self._model(draft_ratio=0.5)
        with self.assertRaises(AssertionError):
            model.generate(PROMPT, greedy(repetition_penalty=1.1))
        _, generated = self._generated(model)
        self.assertEqual(generated, self.expected)

    def test_a_penalty_the_checkpoint_sets_is_switched_off_with_one_warning(self):
        directory = tempfile.mkdtemp()
        try:
            shutil.copytree(self.directory, directory, dirs_exist_ok=True)
            GenerationConfig(
                bos_token_id=2, eos_token_id=1, pad_token_id=0, repetition_penalty=1.1
            ).save_pretrained(directory)
            model = LLMModel(
                directory,
                dtype=torch.float32,
                generation_backend_config=StepGenerationBackendConfig(
                    draft_ratio=0.5,
                    speculation="on",
                    residual_backend="torch",
                    max_seq_length=256,
                ),
            )
            with patch("pace.llm.step.backend.PACE_LLM_WARNING") as warning:
                for _ in range(2):
                    _, generated = self._generated(model)
                    self.assertEqual(generated, self.expected)
            self.assertEqual(warning.call_count, 1)
            self.assertIn("repetition_penalty=1.1", warning.call_args.args[0])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_seeded_sampling_is_reproducible(self):
        model = self._model(draft_ratio=0.5, speculation="on")
        sampling = SamplingConfig(
            max_new_tokens=24, do_sample=True, temperature=0.8, seed=7
        )
        _, first = self._generated(model, sampling)
        _, second = self._generated(model, sampling)
        self.assertEqual(first, second)

    def test_logprobs_cover_every_generated_token(self):
        model = self._model(draft_ratio=0.5, speculation="on")
        output, generated = self._generated(model, greedy(return_logprobs=True))
        self.assertEqual(output.logprobs.shape, (1, len(generated) * VOCAB))
        rows = output.logprobs.reshape(len(generated), VOCAB)
        self.assertEqual(rows.argmax(dim=-1).tolist(), generated)

    def test_a_trimmed_round_leaves_the_last_kept_token_for_the_next(self):
        backend = self._model(draft_ratio=1.0, speculation="on")._backend
        state = backend.create_request(
            GenerationRequest(
                request_id="trimmed",
                input_ids=PROMPT,
                sampling_config=greedy(max_new_tokens=3),
            )
        )
        state.defer_release = True
        backend.prefill([state])
        backend.decode([state])
        request = state.backend_state
        self.assertEqual(state.generated_count, 3)
        self.assertEqual(request.verify_seen, state.token_count - 1)
        self.assertEqual(request.verify_cache.get_seq_length(), state.token_count - 1)
        backend.release(state, TerminationReason.COMPLETED)


@unittest.skipUnless(gpu_available(), "STeP needs a GPU")
@suppress_logging_cls()
class TestSlidingWindowGeneration(TestCase):
    """
    End to end on tiny random sliding-window models in float32, generating well past the window.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.directory = tempfile.mkdtemp()
        cls.models = {}
        for index, (name, build) in enumerate(sliding_window_models().items()):
            path = f"{cls.directory}/{index}"
            torch.manual_seed(0)
            build().save_pretrained(path)
            word_tokenizer().save_pretrained(path)
            cls.models[name] = (path, huggingface_greedy(path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)
        super().tearDownClass()

    def _generated(self, path, **settings):
        model = LLMModel(
            path,
            dtype=torch.float32,
            generation_backend_config=StepGenerationBackendConfig(
                residual_backend="torch", max_seq_length=256, **settings
            ),
        )
        output = model.generate(PROMPT, greedy())
        return output, output.output_token_ids[0, PROMPT.shape[1] :].tolist()

    def test_greedy_output_matches_huggingface(self):
        for name, (path, expected) in self.models.items():
            for settings in (
                {"draft_ratio": 0.5, "speculation": "on"},
                {"draft_ratio": 0.5, "speculation": "off"},
                {
                    "draft_ratio": 0.5,
                    "speculation": "on",
                    "draft_kv_window": 12,
                    "draft_kv_sink": 2,
                },
            ):
                with self.subTest(model=name, settings=settings):
                    _, generated = self._generated(path, **settings)
                    self.assertEqual(generated, expected)

    def test_the_whole_ffn_on_the_gpu_accepts_every_draft_token(self):
        for name, (path, expected) in self.models.items():
            with self.subTest(model=name):
                output, generated = self._generated(
                    path, draft_ratio=1.0, speculation="on", tree_depth=3
                )
                self.assertEqual(generated, expected)
                self.assertEqual(output.speculative_stats.accepted_per_step, 3.0)
