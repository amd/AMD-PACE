# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import filecmp
import json
import os
import random
import shutil
import tempfile
import threading
import unittest
from unittest.mock import DEFAULT, Mock, patch

import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.testing._internal.common_utils import TestCase
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    MistralConfig,
    PixtralVisionConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from pace.llm.step.calibrate import (
    MAX_SHARD_SIZE,
    calibrate,
    layerwise_saliency,
    permuted_channels,
    save_reordered,
)
from pace.llm.step.config import StepCalibrationConfig
from pace.llm.step.hf_modules import find_gated_mlps
from pace.llm.step.loading import load_model
from pace.llm.step.saliency import SaliencyAccumulator, channel_order
from pace.utils.device import gpu_available
from pace.utils.logging import suppress_logging_cls

D_MODEL = 32
CHANNELS = 16
VOCAB = 128
SMALL = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=VOCAB,
    max_position_embeddings=128,
)
ALTERNATING = ["sliding_attention", "full_attention"] * 2


class GatedMLP(nn.Module):
    def __init__(self, d_model=D_MODEL, channels=CHANNELS):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, channels, bias=False)
        self.up_proj = nn.Linear(d_model, channels, bias=False)
        self.down_proj = nn.Linear(channels, d_model, bias=False)

    def forward(self, hidden):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


class TinyModel(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([GatedMLP() for _ in range(blocks)])

    def forward(self, input_ids=None, hidden=None, **kwargs):
        if hidden is None:
            hidden = input_ids.float()
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return hidden


def gemma4_config(**overrides):
    settings = dict(
        SMALL,
        head_dim=16,
        global_head_dim=16,
        sliding_window=8,
        layer_types=ALTERNATING,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
    )
    settings.update(overrides)
    return Gemma4TextConfig(**settings)


def tiny_decoders():
    """Random models of each decoder family STeP supports, the sliding ones with an 8-token window."""
    vision = PixtralVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
    )
    return {
        "llama": lambda: LlamaForCausalLM(LlamaConfig(**SMALL)),
        "qwen3 with sliding layers": lambda: Qwen3ForCausalLM(
            Qwen3Config(
                **SMALL,
                head_dim=16,
                use_sliding_window=True,
                sliding_window=8,
                layer_types=ALTERNATING,
            )
        ),
        "gemma3": lambda: Gemma3ForCausalLM(
            Gemma3TextConfig(
                **SMALL, head_dim=16, sliding_window=8, layer_types=ALTERNATING
            )
        ),
        "gemma4": lambda: Gemma4ForCausalLM(gemma4_config()),
        "multimodal mistral3": lambda: Mistral3ForConditionalGeneration(
            Mistral3Config(text_config=MistralConfig(**SMALL), vision_config=vision)
        ),
    }


def token_batches(sizes=(2, 2, 2, 1), length=16):
    # Longer than the 8-token sliding window, so sliding layers are masked differently from full ones.
    generator = torch.Generator().manual_seed(0)
    return [
        torch.randint(3, VOCAB, (size, length), generator=generator) for size in sizes
    ]


def whole_model_saliency(model, batches):
    device = next(model.parameters()).device
    with SaliencyAccumulator(model) as accumulator:
        for batch in batches:
            model(input_ids=batch.to(device), use_cache=False)
        return [accumulator.score(index) for index in range(len(accumulator.blocks))]


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


def write_calibration_source(directory):
    """A small saved checkpoint and a corpus of its words, for calibrating end to end."""
    source, corpus = f"{directory}/source", f"{directory}/corpus.jsonl"
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(**SMALL)).save_pretrained(source)
    word_tokenizer().save_pretrained(source)
    words = random.Random(0)
    with open(corpus, "w") as handle:
        for _ in range(64):
            text = " ".join(f"t{words.randrange(3, VOCAB)}" for _ in range(120))
            handle.write(json.dumps({"text": text}) + "\n")
    return source, corpus


def calibration_config(source, output, corpus):
    return StepCalibrationConfig(
        model_name_or_path=source,
        output_dir=output,
        datasets=[corpus],
        num_samples=8,
        sequence_length=32,
        batch_size=4,
    )


@suppress_logging_cls()
class TestSaliency(TestCase):

    def test_accumulator_scores_every_block(self):
        torch.manual_seed(0)
        model = TinyModel()
        with SaliencyAccumulator(model) as accumulator:
            for _ in range(3):
                model(hidden=torch.randn(2, 5, D_MODEL))
            scores = [accumulator.score(index) for index in range(2)]
        for value in scores:
            self.assertEqual(value.shape, (CHANNELS,))
            self.assertTrue(torch.all(value >= 0))

    def test_hooks_are_removed_on_exit(self):
        model = TinyModel()
        with SaliencyAccumulator(model):
            pass
        self.assertEqual(len(model.blocks[0].down_proj._forward_hooks), 0)

    def test_scoring_without_data_is_rejected(self):
        model = TinyModel()
        with SaliencyAccumulator(model) as accumulator:
            with self.assertRaises(AssertionError):
                accumulator.score(0)

    def test_a_model_without_gated_blocks_is_rejected(self):
        with self.assertRaises(AssertionError):
            SaliencyAccumulator(nn.Sequential(nn.Linear(4, 4)))

    def test_dead_channels_score_lowest(self):
        torch.manual_seed(0)
        model = TinyModel(blocks=1)
        # Zeroing a channel's down_proj column removes its contribution entirely.
        with torch.no_grad():
            model.blocks[0].down_proj.weight[:, 3] = 0.0
        with SaliencyAccumulator(model) as accumulator:
            model(hidden=torch.randn(2, 8, D_MODEL))
            score = accumulator.score(0)
        self.assertEqual(int(channel_order(score)[-1]), 3)

    def test_channel_order_is_descending(self):
        scores = torch.tensor([0.1, 0.9, 0.5])
        self.assertEqual(channel_order(scores).tolist(), [1, 2, 0])


def reorder_in_memory(model, orders):
    for index, (_, block) in enumerate(find_gated_mlps(model)):
        for name, columns in (
            ("gate_proj", False),
            ("up_proj", False),
            ("down_proj", True),
        ):
            projection = getattr(block, name)
            projection.weight.data = permuted_channels(
                projection.weight.data, orders[index], columns
            )


def random_orders(model):
    generator = torch.Generator().manual_seed(0)
    return {
        index: torch.randperm(block.up_proj.out_features, generator=generator)
        for index, (_, block) in enumerate(find_gated_mlps(model))
    }


@suppress_logging_cls()
class TestPermutedChannels(TestCase):

    def test_reordering_preserves_the_block_output(self):
        torch.manual_seed(0)
        model = TinyModel(blocks=1)
        hidden = torch.randn(1, 4, D_MODEL)
        before = model(hidden=hidden)
        reorder_in_memory(model, random_orders(model))
        self.assertTrue(torch.allclose(before, model(hidden=hidden), atol=1e-5))


@suppress_logging_cls()
class TestSaveReordered(TestCase):
    """The checkpoint is written one shard at a time, its FFN weights permuted on the way out."""

    def test_the_files_match_reordering_in_memory_then_saving(self):
        # Mistral3 checks the key renaming transformers applies on save.
        for family in ("llama", "multimodal mistral3"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                torch.manual_seed(0)
                tiny_decoders()[family]().save_pretrained(f"{root}/source")
                # Loaded back from safetensors, so the weights are memory-mapped as a real checkpoint's are.
                streamed = load_model(f"{root}/source", torch.float32)
                save_reordered(streamed, random_orders(streamed), f"{root}/streamed")
                in_memory = load_model(f"{root}/source", torch.float32)
                reorder_in_memory(in_memory, random_orders(in_memory))
                in_memory.save_pretrained(
                    f"{root}/in_memory", max_shard_size=MAX_SHARD_SIZE
                )

                files = sorted(os.listdir(f"{root}/in_memory"))
                self.assertEqual(sorted(os.listdir(f"{root}/streamed")), files)
                for name in files:
                    self.assertTrue(
                        filecmp.cmp(
                            f"{root}/in_memory/{name}",
                            f"{root}/streamed/{name}",
                            shallow=False,
                        ),
                        name,
                    )

    def test_nothing_is_written_if_a_weight_would_go_out_unpermuted(self):
        with tempfile.TemporaryDirectory() as root:
            model = LlamaForCausalLM(LlamaConfig(**SMALL))
            # save_pretrained copies a non-contiguous weight before writing it, so the copy is not recognised.
            up = model.model.layers[1].mlp.up_proj
            up.weight.data = up.weight.data.t().contiguous().t()
            with self.assertRaisesRegex(AssertionError, "1 of 12 FFN weights"):
                save_reordered(model, random_orders(model), f"{root}/out")
            self.assertFalse(os.path.exists(f"{root}/out"))

    def test_a_save_in_another_thread_is_written_as_usual(self):
        with tempfile.TemporaryDirectory() as root:
            model = LlamaForCausalLM(LlamaConfig(**SMALL))
            other = LlamaForCausalLM(LlamaConfig(**SMALL))
            in_dry_run, finish = threading.Event(), threading.Event()
            save = model.save_pretrained

            def save_once_the_other_has(*args, **kwargs):
                if not in_dry_run.is_set():
                    in_dry_run.set()
                    finish.wait(timeout=60)
                save(*args, **kwargs)

            model.save_pretrained = save_once_the_other_has
            saver = threading.Thread(
                target=save_reordered,
                args=(model, random_orders(model), f"{root}/reordered"),
            )
            saver.start()
            try:
                self.assertTrue(in_dry_run.wait(timeout=60))
                other.save_pretrained(f"{root}/other")
            finally:
                finish.set()
                saver.join()

            self.assertTrue(os.path.exists(f"{root}/reordered/model.safetensors"))
            reloaded = LlamaForCausalLM.from_pretrained(f"{root}/other").state_dict()
            for name, tensor in other.state_dict().items():
                self.assertEqual(reloaded[name], tensor)


@suppress_logging_cls()
class TestLayerwiseSaliency(TestCase):
    """One decoder layer at a time has to score every channel exactly as a whole-model forward does."""

    def test_every_decoder_family_matches_a_whole_model_forward(self):
        batches = token_batches()
        for name, build in tiny_decoders().items():
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build().eval()
                expected = whole_model_saliency(model, batches)
                got = layerwise_saliency(model, batches, torch.device("cpu"))
                self.assertEqual(sorted(got), list(range(len(expected))))
                for index, score in enumerate(expected):
                    self.assertEqual(got[index], score)

    def test_a_failing_forward_raises_and_puts_the_real_layers_back(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**SMALL)).eval()
        layers = list(model.model.layers)
        layerwise_saliency(model, token_batches(), torch.device("cpu"))
        self.assertTrue(all(a is b for a, b in zip(model.model.layers, layers)))
        # A token id past the vocabulary fails inside the recording forward.
        with self.assertRaises(IndexError):
            layerwise_saliency(
                model, [torch.full((1, 4), VOCAB + 1)], torch.device("cpu")
            )
        self.assertTrue(all(a is b for a, b in zip(model.model.layers, layers)))

    def test_layers_that_share_keys_and_values_are_refused(self):
        model = Gemma4ForCausalLM(gemma4_config(num_kv_shared_layers=2)).eval()
        with self.assertRaisesRegex(AssertionError, "reuse keys and values"):
            layerwise_saliency(model, token_batches(), torch.device("cpu"))

    def test_layer_arguments_that_change_between_batches_are_refused(self):
        # Per-layer inputs are looked up from the token ids, so every batch gets different ones.
        model = Gemma4ForCausalLM(
            gemma4_config(
                hidden_size_per_layer_input=16, vocab_size_per_layer_input=VOCAB
            )
        ).eval()
        with self.assertRaisesRegex(AssertionError, "change from batch to batch"):
            layerwise_saliency(model, token_batches(), torch.device("cpu"))


@unittest.skipUnless(gpu_available(), "needs a GPU")
@suppress_logging_cls()
class TestLayerwiseSaliencyOnGPU(TestCase):

    def test_scores_match_a_whole_model_forward_on_the_gpu(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**SMALL)).eval()
        batches = token_batches()
        got = layerwise_saliency(model, batches, torch.device("cuda"))
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
        expected = whole_model_saliency(model.cuda(), batches)
        for index, score in enumerate(expected):
            self.assertEqual(got[index], score)

    def test_hidden_states_kept_on_the_host_give_the_same_scores(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**SMALL)).eval()
        batches = token_batches()
        on_device = layerwise_saliency(model, batches, torch.device("cuda"))
        # No free memory reported, so the hidden states have to wait on the host between layers.
        with patch("torch.cuda.mem_get_info", return_value=(0, 1 << 40)):
            on_host = layerwise_saliency(model, batches, torch.device("cuda"))
        for index, score in on_device.items():
            self.assertEqual(on_host[index], score)

    def test_only_one_decoder_layer_is_on_the_gpu_at_a_time(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**SMALL)).eval()
        layers = list(model.model.layers)
        resident = []

        def count(module, args, kwargs):
            resident.append(
                sum(any(p.is_cuda for p in layer.parameters()) for layer in layers)
            )

        for layer in layers:
            layer.register_forward_pre_hook(count, with_kwargs=True)
        layerwise_saliency(model, token_batches(), torch.device("cuda"))
        self.assertEqual(len(resident), len(layers) * len(token_batches()))
        self.assertEqual(set(resident), {1})
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))

    def test_the_host_weights_are_left_where_they_were(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**SMALL)).eval()
        pointers = [parameter.data_ptr() for parameter in model.parameters()]
        layerwise_saliency(model, token_batches(), torch.device("cuda"))
        self.assertEqual(
            [parameter.data_ptr() for parameter in model.parameters()], pointers
        )


@suppress_logging_cls()
class TestCalibrate(TestCase):

    def test_a_model_without_gated_blocks_is_rejected_before_loading_data(self):
        config = StepCalibrationConfig(
            model_name_or_path="m", output_dir="o", device="cuda:0"
        )
        with patch.multiple(
            "pace.llm.step.calibrate",
            require_gpu=DEFAULT,
            AutoTokenizer=DEFAULT,
            AutoConfig=DEFAULT,
            load_model=Mock(return_value=nn.Sequential(nn.Linear(4, 4))),
            CalibrationDataset=DEFAULT,
        ) as patched:
            with self.assertRaisesRegex(AssertionError, "no dense gate/up/down FFN"):
                calibrate(config)
        patched["CalibrationDataset"].assert_not_called()

    def test_a_failed_calibration_writes_nothing(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        output = f"{directory}/calibrated"
        config = StepCalibrationConfig(
            model_name_or_path="m", output_dir=output, device="cuda:0"
        )
        dataset = Mock(
            size=Mock(return_value=4),
            get_batch=Mock(return_value={"input_ids": torch.zeros(4, 8).long()}),
        )
        with patch.multiple(
            "pace.llm.step.calibrate",
            require_gpu=DEFAULT,
            AutoTokenizer=DEFAULT,
            AutoConfig=DEFAULT,
            load_model=Mock(return_value=TinyModel()),
            CalibrationDataset=Mock(return_value=dataset),
            layerwise_saliency=Mock(side_effect=RuntimeError("out of memory")),
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                calibrate(config)
        self.assertFalse(os.path.exists(output))

    @unittest.skipUnless(gpu_available(), "needs a GPU")
    def test_the_written_checkpoint_computes_what_its_source_does(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source, corpus = write_calibration_source(directory)
        output = f"{directory}/calibrated"

        calibrate(calibration_config(source, output, corpus))

        before = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
        after = AutoModelForCausalLM.from_pretrained(output, dtype=torch.float32)
        probe = token_batches(sizes=(1,), length=24)[0]
        with torch.no_grad():
            self.assertEqual(
                after(probe).logits, before(probe).logits, atol=1e-4, rtol=1e-4
            )
        for original, reordered in zip(before.model.layers, after.model.layers):
            self.assertFalse(
                torch.equal(original.mlp.up_proj.weight, reordered.mlp.up_proj.weight)
            )

    @unittest.skipUnless(gpu_available(), "needs a GPU")
    def test_calibrating_into_the_source_directory_writes_the_same_checkpoint(self):
        # The save replaces files that the model's weights are still memory-mapped from.
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source, corpus = write_calibration_source(directory)
        elsewhere = f"{directory}/calibrated"

        calibrate(calibration_config(source, elsewhere, corpus))
        calibrate(calibration_config(source, source, corpus))

        expected = AutoModelForCausalLM.from_pretrained(elsewhere).state_dict()
        in_place = AutoModelForCausalLM.from_pretrained(source).state_dict()
        self.assertEqual(in_place.keys(), expected.keys())
        for name, tensor in in_place.items():
            self.assertEqual(tensor, expected[name])


@suppress_logging_cls()
class TestConfigValidation(TestCase):

    def test_calibration_rejects_nonsense(self):
        for override in (
            {"num_samples": 0},
            {"sequence_length": -1},
            {"batch_size": 0},
            {"datasets": []},
        ):
            with self.assertRaises(AssertionError):
                StepCalibrationConfig(
                    model_name_or_path="m", output_dir="o", **override
                )

    def test_calibration_runs_on_a_gpu(self):
        for device in ("cuda", "cuda:3"):
            StepCalibrationConfig(model_name_or_path="m", output_dir="o", device=device)
        for device in ("cpu", "auto", "gpu", "cuda:x", "cuda:0,1"):
            with self.subTest(device=device), self.assertRaises(AssertionError):
                StepCalibrationConfig(
                    model_name_or_path="m", output_dir="o", device=device
                )
