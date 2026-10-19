# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import itertools
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch.testing._internal.common_utils import TestCase
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    FineGrainedFP8Config,
    Gemma4TextConfig,
    LlamaConfig,
    Mistral3Config,
)

from pace.llm.step.loading import (
    activation_of,
    count_parameters,
    is_multimodal,
    load_model,
    move_to_device,
    on_device,
)
from pace.utils.device import gpu_available
from pace.utils.logging import suppress_logging_cls


@suppress_logging_cls()
class TestConfigHelpers(TestCase):

    def test_plain_config_is_not_multimodal(self):
        self.assertFalse(
            is_multimodal(LlamaConfig(hidden_size=64, intermediate_size=128))
        )

    def test_a_multimodal_config_is_recognised(self):
        self.assertTrue(is_multimodal(Mistral3Config()))


@suppress_logging_cls()
class TestActivationResolution(TestCase):
    """
    Llama, Mistral and Qwen name the activation field `hidden_act`; Gemma names it `hidden_activation`. A caller that reads only one crashes on the other family.
    """

    def test_hidden_act_is_read(self):
        self.assertEqual(activation_of(LlamaConfig(hidden_act="silu")), "silu")

    def test_hidden_activation_is_read(self):
        config = Gemma4TextConfig(hidden_activation="gelu_pytorch_tanh")
        self.assertFalse(hasattr(config, "hidden_act"))
        self.assertEqual(activation_of(config), "gelu_pytorch_tanh")

    def test_a_config_naming_neither_is_rejected(self):
        with self.assertRaises(AssertionError):
            activation_of(SimpleNamespace())


def tiny_llama(**overrides):
    settings = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    settings.update(overrides)
    return LlamaConfig(**settings)


@suppress_logging_cls()
class TestCountParameters(TestCase):

    def test_the_count_matches_a_built_model(self):
        for tied in (True, False):
            with self.subTest(tie_word_embeddings=tied):
                config = tiny_llama(tie_word_embeddings=tied)
                built = AutoModelForCausalLM.from_config(config)
                self.assertEqual(
                    count_parameters(config),
                    sum(parameter.numel() for parameter in built.parameters()),
                )

    def test_a_vision_tower_is_counted(self):
        # Untied like real Mistral3 checkpoints; the bare default ties only the wrapper, dropping an embedding matrix.
        config = Mistral3Config(tie_word_embeddings=False)
        self.assertGreater(
            count_parameters(config), count_parameters(config.text_config)
        )

    def test_the_config_keeps_its_quantization_record(self):
        config = tiny_llama()
        config.quantization_config = {"quant_method": "fp8"}
        count_parameters(config)
        self.assertEqual(config.quantization_config, {"quant_method": "fp8"})


@unittest.skipUnless(gpu_available(), "transformers needs a GPU to quantize to FP8")
@suppress_logging_cls()
class TestFp8Checkpoints(TestCase):
    """`load_model` dequantizes an FP8 checkpoint, and the result must save as a dense model."""

    def test_a_dequantized_fp8_checkpoint_saves_dense(self):
        """The save holds bf16 weights only, declares no quantization and reloads exactly."""
        with tempfile.TemporaryDirectory() as directory:
            # FP8 quantizes in 128-wide blocks, so every projection needs a multiple of 128.
            config = tiny_llama(hidden_size=256, intermediate_size=512)
            dense = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
            dense.save_pretrained(f"{directory}/dense")
            AutoModelForCausalLM.from_pretrained(
                f"{directory}/dense",
                quantization_config=FineGrainedFP8Config(),
                dtype=torch.bfloat16,
            ).save_pretrained(f"{directory}/fp8")

            model = load_model(f"{directory}/fp8", torch.bfloat16)
            model.save_pretrained(f"{directory}/saved")

            with safe_open(f"{directory}/saved/model.safetensors", "pt") as saved:
                dtypes = {saved.get_slice(key).get_dtype() for key in saved.keys()}
                self.assertFalse(any("scale" in key for key in saved.keys()))
            self.assertEqual(dtypes, {"BF16"})
            self.assertIsNone(
                getattr(
                    AutoConfig.from_pretrained(f"{directory}/saved"),
                    "quantization_config",
                    None,
                )
            )
            reloaded = AutoModelForCausalLM.from_pretrained(
                f"{directory}/saved", dtype=torch.bfloat16
            ).state_dict()
            for name, tensor in model.state_dict().items():
                self.assertEqual(reloaded[name], tensor)


@unittest.skipUnless(gpu_available(), "moves a model to a GPU")
@suppress_logging_cls()
class TestMoveToDevice(TestCase):
    """`move_to_device` and `on_device` on models whose weights are memory-mapped from safetensors."""

    def test_a_loaded_checkpoint_arrives_unchanged_with_its_ties(self):
        """Every parameter and buffer reaches the device unchanged, and tied weights stay tied."""
        with tempfile.TemporaryDirectory() as directory:
            built = AutoModelForCausalLM.from_config(
                tiny_llama(tie_word_embeddings=True)
            )
            built.save_pretrained(directory)
            # Loaded back from safetensors, so the parameters are memory-mapped as a real checkpoint's are.
            model = AutoModelForCausalLM.from_pretrained(directory)
            expected = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }

            self.assertIs(move_to_device(model, torch.device("cuda")), model)

            self.assertTrue(
                all(
                    tensor.is_cuda
                    for tensor in itertools.chain(model.parameters(), model.buffers())
                )
            )
            self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
            for name, tensor in model.state_dict().items():
                self.assertEqual(tensor.cpu(), expected[name])

    def test_on_device_gives_back_the_host_weights_even_after_an_error(self):
        """The original host tensors are back in place after the block raises."""
        model = AutoModelForCausalLM.from_config(tiny_llama())
        pointers = [parameter.data_ptr() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "inside the block"):
            with on_device(model, torch.device("cuda")) as moved:
                self.assertTrue(all(p.is_cuda for p in moved.parameters()))
                raise RuntimeError("inside the block")

        self.assertEqual(
            [parameter.data_ptr() for parameter in model.parameters()], pointers
        )
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for tensor in itertools.chain(model.parameters(), model.buffers())
            )
        )
