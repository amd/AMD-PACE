# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from torch.testing._internal.common_utils import TestCase
from transformers import FineGrainedFP8Config, LlamaConfig, Mistral3Config

from pace.llm.utils import (
    dequantization_config,
    drop_quantization_config,
    quantization_method,
)
from pace.utils.logging import suppress_logging_cls


@suppress_logging_cls()
class TestQuantizationMethod(TestCase):

    def test_unquantized_config_reports_nothing(self):
        self.assertEqual(quantization_method(LlamaConfig()), "")

    def test_dict_quantization_config_is_read(self):
        config = LlamaConfig()
        config.quantization_config = {"quant_method": "FP8"}
        self.assertEqual(quantization_method(config), "fp8")

    def test_object_quantization_config_is_read(self):
        class Quant:
            quant_method = "fp8"

        config = LlamaConfig()
        config.quantization_config = Quant()
        self.assertEqual(quantization_method(config), "fp8")


@suppress_logging_cls()
class TestDequantizationConfig(TestCase):

    def test_an_unquantized_checkpoint_needs_nothing(self):
        self.assertIsNone(dequantization_config(LlamaConfig(), torch.bfloat16))

    def test_fp8_is_dequantized_on_load(self):
        config = LlamaConfig()
        config.quantization_config = {"quant_method": "fp8"}
        found = dequantization_config(config, torch.bfloat16)
        self.assertIsInstance(found, FineGrainedFP8Config)
        self.assertTrue(found.dequantize)

    def test_a_scheme_without_a_dequantizing_path_loads_as_saved(self):
        config = LlamaConfig()
        config.quantization_config = {"quant_method": "awq"}
        self.assertIsNone(dequantization_config(config, torch.bfloat16))


@suppress_logging_cls()
class TestDropQuantizationConfig(TestCase):

    def test_dropping_clears_the_config_so_reloads_do_not_requantize(self):
        config = LlamaConfig()
        config.quantization_config = {"quant_method": "fp8"}
        config._pre_quantization_dtype = "float8"
        drop_quantization_config(config)
        self.assertIsNone(config.quantization_config)
        self.assertFalse(hasattr(config, "_pre_quantization_dtype"))

    def test_a_nested_text_config_is_cleared_too(self):
        config = Mistral3Config()
        config.quantization_config = {"quant_method": "fp8"}
        config.text_config.quantization_config = {"quant_method": "fp8"}
        drop_quantization_config(config, config.text_config)
        self.assertIsNone(config.quantization_config)
        self.assertIsNone(config.text_config.quantization_config)
