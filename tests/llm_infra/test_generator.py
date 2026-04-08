# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from unittest.mock import patch

from transformers import BatchEncoding
from torch.testing._internal.common_utils import TestCase

from pace.llm.generator import Generator
from pace.llm.configs import SamplingConfig
from pace.llm.outputs import SamplerOutput, GeneratorOutput
from pace.utils.logging import suppress_logging_cls


class MockModelConfig:
    def __init__(self, max_position_embeddings):
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = 12


class MockModel:
    def __init__(self, max_position_embeddings):
        self.config = MockModelConfig(max_position_embeddings)


@suppress_logging_cls()
class TestGeneratorMethods(TestCase):

    @patch("pace.llm.generator.init_model", return_value=MockModel(512))
    @patch("pace.llm.generator.get_tokenizer", return_value=None)
    @patch("pace.llm.generator.resolve_model_path", return_value=None)
    @patch("pace.llm.generator.validate_generator_inputs", return_value=None)
    def setUp(
        self,
        mock_init_model,
        mock_get_tokenizer,
        mock_resolve_model_path,
        mock_validate_generator_inputs,
    ):
        self.model_path = "./"
        self.tokenizer_path = "./"
        self.dtype = torch.bfloat16
        self.generator = Generator(
            self.model_path, self.tokenizer_path, self.dtype, disable_tqdm=True
        )
        # Overriding the path to avoid file not found error
        self.generator.model_path = self.model_path

    def test_prepare_inputs(self):
        prompts_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        prompts_batch = BatchEncoding(data={"input_ids": prompts_tensor})
        self.assertTrue(
            torch.equal(self.generator._prepare_inputs(prompts_tensor), prompts_tensor)
        )
        self.assertTrue(
            torch.equal(self.generator._prepare_inputs(prompts_batch), prompts_tensor)
        )

    def test_prepare_sampling_config(self):
        user_sampling_config = SamplingConfig(eos_token_id=[2])
        sampling_config = self.generator._prepare_sampling_config(user_sampling_config)
        self.assertIsInstance(sampling_config, SamplingConfig)

    def test_prepare_streamer(self):
        text_streamer = None
        input_prompts = torch.tensor([[1, 2, 3]])
        self.assertIsNone(
            self.generator._prepare_streamer(text_streamer, input_prompts)
        )

    def test_update_probs_logprobs(self):

        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        user_sampling_config = SamplingConfig(
            return_probs=True, return_logprobs=True, eos_token_id=[2]
        )
        self.generator.prepare_for_generate(prompts, user_sampling_config)

        probs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        logprobs = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        next_tokens = torch.randint(0, 200, (2, 2))
        sampler_output = SamplerOutput(
            next_tokens=next_tokens,
            probs=torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
            logprobs=torch.tensor([[1.3, 1.4], [1.5, 1.6]]),
        )

        updated_probs, updated_logprobs = self.generator._update_probs_logprobs(
            probs, logprobs, sampler_output
        )
        expected_probs = torch.tensor([[0.1, 0.2, 0.9, 1.0], [0.3, 0.4, 1.1, 1.2]])
        expected_logprobs = torch.tensor([[0.5, 0.6, 1.3, 1.4], [0.7, 0.8, 1.5, 1.6]])
        self.assertTrue(torch.equal(updated_probs, expected_probs))
        self.assertTrue(torch.equal(updated_logprobs, expected_logprobs))

    def test_prepare_output_for_generate(self):
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.generator.prepare_for_generate(prompts, SamplingConfig(eos_token_id=[2]))

        output_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        generator_output = self.generator._prepare_output_for_generate(output_token_ids)
        self.assertEqual(generator_output.output_token_ids, output_token_ids)

    def test_prepare_for_generate(self):
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        input_prompts = self.generator.prepare_for_generate(
            prompts, SamplingConfig(eos_token_id=[2])
        )
        self.assertTrue(torch.equal(input_prompts, prompts))

    def test_compute_positions_basic(self):
        self.generator.num_computed_tokens = torch.zeros(2, dtype=torch.long)
        positions = self.generator._compute_positions(3)
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.assertTrue(torch.equal(positions, expected))

    def test_compute_positions_with_offset(self):
        self.generator.num_computed_tokens = torch.tensor([5, 10])
        positions = self.generator._compute_positions(1)
        expected = torch.tensor([[5], [10]])
        self.assertTrue(torch.equal(positions, expected))

    def test_compute_positions_speculative(self):
        self.generator.num_computed_tokens = torch.tensor([5, 5])
        positions = self.generator._compute_positions(3)
        expected = torch.tensor([[5, 6, 7], [5, 6, 7]])
        self.assertTrue(torch.equal(positions, expected))

    def test_compute_positions_explicit_num_computed(self):
        self.generator.num_computed_tokens = torch.zeros(2, dtype=torch.long)
        override = torch.tensor([3, 7])
        positions = self.generator._compute_positions(2, num_computed=override)
        expected = torch.tensor([[3, 4], [7, 8]])
        self.assertTrue(torch.equal(positions, expected))

    def test_initial_positions_no_padding(self):
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.generator.prepare_for_generate(prompts, SamplingConfig(eos_token_id=[2]))
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.assertTrue(torch.equal(self.generator.initial_positions, expected))
        self.assertTrue(
            torch.equal(
                self.generator.num_computed_tokens, torch.zeros(2, dtype=torch.long)
            )
        )

    def test_initial_positions_left_padded(self):
        prompts = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        config = SamplingConfig(eos_token_id=[2], pad_token_id=torch.tensor([0]))
        self.generator.prepare_for_generate(prompts, config)
        expected = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]])
        self.assertTrue(torch.equal(self.generator.initial_positions, expected))


@suppress_logging_cls()
class TestGenerator(TestCase):

    def setUp(self):
        self.model_path = "facebook/opt-125m"
        self.tokenizer_path = "facebook/opt-125m"
        self.dtype = torch.bfloat16
        self.generator = Generator(
            self.model_path, self.tokenizer_path, self.dtype, disable_tqdm=True
        )

    def test_validate_generator_inputs(self):
        with self.assertRaises(FileNotFoundError):
            Generator("/invalid/model/path", self.tokenizer_path, self.dtype)
        with self.assertRaises(FileNotFoundError):
            Generator(self.model_path, "/invalid/tokenizer/path", self.dtype)
        with self.assertRaises(TypeError):
            Generator(self.model_path, self.tokenizer_path, "invalid_dtype")

    def test_generate(self):
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.generator.prepare_for_generate(prompts)
        output = self.generator.generate(prompts)
        self.assertIsInstance(output, GeneratorOutput)
