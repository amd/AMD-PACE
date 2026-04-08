# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import os

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase

from pace.llm import SamplingConfig
from pace.llm.configs import SamplingMode, PardSpecDecodeConfig
from pace.utils.logging import suppress_logging_cls


@suppress_logging_cls()
class TestSamplingConfig(TestCase):

    def test_sampling_config(self):
        config = SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.1,
            eos_token_id=[1],
            pad_token_id=0,
            stop_strings=["stop"],
        )
        config.verify_max_new_tokens()
        config.finalize()
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.min_p, 0.1)
        self.assertEqual(config.eos_token_id.tolist(), [1])
        self.assertEqual(
            config.pad_token_id.tolist(), 0
        )  # pad_token_id is a single value
        self.assertEqual(config.stop_strings, ["stop"])

    def test_from_dict(self):
        config_dict = {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "eos_token_id": [1],
        }
        sampling_config = SamplingConfig.from_dict(**config_dict)
        self.assertEqual(sampling_config.temperature, 0.7)
        self.assertEqual(sampling_config.top_k, 50)
        self.assertEqual(sampling_config.top_p, 0.9)
        self.assertEqual(sampling_config.eos_token_id, [1])

    def test_from_pretrained(self):
        config_path = "generation_config.json"
        config_dict = {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "eos_token_id": [1],
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        sampling_config = SamplingConfig.from_pretrained(config_path)
        self.assertEqual(sampling_config.temperature, 0.7)
        self.assertEqual(sampling_config.top_k, 50)
        self.assertEqual(sampling_config.top_p, 0.9)
        self.assertEqual(sampling_config.eos_token_id, [1])

        # Clean up
        os.remove(config_path)

    def test_merge_from_with_tokenizer(self):
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = 30522

        config = SamplingConfig(temperature=0.7, top_k=50, eos_token_id=[1])
        tokenizer = MockTokenizer()
        config.merge_from(tokenizer=tokenizer)
        self.assertEqual(config.pad_token_id, [0])
        self.assertEqual(config.eos_token_id, [1])
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.vocab_size, 30522)

    def test_set_sampling_method(self):
        config = SamplingConfig(
            temperature=0.0,
            eos_token_id=[1],
        )
        config._set_sampling_method()
        self.assertEqual(config.sampling_mode, SamplingMode.GREEDY_SEARCH)

        config = SamplingConfig(
            temperature=1.0,
            do_sample=True,
            eos_token_id=[1],
        )
        config._set_sampling_method()
        self.assertEqual(config.sampling_mode, SamplingMode.RANDOM_SAMPLING)

    @given(
        temperature=st.floats(min_value=0.0, max_value=10.0),
        top_k=st.integers(min_value=0, max_value=torch.iinfo(torch.long).max),
        top_p=st.floats(min_value=0.0, max_value=1.0),
        min_p=st.floats(min_value=0.0, max_value=1.0),
        eos_token_id=st.lists(
            st.integers(min_value=0, max_value=torch.iinfo(torch.long).max),
            min_size=1,
            max_size=10,
        ),
    )
    def test_sampling_config_hypothesis(
        self, temperature, top_k, top_p, min_p, eos_token_id
    ):
        sampling_config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            eos_token_id=eos_token_id,
        )
        sampling_config.verify_max_new_tokens()
        sampling_config.finalize()
        self.assertEqual(
            sampling_config.temperature, 1.0 if temperature < 1e-5 else temperature
        )
        self.assertEqual(sampling_config.top_k, 0 if temperature < 1e-5 else top_k)
        self.assertEqual(sampling_config.top_p, 1.0 if temperature < 1e-5 else top_p)
        self.assertEqual(sampling_config.min_p, 0.0 if temperature < 1e-5 else min_p)

    def test_merge_from(self):
        config1 = SamplingConfig(temperature=0.7, top_k=50, eos_token_id=[1])
        config2 = SamplingConfig(temperature=0.9, top_p=0.9, eos_token_id=[2])
        config1.merge_from(config2)
        self.assertEqual(config1.temperature, 0.9)
        self.assertEqual(config1.top_k, 50)
        self.assertEqual(config1.top_p, 0.9)
        self.assertEqual(config1.eos_token_id, [1, 2])

    def test_merge_from_defaults_do_not_override_explicit(self):
        config1 = SamplingConfig(top_k=50, temperature=0.7, eos_token_id=[1])
        config2 = SamplingConfig(eos_token_id=[1])
        config1.merge_from(config2)
        self.assertEqual(config1.top_k, 50)
        self.assertEqual(config1.temperature, 0.7)

    def test_finalize(self):
        sampling_config = SamplingConfig(
            temperature=0.0,
            eos_token_id=[0],
        )
        sampling_config.verify_max_new_tokens()
        sampling_config.finalize()
        with self.assertRaisesRegex(
            AssertionError, "finalized and cannot be modified."
        ):
            sampling_config.temperature = 0.5

    def test_invalid_temperature(self):
        config = SamplingConfig(
            temperature=-1.0,  # Invalid temperature
            eos_token_id=[0],
        )
        with self.assertRaisesRegex(AssertionError, "temperature should be >= 0"):
            config.verify_max_new_tokens()
            config.finalize()

    def test_invalid_top_p(self):
        with self.assertRaisesRegex(
            AssertionError, "top_p should be between 0 and 1, Got top_p"
        ):
            sampling_config = SamplingConfig(
                top_p=1.5,
                eos_token_id=[0],
            )
            sampling_config.verify_max_new_tokens()
            sampling_config.finalize()

    def test_invalid_min_p(self):
        with self.assertRaisesRegex(
            AssertionError, "min_p should be between 0 and 1, Got min_p"
        ):
            sampling_config = SamplingConfig(
                min_p=1.5,
                eos_token_id=[0],
            )
            sampling_config.verify_max_new_tokens()
            sampling_config.finalize()

    def test_invalid_top_k(self):
        config = SamplingConfig(
            top_k=-5,  # Invalid top_k
            eos_token_id=[0],
        )
        with self.assertRaisesRegex(AssertionError, "top_k should be >= 0"):
            config.verify_max_new_tokens()
            config.finalize()

    def test_set_sampling_method_greedy_search(self):
        sampling_config = SamplingConfig(
            do_sample=False,
            temperature=0.0,
            eos_token_id=[0],
        )
        sampling_config.verify_max_new_tokens()
        sampling_config.finalize()
        self.assertEqual(sampling_config.sampling_mode, SamplingMode.GREEDY_SEARCH)

    def test_set_sampling_method_random_sampling(self):
        sampling_config = SamplingConfig(
            do_sample=True,
            temperature=1.0,
            eos_token_id=[0],
        )
        sampling_config.verify_max_new_tokens()
        sampling_config.finalize()
        self.assertEqual(sampling_config.sampling_mode, SamplingMode.RANDOM_SAMPLING)

    def test_verify_max_new_tokens_sets_default_and_warns(self):
        config = SamplingConfig(
            max_new_tokens=None,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        self.assertEqual(config.max_new_tokens, 20)

    def test_verify_max_new_tokens_truncates_and_warns(self):
        config = SamplingConfig(
            max_new_tokens=4096,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens(initial_decoder_input_length=11)
        self.assertEqual(config.max_new_tokens, 2037)

    def test_verify_max_new_tokens_no_warning_when_valid(self):
        config = SamplingConfig(
            max_new_tokens=10,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        self.assertEqual(config.max_new_tokens, 10)


@suppress_logging_cls()
class TestPardSpecDecodeConfig(TestCase):

    def test_draft_kv_cache_memory_gb_default(self):
        config = PardSpecDecodeConfig(model_name_or_path="test-model")
        self.assertIsNone(config.draft_kv_cache_memory_gb)

    def test_draft_kv_cache_memory_gb_set(self):
        config = PardSpecDecodeConfig(
            model_name_or_path="test-model",
            draft_kv_cache_memory_gb=4.0,
        )
        self.assertEqual(config.draft_kv_cache_memory_gb, 4.0)

    def test_default_num_speculative_tokens(self):
        config = PardSpecDecodeConfig(model_name_or_path="test-model")
        self.assertEqual(config.num_speculative_tokens, 12)
