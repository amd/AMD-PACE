# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
import math

from pace.llm.configs import SamplingConfig
from pace.llm.sampler import Sampler
from pace.utils.logging import suppress_logging_cls

input_tensor = torch.randint(
    low=0,
    high=5,
    size=(2, 5),
    dtype=torch.int64,
)


@suppress_logging_cls()
class TestSampler(TestCase):
    def test_greedy_search(self):
        config = SamplingConfig(
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (2, 1))
        # Greedy fast path returns None for probs/logprobs (no softmax needed).
        self.assertIsNone(output.probs)
        self.assertIsNone(output.logprobs)

    def test_random_sampling(self):
        config = SamplingConfig(
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (2, 1))
        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)

    @given(
        temperature=st.floats(min_value=0.01, max_value=5.0),
        batch_size=st.integers(min_value=1, max_value=1024),
        vocab_size=st.integers(min_value=2, max_value=1_000),
        min_p=st.floats(min_value=0.0, max_value=1.0),
        top_p=st.floats(min_value=0.0, max_value=1.0),
        top_k=st.integers(min_value=0, max_value=1_000),
    )
    def test_sampling_params(
        self, temperature, batch_size, vocab_size, min_p, top_p, top_k
    ):
        config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(batch_size, vocab_size)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (batch_size, 1))
        if not sampler._greedy_fast_path:
            self.assertIsNotNone(output.probs)
            self.assertIsNotNone(output.logprobs)

    def test_seed_reproducibility(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
            sampling_seed=42,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler1 = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        output1 = sampler1.sample(input_tensor, logits)

        # Reset the seed in a new sampler
        sampler2 = Sampler(config, input_tensor)
        output2 = sampler2.sample(input_tensor, logits)

        self.assertTrue(torch.equal(output1.next_tokens, output2.next_tokens))

    def test_large_top_k_sampling(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=1_000_000,  # Larger than vocab
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(3, 4)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (3, 1))

    def test_zero_temperature(self):
        config = SamplingConfig(
            temperature=0.0000001,
            top_k=0,
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(3, 5)
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.next_tokens.shape, (3, 1))

    def test_infinite_logits(self):
        config = SamplingConfig(
            temperature=1.0,
            top_k=2,
            top_p=1.0,
            min_p=0,
            eos_token_id=[0],
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.randn(2, 5)
        logits[0, 3] = torch.nan
        with self.assertRaisesRegex(AssertionError, "Invalid logits"):
            sampler.sample(input_tensor, logits)

    def test_min_new_tokens(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0,
            top_k=50,
            random_seed=123,
            eos_token_id=1,
            min_new_tokens=4,
            return_logprobs=True,  # Force slow path to verify EOS suppression
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.tensor(
            [[1.3, 0.41, -0.651, 0.1, 9.1], [1.21, 3.41, 0.71, 0.87, 0.1]]
        )
        output = sampler.sample(input_tensor, logits)
        self.assertEqual(output.logprobs[0][1], -math.inf)
        self.assertEqual(output.logprobs[1][1], -math.inf)

    def test_repetition_penalty_application(self):
        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0,
            eos_token_id=3,
            repetition_penalty=1.87,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, input_tensor)
        logits = torch.tensor(
            [[1.3, 0.22, 1.78, 12, 1.2], [-1.7, 1.43, 1.45, 0.651, 0.41]]
        )
        output = sampler.sample(input_tensor, logits)
        logit = torch.gather(logits, 1, input_tensor)
        logit = torch.where(
            logit > 0,
            logit / config.repetition_penalty,
            logit * config.repetition_penalty,
        )
        processed_logits = logits.scatter(1, input_tensor, logit)
        processed_logits = torch.softmax(processed_logits, dim=-1, dtype=torch.float)
        self.assertEqual(processed_logits, output.probs)

    def test_greedy_fast_path_flag(self):
        """Greedy fast path flag is set correctly at init."""
        # Default greedy: fast path eligible
        greedy_config = SamplingConfig(do_sample=False, eos_token_id=[0])
        greedy_config.verify_max_new_tokens()
        greedy_config.finalize()
        s1 = Sampler(greedy_config, input_tensor)
        self.assertTrue(s1._greedy_fast_path)

        # Greedy with penalty: NOT eligible
        penalty_config = SamplingConfig(
            do_sample=False, frequency_penalty=0.5, eos_token_id=[0]
        )
        penalty_config.verify_max_new_tokens()
        penalty_config.finalize()
        s2 = Sampler(penalty_config, input_tensor)
        self.assertFalse(s2._greedy_fast_path)

        # Sampling: NOT eligible
        sample_config = SamplingConfig(
            do_sample=True, temperature=0.7, eos_token_id=[0]
        )
        sample_config.verify_max_new_tokens()
        sample_config.finalize()
        s3 = Sampler(sample_config, input_tensor)
        self.assertFalse(s3._greedy_fast_path)

        # Greedy with min_new_tokens: NOT eligible (needs EOS masking)
        min_tokens_config = SamplingConfig(
            do_sample=False, min_new_tokens=4, eos_token_id=[0]
        )
        min_tokens_config.verify_max_new_tokens()
        min_tokens_config.finalize()
        s4 = Sampler(min_tokens_config, input_tensor)
        self.assertFalse(s4._greedy_fast_path)

    def test_greedy_fast_path_matches_slow_path(self):
        """Greedy fast path produces same tokens as full pipeline."""
        test_input = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.randn(1, 10)

        # Fast path (default greedy)
        fast_config = SamplingConfig(do_sample=False, eos_token_id=[0])
        fast_config.verify_max_new_tokens()
        fast_config.finalize()
        fast_sampler = Sampler(fast_config, test_input)
        fast_out = fast_sampler.sample(test_input, logits.clone())

        # Slow path (greedy + return_logprobs forces slow path)
        slow_config = SamplingConfig(
            do_sample=False, return_logprobs=True, eos_token_id=[0]
        )
        slow_config.verify_max_new_tokens()
        slow_config.finalize()
        slow_sampler = Sampler(slow_config, test_input)
        slow_out = slow_sampler.sample(test_input, logits.clone())

        self.assertEqual(fast_out.next_tokens, slow_out.next_tokens)
        self.assertIsNone(fast_out.probs)
        self.assertIsNotNone(slow_out.probs)

    def test_frequency_penalty_application(self):
        prompt_tokens = torch.tensor([[0, 1, 2, 3, 4]])
        prompt_and_output = torch.tensor([[0, 1, 2, 3, 4, 2, 2, 3]])

        config = SamplingConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0,
            eos_token_id=3,
            frequency_penalty=0.5,
        )
        config.verify_max_new_tokens()
        config.finalize()
        sampler = Sampler(config, prompt_tokens)

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        output = sampler.sample(prompt_and_output, logits)

        # Output tokens are indices [5:] = [2, 2, 3]
        # Token 2 appears 2 times, token 3 appears 1 time
        expected_logits = logits.clone()
        bin_counts = torch.zeros_like(expected_logits)
        output_tokens = prompt_and_output[:, prompt_tokens.shape[-1] :]
        bin_counts.scatter_add_(
            1,
            output_tokens,
            torch.ones_like(output_tokens, dtype=expected_logits.dtype),
        )
        expected_logits = expected_logits - 0.5 * bin_counts
        expected_probs = torch.softmax(expected_logits, dim=-1, dtype=torch.float)
        self.assertEqual(expected_probs, output.probs)
