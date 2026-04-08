# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch
from unittest.mock import patch

from torch.testing._internal.common_utils import TestCase

from pace.llm.configs import PardSpecDecodeConfig, SamplingConfig, SpecDecodeConfig
from pace.llm.attention import KVCacheType
from pace.llm.outputs import ModelOutput
from pace.llm.speculative import (
    PardSpeculativeDecoder,
    SpeculationOutput,
    VerificationOutput,
    create_speculative_decoder,
)
from pace.llm.stopping_criteria import StoppingCriteria
from pace.utils.logging import suppress_logging_cls


class MockDraftModelConfig:
    def __init__(self):
        self.max_position_embeddings = 2048
        self.num_hidden_layers = 4
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 512
        self.pard_token = 99999


class MockDraftModel:
    def __init__(self):
        self.config = MockDraftModelConfig()
        self._dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def parameters(self):
        return iter([self._dummy_param])

    def __call__(self, input_ids, positions, kv_cache, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return ModelOutput(logits=logits)


@suppress_logging_cls()
class TestPardSpeculativeDecoder(TestCase):

    @patch("pace.llm.speculative.init_model", return_value=MockDraftModel())
    @patch("pace.llm.speculative.resolve_model_path", return_value="/mock/path")
    @patch("pace.llm.generator.validate_generator_inputs", return_value=None)
    def setUp(self, mock_validate, mock_resolve, mock_init):
        self.config = PardSpecDecodeConfig(
            model_name_or_path="/mock/draft",
            pard_token=99999,
            num_speculative_tokens=4,
        )
        self.decoder = PardSpeculativeDecoder(config=self.config, dtype=torch.bfloat16)

    def test_num_speculative_tokens(self):
        self.assertEqual(self.decoder.num_speculative_tokens, 4)

    def test_pard_token_resolved(self):
        self.assertIsNotNone(self.decoder.config.pard_token)
        self.assertEqual(len(self.decoder._pard_token_list), 32)

    def test_prepare_draft_input(self):
        last_token = torch.tensor([[42]])
        draft_input = self.decoder.prepare_draft_input(last_token)
        self.assertEqual(draft_input.shape, (1, 4))
        self.assertEqual(draft_input[0, 0].item(), 42)

    def test_create_draft_kv_cache(self):
        kv_cache = self.decoder.create_draft_kv_cache(128, KVCacheType.DYNAMIC)
        self.assertIsNotNone(kv_cache)

    def test_create_draft_kv_cache_paged(self):
        kv_cache = self.decoder.create_draft_kv_cache(128, KVCacheType.PAGED)
        self.assertIsNotNone(kv_cache)
        self.assertEqual(kv_cache.cache_type, KVCacheType.PAGED)
        self.assertEqual(kv_cache.num_layers, 4)

    def test_speculate_with_paged_cache(self):
        kv_cache = self.decoder.create_draft_kv_cache(128, KVCacheType.PAGED)
        num_computed = torch.zeros(1, dtype=torch.long)

        model_input = torch.tensor([[42]])

        output = self.decoder.speculate(
            model_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )

        self.assertIsInstance(output, SpeculationOutput)
        self.assertEqual(output.extended_input.shape[0], 1)
        self.assertEqual(output.extended_input.shape[1], 1 + 4)
        self.assertTrue(num_computed.item() > 0)

    def test_speculate_with_external_state(self):
        kv_cache = self.decoder.create_draft_kv_cache(128, KVCacheType.DYNAMIC)
        num_computed = torch.zeros(1, dtype=torch.long)

        model_input = torch.tensor([[42]])

        output = self.decoder.speculate(
            model_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )

        self.assertIsInstance(output, SpeculationOutput)
        self.assertEqual(output.extended_input.shape[0], 1)
        self.assertEqual(output.extended_input.shape[1], 1 + 4)
        self.assertTrue(num_computed.item() > 0)

    def test_verify_with_external_state(self):
        kv_cache = self.decoder.create_draft_kv_cache(128, KVCacheType.DYNAMIC)
        num_computed = torch.zeros(1, dtype=torch.long)

        model_input = torch.tensor([[42]])

        spec_output = self.decoder.speculate(
            model_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )

        sampled_tokens = spec_output.extended_input[:, 1:]
        sampled_tokens = torch.cat([sampled_tokens, torch.tensor([[7]])], dim=-1)

        result = self.decoder.verify(
            sampled_tokens,
            spec_output.extended_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )

        self.assertIsInstance(result, VerificationOutput)
        self.assertGreaterEqual(result.accepted_tokens.shape[1], 1)

    def test_get_stats_empty(self):
        self.decoder._total_accepted = []
        self.assertIsNone(self.decoder.get_stats())

    def test_get_stats_after_verify(self):
        self.decoder._total_accepted = [3, 2, 4]
        stats = self.decoder.get_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total_speculated_tokens, 9)
        self.assertAlmostEqual(stats.mean_accepted_tokens, 3.0)


@suppress_logging_cls()
class TestCreateSpeculativeDecoder(TestCase):

    @patch("pace.llm.speculative.init_model", return_value=MockDraftModel())
    @patch("pace.llm.speculative.resolve_model_path", return_value="/mock/path")
    @patch("pace.llm.generator.validate_generator_inputs", return_value=None)
    def test_creates_pard_decoder(self, mock_validate, mock_resolve, mock_init):
        config = PardSpecDecodeConfig(
            model_name_or_path="/mock/draft",
            pard_token=99999,
            num_speculative_tokens=8,
        )
        decoder = create_speculative_decoder(config, dtype=torch.bfloat16)
        self.assertIsInstance(decoder, PardSpeculativeDecoder)
        self.assertEqual(decoder.num_speculative_tokens, 8)

    def test_raises_for_unknown_config(self):
        config = SpecDecodeConfig()
        with self.assertRaises(ValueError):
            create_speculative_decoder(config)

    @patch("pace.llm.speculative.init_model", return_value=MockDraftModel())
    @patch("pace.llm.speculative.resolve_model_path", return_value="/mock/path")
    @patch("pace.llm.generator.validate_generator_inputs", return_value=None)
    def test_pard_token_none_raises(self, mock_validate, mock_resolve, mock_init):
        mock_model = MockDraftModel()
        mock_model.config.pard_token = None
        mock_init.return_value = mock_model

        config = PardSpecDecodeConfig(
            model_name_or_path="/mock/draft",
            pard_token=None,
            num_speculative_tokens=4,
        )
        with self.assertRaises(AssertionError):
            PardSpeculativeDecoder(config=config, dtype=torch.bfloat16)


@suppress_logging_cls()
class TestSpeculativeAcceptanceRange(TestCase):
    """Simulate 1-DRAFT_SIZE accepted tokens through verify and the downstream
    generator operations (squeeze, cat, stopping criteria)
    to ensure no shape errors anywhere in the pipeline."""

    DRAFT_SIZE = 12
    PROMPT_LEN = 20
    VOCAB_SIZE = 100
    PARD_TOKEN = 99999

    @patch("pace.llm.speculative.init_model", return_value=MockDraftModel())
    @patch("pace.llm.speculative.resolve_model_path", return_value="/mock/path")
    @patch("pace.llm.generator.validate_generator_inputs", return_value=None)
    def setUp(self, mock_validate, mock_resolve, mock_init):
        self.config = PardSpecDecodeConfig(
            model_name_or_path="/mock/draft",
            pard_token=self.PARD_TOKEN,
            num_speculative_tokens=self.DRAFT_SIZE,
        )
        self.decoder = PardSpeculativeDecoder(config=self.config, dtype=torch.bfloat16)

    def _run_acceptance_scenario(self, num_accepted: int):
        """Run verify + downstream ops for a given number of accepted
        draft tokens (0 = none accepted, DRAFT_SIZE = all accepted)."""
        kv_cache = self.decoder.create_draft_kv_cache(256, KVCacheType.DYNAMIC)
        num_computed = torch.zeros(1, dtype=torch.long)

        model_input = torch.tensor([[42]])

        spec_output = self.decoder.speculate(
            model_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )
        speculated_input = spec_output.extended_input
        speculated_tokens = speculated_input[:, -self.DRAFT_SIZE :]

        sampled = speculated_tokens.clone()
        if num_accepted < self.DRAFT_SIZE:
            sampled[0, num_accepted] = sampled[0, num_accepted] + 1
        bonus_token = torch.tensor([[7]])
        sampled_tokens = torch.cat([sampled, bonus_token], dim=-1)

        result = self.decoder.verify(
            sampled_tokens,
            speculated_input,
            draft_kv_cache=kv_cache,
            draft_num_computed=num_computed,
        )
        expected_keep = num_accepted + 1
        self.assertEqual(
            result.accepted_tokens.shape,
            (1, expected_keep),
            f"num_accepted={num_accepted}: accepted_tokens shape mismatch",
        )
        remove_count = self.DRAFT_SIZE - expected_keep
        expected_kv_trim = remove_count + 1
        self.assertEqual(result.target_kv_cache_trim, expected_kv_trim)

        next_tokens = result.accepted_tokens

        next_tokens_3d = next_tokens.unsqueeze(-1)
        self.assertEqual(next_tokens_3d.dim(), 3)
        if next_tokens_3d.dim() == 3:
            next_tokens_3d = next_tokens_3d.squeeze(-1)
        self.assertEqual(next_tokens_3d.shape, next_tokens.shape)

        all_tokens = torch.randint(0, self.VOCAB_SIZE, (1, self.PROMPT_LEN))
        unfinished = torch.ones(1, dtype=torch.long)
        padded = next_tokens * unfinished.reshape(-1, 1)
        all_tokens = torch.cat([all_tokens, padded], dim=-1)
        self.assertEqual(
            all_tokens.shape,
            (1, self.PROMPT_LEN + expected_keep),
        )

        eos_ids = [2, 3]
        sampling_cfg = SamplingConfig(max_new_tokens=2048, eos_token_id=eos_ids)
        prompt_tensor = torch.zeros(1, self.PROMPT_LEN)
        stopping = StoppingCriteria(sampling_cfg, prompt_tensor)
        stop_out = stopping.stop_now(all_tokens, num_new_tokens=next_tokens.shape[-1])
        self.assertEqual(stop_out.shape, (1,))
        self.assertEqual(stop_out.dtype, torch.bool)

    def test_all_acceptance_scenarios(self):
        for num_accepted in range(self.DRAFT_SIZE + 1):
            self._run_acceptance_scenario(num_accepted)
