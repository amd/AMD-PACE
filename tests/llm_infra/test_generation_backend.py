# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.testing._internal.common_utils import TestCase

from pace.llm.configs import SamplingConfig
from pace.llm.generation import (
    GenerationBackend,
    GenerationProposal,
    GenerationRequest,
    GenerationStatus,
    TerminationReason,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 16

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in token_ids)


class FakeGenerationBackend(GenerationBackend):
    def __init__(self):
        self._tokenizer = FakeTokenizer()
        self._model = object()
        self.finalized = []
        self.released = []

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_config(self):
        return SimpleNamespace(max_position_embeddings=128)

    def _prepare_sampling_config(self, user_config, prompt_length):
        max_new_tokens = (
            user_config.max_new_tokens
            if user_config is not None and user_config.max_new_tokens is not None
            else 4
        )
        config = SamplingConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        config.finalize()
        return config

    def _allocate(self, state):
        state.backend_state = {"decode_calls": 0}

    def _prefill(self, states, chunk_info):
        return {
            state.request_id: GenerationProposal(
                token_ids=torch.tensor([[4]]),
                status=GenerationStatus.PREFILL_COMPLETED,
                computed_prompt_tokens=state.prompt_length,
            )
            for state in states
        }

    def _decode(self, states):
        proposals = {}
        for state in states:
            state.backend_state["decode_calls"] += 1
            proposals[state.request_id] = GenerationProposal(
                token_ids=torch.tensor([[5, 2, 9]]),
                status=GenerationStatus.DECODING_IN_PROGRESS,
                computed_prompt_tokens=state.token_count,
            )
        return proposals

    def _mixed(self, prefills, decodes, chunk_info):
        proposals = self._prefill(prefills, chunk_info)
        proposals.update(self._decode(decodes))
        return proposals

    def _finalize_proposal(self, state, proposal, keep_count, stop_reason=None):
        self.finalized.append((state.request_id, keep_count))

    def _release(self, state, reason):
        self.released.append((state.request_id, reason))


class TestGenerationBackend(TestCase):
    def test_generate_uses_common_lifecycle(self):
        backend = FakeGenerationBackend()
        output = backend.generate(
            torch.tensor([[1, 3]]),
            SamplingConfig(max_new_tokens=4),
        )

        self.assertEqual(output.output_token_ids, torch.tensor([[1, 3, 4, 5, 2]]))
        self.assertEqual([count for _, count in backend.finalized], [1, 2])
        self.assertEqual(len(backend.released), 1)
        self.assertEqual(backend.released[0][1], TerminationReason.COMPLETED)

    def test_release_is_idempotent(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            request=GenerationRequest(
                request_id="request",
                input_ids=torch.tensor([[1, 3]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
                attention_mask=None,
                logprobs_n=0,
            )
        )

        backend.release(state, TerminationReason.CANCELLED)
        backend.release(state, TerminationReason.CANCELLED)
        self.assertEqual(len(backend.released), 1)

    def test_request_state_is_not_shared(self):
        backend = FakeGenerationBackend()
        first = backend.create_request(
            GenerationRequest(
                request_id="first",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
                attention_mask=None,
                logprobs_n=0,
            )
        )
        second = backend.create_request(
            GenerationRequest(
                request_id="second",
                input_ids=torch.tensor([[7]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
                attention_mask=None,
                logprobs_n=0,
            )
        )

        first.backend_state["decode_calls"] = 3
        self.assertEqual(second.backend_state["decode_calls"], 0)

    def test_batch_prepare_failure_releases_every_state(self):
        backend = FakeGenerationBackend()

        def fail_prepare(states):
            states[0].backend_state = {"prepared": True}
            raise RuntimeError("prepare failed")

        backend._prepare_batch = fail_prepare
        requests = [
            GenerationRequest(
                request_id=str(index),
                input_ids=torch.tensor([[index + 1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
            for index in range(2)
        ]

        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
            backend.create_requests(requests)

        self.assertEqual(
            [reason for _, reason in backend.released],
            [TerminationReason.ERROR, TerminationReason.ERROR],
        )

    def test_abstract_backend_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            GenerationBackend()

    def test_ragged_prompt_logprobs_are_left_padded(self):
        values = [
            torch.ones(2, 4),
            torch.ones(3, 4),
        ]

        result = GenerationBackend._stack_optional(values)

        self.assertEqual(result.shape, (2, 3, 4))
        self.assertTrue(torch.isneginf(result[0, 0]).all())
        self.assertEqual(result[0, 1:], torch.ones(2, 4))

    def test_prefill_runs_in_inference_mode(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            GenerationRequest(
                request_id="request",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )
        original_prefill = backend._prefill

        def checked_prefill(states, chunk_info):
            self.assertTrue(torch.is_inference_mode_enabled())
            return original_prefill(states, chunk_info)

        backend._prefill = checked_prefill
        backend.prefill([state])

    def test_terminal_proposal_carries_reason(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            GenerationRequest(
                request_id="cancelled",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )
        proposal = GenerationProposal(
            token_ids=torch.empty((1, 0), dtype=torch.long),
            status=GenerationStatus.CANCELLED,
            finish_reason="client_cancelled",
        )

        delta = backend._commit_proposal(state, proposal)

        self.assertEqual(delta.status, GenerationStatus.CANCELLED)
        self.assertEqual(delta.finish_reason, "client_cancelled")
        self.assertEqual(backend.released[0][1], TerminationReason.CANCELLED)

    def test_malformed_proposal_batch_releases_request(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            GenerationRequest(
                request_id="expected",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )
        backend._prefill = Mock(
            return_value={
                "unknown": GenerationProposal(
                    token_ids=torch.tensor([[2]]),
                    status=GenerationStatus.PREFILL_COMPLETED,
                )
            }
        )

        result = backend.prefill([state])

        self.assertEqual(
            result.deltas["expected"].status,
            GenerationStatus.ERROR,
        )
        self.assertEqual(backend.released[0][1], TerminationReason.ERROR)

    def test_duplicate_request_ids_fail_before_backend_execution(self):
        backend = FakeGenerationBackend()
        states = [
            backend.create_request(
                GenerationRequest(
                    request_id="duplicate",
                    input_ids=torch.tensor([[token_id]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                )
            )
            for token_id in (1, 2)
        ]
        backend._prefill = Mock()

        result = backend.prefill(states)

        backend._prefill.assert_not_called()
        self.assertEqual(
            result.deltas["duplicate"].status,
            GenerationStatus.ERROR,
        )
        self.assertEqual(len(backend.released), 2)

    def test_failure_ends_stream_and_releases_request(self):
        backend = FakeGenerationBackend()
        backend._prefill = Mock(side_effect=RuntimeError("forward failed"))
        streamer = Mock()

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            backend.generate(
                torch.tensor([[1, 3]]),
                SamplingConfig(max_new_tokens=4),
                text_streamer=streamer,
            )

        streamer.end.assert_called_once_with()
        self.assertEqual(len(backend.released), 1)
        self.assertEqual(backend.released[0][1], TerminationReason.ERROR)

    def test_batch_streamer_receives_one_put_per_generation_step(self):
        backend = FakeGenerationBackend()
        streamer = Mock()

        backend.generate(
            torch.tensor([[1, 3], [1, 3]]),
            SamplingConfig(max_new_tokens=4),
            text_streamer=streamer,
        )

        self.assertEqual(streamer.put.call_count, 3)
        for call in streamer.put.call_args_list:
            self.assertEqual(call.args[0].shape[0], 2)
        streamer.end.assert_called_once_with()

    def test_server_state_does_not_accumulate_output_distributions(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            GenerationRequest(
                request_id="request",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(
                    max_new_tokens=2,
                    return_probs=True,
                    return_logprobs=True,
                ),
            )
        )
        proposal = GenerationProposal(
            token_ids=torch.tensor([[3]]),
            status=GenerationStatus.PREFILL_COMPLETED,
            probs=torch.ones(1, 8),
            logprobs=torch.ones(1, 8),
            prompt_logprobs=torch.ones(1, 8),
        )

        backend._commit_proposal(state, proposal)

        self.assertEqual(state.accumulated_probs, [])
        self.assertEqual(state.accumulated_logprobs, [])
        self.assertIsNone(state.input_logprobs)

    def test_commit_trims_delta_distributions_to_retained_tokens(self):
        backend = FakeGenerationBackend()
        state = backend.create_request(
            GenerationRequest(
                request_id="request",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=4),
            )
        )
        proposal = GenerationProposal(
            token_ids=torch.tensor([[5, 2, 9]]),
            status=GenerationStatus.DECODING_IN_PROGRESS,
            probs=torch.ones(1, 3, 8),
            logprobs=torch.ones(1, 3, 8),
        )

        delta = backend._commit_proposal(state, proposal)

        self.assertEqual(delta.token_ids, [5, 2])
        self.assertEqual(delta.probs.shape, (1, 2, 8))
        self.assertEqual(delta.logprobs.shape, (1, 2, 8))
