# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import uuid
from unittest.mock import Mock, patch

import torch
from torch.testing._internal.common_utils import TestCase

from pace.llm.attention import KVCacheType
from pace.llm.configs import PardSpecDecodeConfig
from pace.llm.generation import (
    GenerationBatchResult,
    GenerationDelta,
    GenerationStatus,
    NativeGenerationBackendConfig,
)
from pace.server.engine.server import ModelExecutor
from pace.server.engine.utils import ModelConfig


def generation_state(request_id):
    state = Mock()
    state.request_id = str(request_id)
    state.logprobs_n = 0
    state.prompt_token_ids = torch.tensor([[1, 2]])
    state.prompt_length = 2
    return state


class TestModelExecutorBackendAdapter(TestCase):
    def test_prompt_logprobs_exclude_left_padding(self):
        executor = ModelExecutor()
        state = generation_state(uuid.uuid4())
        state.prompt_token_ids = torch.tensor([[0, 0, 1, 3]])
        state.backend_state = Mock(prompt_start=2)
        input_logprobs = torch.log_softmax(torch.randn(2, 8), dim=-1)

        result = executor._prompt_logprobs_from_tensor(state, input_logprobs)

        self.assertEqual(result["token_ids"], [1, 3])
        self.assertEqual(len(result["token_logprobs"]), 2)

    def test_server_request_disables_padding_inference(self):
        executor = ModelExecutor()
        executor._backend = Mock()
        executor._backend.create_request.return_value = Mock()

        executor.create_sequence(uuid.uuid4(), [0, 1, 2])

        request = executor._backend.create_request.call_args.args[0]
        self.assertFalse(request.infer_padding)

    def test_create_sequence_rejects_active_request_id(self):
        executor = ModelExecutor()
        request_id = uuid.uuid4()
        executor._backend = Mock()
        executor.decode_queue[request_id] = generation_state(request_id)

        with self.assertRaisesRegex(ValueError, "already active"):
            executor.create_sequence(request_id, [1, 2])

        executor._backend.create_request.assert_not_called()

    def test_decode_removes_terminal_requests(self):
        cases = (
            (GenerationStatus.COMPLETED, "stop", [3], None),
            (GenerationStatus.ERROR, "error", [], "forward failed"),
            (GenerationStatus.CANCELLED, "client_cancelled", [], None),
        )
        for status, reason, token_ids, error in cases:
            with self.subTest(status=status):
                executor = ModelExecutor()
                request_id = uuid.uuid4()
                state = generation_state(request_id)
                executor._backend = Mock()
                executor.decode_queue[request_id] = state
                executor._backend.decode.return_value = GenerationBatchResult(
                    deltas={
                        str(request_id): GenerationDelta(
                            request_id=str(request_id),
                            token_ids=token_ids,
                            status=status,
                            finish_reason=reason,
                            error=error,
                        )
                    }
                )

                result = executor._backend_decode()

                executor._backend.decode.assert_called_once_with([state])
                self.assertEqual(result[str(request_id)]["token_ids"], token_ids)
                self.assertEqual(result[str(request_id)]["status"], status.name)
                if error is not None:
                    self.assertEqual(result[str(request_id)]["error"], error)
                self.assertNotIn(request_id, executor.decode_queue)

    def test_remove_delegates_to_idempotent_backend_release(self):
        executor = ModelExecutor()
        request_id = uuid.uuid4()
        state = generation_state(request_id)
        executor._backend = Mock()
        executor.prefill_queue[request_id] = state

        executor._backend_remove([request_id])

        executor._backend.release.assert_called_once()
        self.assertNotIn(request_id, executor.prefill_queue)

    def test_chunked_prefill_keeps_progress_cursor(self):
        executor = ModelExecutor()
        request_id = uuid.uuid4()
        state = generation_state(request_id)
        delta = GenerationDelta(
            request_id=str(request_id),
            token_ids=[],
            status=GenerationStatus.PREFILL_IN_PROGRESS,
            computed_prompt_tokens=1,
        )

        entry = executor._delta_entry(state, delta)

        self.assertEqual(entry["num_computed_tokens"], 1)
        self.assertEqual(entry["actual_input_length"], 2)


class TestModelExecutorSpecConfig(TestCase):
    def spec_config_passed_to_backend(self, spec_config):
        executor = ModelExecutor()
        with patch.object(
            NativeGenerationBackendConfig, "initialize_backend", return_value=Mock()
        ) as initialize:
            executor.load_model(ModelConfig(modelId="target", spec_config=spec_config))
        return initialize.call_args.kwargs["spec_config"]

    def test_pard_spec_config_reaches_the_backend(self):
        spec_config = self.spec_config_passed_to_backend(
            {"type": "pard", "model_name": "draft", "num_speculative_tokens": 4}
        )

        self.assertIsInstance(spec_config, PardSpecDecodeConfig)
        self.assertEqual(spec_config.model_name_or_path, "draft")
        self.assertEqual(spec_config.num_speculative_tokens, 4)

    def test_pard_spec_config_needs_a_model_name(self):
        with self.assertRaisesRegex(ValueError, "model_name"):
            self.spec_config_passed_to_backend({"type": "pard"})

    def test_unknown_spec_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown speculative decoder type"):
            self.spec_config_passed_to_backend({"type": "medusa", "model_name": "d"})

    def test_fp32_server_defaults_to_paged(self):
        executor = ModelExecutor()
        with patch.object(
            NativeGenerationBackendConfig,
            "initialize_backend",
            return_value=Mock(),
        ) as initialize:
            executor.load_model(ModelConfig(modelId="target", dataType="fp32"))

        self.assertEqual(
            initialize.call_args.kwargs["kv_cache_type"],
            KVCacheType.PAGED,
        )
