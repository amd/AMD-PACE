# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import asyncio
import json
import unittest
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError
from pace.server.router.scheduler import IterativeScheduler
from pace.server.router.protocol import ChatCompletionRequest
from pace.server.router.utils import CompletionRequest, RequestStatus

import pace.server.router.request_handler as request_handler
import pace.server.engine.frontend as engine_frontend
from pace.server.router.streaming import set_scheduler
from pace.server.model_list import SUPPORTED_MODEL_LIST

os.environ["PACE_LOG_LEVEL"] = "none"


class RouterDirectTests(unittest.IsolatedAsyncioTestCase):
    """
    Tests for router functionality using direct function calls to endpoint handlers.
    This eliminates the need to start a separate server process.
    """

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.mock_args = Mock()
        self.mock_args.model = "facebook/opt-6.7b"
        self.mock_args.server_host = "localhost"
        self.mock_args.server_port = 3000

        self.engine_url = "http://localhost:3000"
        self.scheduler = IterativeScheduler(
            self.engine_url, scheduler_metrics_enabled=True
        )

        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.decode = Mock(return_value="Hello, how are you?")
        mock_tokenizer.apply_chat_template = Mock(
            return_value={
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            }
        )
        mock_tokenizer.chat_template = (
            "{% for m in messages %}{{ m.content }}{% endfor %}"
        )
        self.mock_tokenizer = mock_tokenizer

        request_handler.set_dependencies(self.scheduler, self.mock_args, mock_tokenizer)
        set_scheduler(self.scheduler)

        await self.scheduler.start()

    async def asyncTearDown(self):
        """Clean up after tests."""
        if self.scheduler and self.scheduler.is_running:
            await self.scheduler.stop()

    def _setup_engine_mock(
        self, mock_session, prefill_token_id=100, decode_token_id=200
    ):
        """Setup mock that returns token IDs matching the new engine contract."""
        captured_request_ids = []

        def create_mock_post(url, json=None):
            async def mock_json():
                if json and "prefill_batch" in json:
                    req_id = json["prefill_batch"][0]["req_id"]
                    captured_request_ids.append(req_id)
                    return {
                        "status": "success",
                        "results": [
                            {
                                "result": {
                                    req_id: {
                                        "token_ids": [prefill_token_id],
                                        "status": "DECODING_IN_PROGRESS",
                                        "num_tokens_generated": 1,
                                    }
                                }
                            }
                        ],
                    }
                else:
                    result_dict = {}
                    for req_id in captured_request_ids:
                        result_dict[req_id] = {
                            "token_ids": [decode_token_id],
                            "status": "COMPLETED",
                            "num_tokens_generated": 1,
                        }
                    return {"status": "success", "result": result_dict}

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()
            mock_response.json = mock_json

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        mock_session_instance = AsyncMock()
        mock_session_instance.post = Mock(side_effect=create_mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

    def _create_completion_request(self, **kwargs) -> CompletionRequest:
        """Helper to create a completion request with defaults."""
        defaults = {
            "model": self.mock_args.model,
            "prompt": "Hello, how are you?",
            "stream": False,
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
        }
        defaults.update(kwargs)
        return CompletionRequest(**defaults)

    def _create_chat_request(self, **kwargs) -> ChatCompletionRequest:
        """Helper to create a chat completion request with defaults."""
        defaults = {
            "model": self.mock_args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "stream": False,
            "max_tokens": 50,
            "temperature": 0.7,
        }
        defaults.update(kwargs)
        return ChatCompletionRequest(**defaults)

    # ========== TESTS ==========

    async def test_health_check(self):
        """Test the health check function by calling it directly."""
        response = await request_handler.health_check()

        data = json.loads(response.body.decode())

        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "frontend")
        self.assertIsInstance(data["scheduler_running"], bool)
        self.assertIsInstance(data["queue_size"], int)
        self.assertIsInstance(data["active_requests"], int)
        self.assertTrue(data["scheduler_running"])

    async def test_queue_status(self):
        """Test the queue status function by calling it directly."""
        data = await request_handler.get_queue_status()

        self.assertIn("queue_size", data)
        self.assertIn("active_requests", data)
        self.assertIsInstance(data["queue_size"], int)
        self.assertIsInstance(data["active_requests"], int)
        self.assertGreaterEqual(data["queue_size"], 0)
        self.assertGreaterEqual(data["active_requests"], 0)

    async def test_server_metrics(self):
        """Test the server metrics function by calling it directly."""
        data = await request_handler.get_server_wide_metrics()

        self.assertIsInstance(data, dict)
        for key, value in data.items():
            self.assertIsInstance(
                value,
                (int, float),
                f"Expected numeric value for metric '{key}', got {type(value)}",
            )

    def test_get_models(self):
        """Test the models endpoint by calling backend function directly."""
        data = engine_frontend.list_models()

        self.assertIn("data", data)
        self.assertIn("object", data)
        self.assertEqual(data["object"], "list")

        models_list = data["data"]
        self.assertIsInstance(
            models_list, list, "Expected list of models in 'data' field"
        )
        self.assertGreater(len(models_list), 0, "Expected at least one available model")

        supported_model_ids = {model["id"] for model in SUPPORTED_MODEL_LIST}

        for model in models_list:
            self.assertIn("id", model)
            self.assertIn("object", model)
            self.assertIn("created", model)
            self.assertIn("owned_by", model)
            self.assertIn("dtypes", model)

            self.assertEqual(model["object"], "model")
            self.assertEqual(model["owned_by"], "local")
            self.assertIsInstance(model["created"], int)
            self.assertIsInstance(model["dtypes"], list)

            model_id = model["id"]
            self.assertIn(
                model_id,
                supported_model_ids,
                f"Model {model_id} not found in supported model list",
            )

            supported_model = next(
                m for m in SUPPORTED_MODEL_LIST if m["id"] == model_id
            )
            self.assertEqual(
                model["dtypes"],
                supported_model["dtypes"],
                f"Model {model_id} dtypes don't match supported configuration",
            )

        available_model_ids = {model["id"] for model in models_list}
        self.assertEqual(
            available_model_ids,
            supported_model_ids,
            "Available models should match supported model list exactly",
        )

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_non_streaming_basic(self, mock_session):
        """Test basic non-streaming completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            temperature=0.8,
        )

        response = await request_handler.completions(request_data)

        data = json.loads(response.body.decode())

        self.assertIn("id", data)
        self.assertIn("model", data)
        self.assertIn("choices", data)
        self.assertEqual(data["model"], request_data.model)
        self.assertEqual(data["object"], "text_completion")

        choices = data["choices"]
        self.assertIsInstance(choices, list)
        self.assertGreater(len(choices), 0)

        choice = choices[0]
        self.assertIn("text", choice)
        self.assertIn("finish_reason", choice)
        self.assertIsInstance(choice["text"], str)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_streaming_basic(self, mock_session):
        """Test basic streaming completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Count from 1 to 5:",
            max_tokens=5,
            stream=True,
        )

        response = await request_handler.completions(request_data)

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("X-Request-ID", response.headers)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk_str.startswith("data: "):
                data_part = chunk_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data_part)
                    chunks.append(chunk_data)
                except json.JSONDecodeError:
                    continue

        self.assertGreaterEqual(len(chunks), 0)

    async def test_completion_invalid_model(self):
        """Test completion request with invalid model by calling endpoint directly."""
        request_data = self._create_completion_request(
            model="non-existent-model",
            prompt="Hello world",
        )

        response = await request_handler.completions(request_data)

        self.assertEqual(response.status_code, 404)
        data = json.loads(response.body.decode())
        self.assertIn("error", data)

    async def test_request_status_not_found(self):
        """Test request status for non-existent request by calling endpoint directly."""
        non_existent_id = "non-existent-request-id"

        with self.assertRaises(Exception):
            await request_handler.get_request_status(non_existent_id)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_non_streaming_custom_params(self, mock_session):
        """Test non-streaming completion with custom parameters."""
        self._setup_engine_mock(mock_session, 101, 202)

        request_data = self._create_completion_request(
            prompt="Explain machine learning in one sentence-",
            max_tokens=40,
            temperature=0.5,
            top_p=0.85,
        )

        response = await request_handler.completions(request_data)
        data = json.loads(response.body.decode())

        self.assertIn("id", data)
        self.assertIn("model", data)
        self.assertIn("choices", data)

        choice = data["choices"][0]
        self.assertIn("text", choice)
        self.assertIsInstance(choice["text"], str)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_streaming_long_response(self, mock_session):
        """Test streaming completion with longer response."""
        self._setup_engine_mock(mock_session, 300, 400)

        request_data = self._create_completion_request(
            prompt="A short stroll back into the past,",
            max_tokens=80,
            stream=True,
            temperature=0.6,
        )

        response = await request_handler.completions(request_data)

        self.assertEqual(response.media_type, "text/event-stream")

        content_parts = []
        done_received = False

        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk_str.startswith("data: "):
                data_part = chunk_str[6:].strip()
                if data_part == "[DONE]":
                    done_received = True
                    break
                try:
                    chunk_data = json.loads(data_part)
                    choices = chunk_data.get("choices", [])
                    if len(choices) > 0:
                        text = choices[0].get("text", "")
                        if text:
                            content_parts.append(text)
                except json.JSONDecodeError:
                    continue

        self.assertTrue(done_received, "Expected [DONE] signal")

    async def test_completion_malformed_request(self):
        """Test completion with malformed request data."""
        with self.assertRaises(Exception):
            CompletionRequest(
                model="facebook/opt-6.7b",
            )

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_request_status_tracking(self, mock_session):
        """Test request status tracking for a valid completion request."""
        self._setup_engine_mock(mock_session, 500, 600)

        request_data = self._create_completion_request(
            prompt="Track this request status.",
            max_tokens=10,
            stream=True,
        )

        response = await request_handler.completions(request_data)
        self.assertIsNotNone(
            response.headers.get("X-Request-ID"),
            "Request ID should be present in headers",
        )

        active_ids = list(self.scheduler.active_requests.keys())
        self.assertTrue(len(active_ids) > 0, "Should have at least one active request")
        request_id = active_ids[0]

        status_response = await request_handler.get_request_status(request_id)

        self.assertIn("status", status_response.model_dump())
        self.assertIn(
            status_response.status,
            [
                RequestStatus.PROCESSING,
                RequestStatus.COMPLETED,
                RequestStatus.ERROR,
                RequestStatus.QUEUED,
            ],
        )
        self.assertIn("message", status_response.model_dump())
        self.assertIn("created_at", status_response.model_dump())

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_concurrent_requests(self, mock_session):
        """Test handling of concurrent completion requests using direct calls."""
        self._setup_engine_mock(mock_session, 700, 800)

        async def make_request(request_num):
            request_data = self._create_completion_request(
                prompt=f"Request {request_num}: What is AI?",
                max_tokens=5,
            )
            return await request_handler.completions(request_data)

        tasks = [make_request(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 3)
        for response in results:
            data = json.loads(response.body.decode())
            self.assertIn("id", data)
            self.assertIn("choices", data)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_chat_completion_non_streaming_basic(self, mock_session):
        """Test basic non-streaming chat completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_chat_request(max_tokens=5)

        response = await request_handler.chat_completions(request_data)

        data = json.loads(response.body.decode())

        self.assertIn("id", data)
        self.assertTrue(data["id"].startswith("chatcmpl-"))
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], request_data.model)

        choice = data["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIsInstance(choice["message"]["content"], str)
        self.assertIn("usage", data)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_chat_completion_streaming_basic(self, mock_session):
        """Test basic streaming chat completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_chat_request(max_tokens=5, stream=True)

        response = await request_handler.chat_completions(request_data)

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("X-Request-ID", response.headers)

        roles = []
        done_received = False
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if not chunk_str.startswith("data: "):
                continue
            data_part = chunk_str[6:].strip()
            if data_part == "[DONE]":
                done_received = True
                break
            chunk_data = json.loads(data_part)
            self.assertEqual(chunk_data["object"], "chat.completion.chunk")
            delta = chunk_data["choices"][0]["delta"]
            if "role" in delta:
                roles.append(delta["role"])

        self.assertTrue(done_received, "Expected [DONE] signal")
        self.assertIn("assistant", roles)

    async def test_chat_completion_invalid_model(self):
        """Test chat completion request with invalid model."""
        request_data = self._create_chat_request(model="non-existent-model")

        response = await request_handler.chat_completions(request_data)

        self.assertEqual(response.status_code, 404)
        data = json.loads(response.body.decode())
        self.assertIn("error", data)

    async def test_chat_completion_missing_template_returns_400(self):
        """A tokenizer without a chat template yields a 400, not a 500."""
        original_template = self.mock_tokenizer.chat_template
        self.mock_tokenizer.chat_template = None
        try:
            response = await request_handler.chat_completions(
                self._create_chat_request()
            )
        finally:
            self.mock_tokenizer.chat_template = original_template

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.body.decode())
        self.assertEqual(data["error"]["code"], "no_chat_template")

    def test_chat_completion_rejects_n_greater_than_one(self):
        """n != 1 is rejected at request validation (parity with completions)."""
        with self.assertRaises(ValidationError):
            self._create_chat_request(n=2)

    def test_chat_content_parts_normalized_to_string(self):
        """OpenAI array-of-parts content is flattened to a plain string."""
        req = ChatCompletionRequest(
            model=self.mock_args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ],
        )
        self.assertEqual(req.messages[0].content, "Hello world")

    def test_chat_unsupported_content_part_rejected(self):
        """Non-text content parts (e.g. images) are rejected at validation."""
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model=self.mock_args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "http://x"}}
                        ],
                    }
                ],
            )

    def test_chat_max_completion_tokens_supersedes_max_tokens(self):
        """``max_completion_tokens`` is honored and maps to the engine config."""
        req = self._create_chat_request(max_completion_tokens=42)
        self.assertEqual(req.max_tokens, 42)
        self.assertEqual(req.to_engine_config()["max_new_tokens"], 42)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_chat_streaming_includes_usage_chunk(self, mock_session):
        """stream_options.include_usage emits a trailing usage-only chunk."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_chat_request(
            max_tokens=5,
            stream=True,
            stream_options={"include_usage": True},
        )

        response = await request_handler.chat_completions(request_data)

        usage_chunks = []
        done_received = False
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if not chunk_str.startswith("data: "):
                continue
            data_part = chunk_str[6:].strip()
            if data_part == "[DONE]":
                done_received = True
                break
            chunk_data = json.loads(data_part)
            if "usage" in chunk_data:
                self.assertEqual(chunk_data["choices"], [])
                usage_chunks.append(chunk_data["usage"])

        self.assertTrue(done_received, "Expected [DONE] signal")
        self.assertEqual(len(usage_chunks), 1)
        usage = usage_chunks[0]
        self.assertIn("prompt_tokens", usage)
        self.assertIn("completion_tokens", usage)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )

    def _setup_engine_mock_with_logprobs(
        self, mock_session, prefill_token_id=100, decode_token_id=200
    ):
        """Setup mock that returns token IDs + logprobs matching the new engine contract."""
        captured_request_ids = []

        def make_logprobs(tid):
            # Numeric engine contract: token IDs + [id, logprob] top-k pairs.
            return {
                "token_ids": [tid],
                "token_logprobs": [-0.5],
                "top_logprobs": [[[tid, -0.5], [tid + 1, -2.0]]],
            }

        def create_mock_post(url, json=None):
            async def mock_json():
                if json and "prefill_batch" in json:
                    req_id = json["prefill_batch"][0]["req_id"]
                    captured_request_ids.append(req_id)
                    return {
                        "status": "success",
                        "results": [
                            {
                                "result": {
                                    req_id: {
                                        "token_ids": [prefill_token_id],
                                        "status": "DECODING_IN_PROGRESS",
                                        "num_tokens_generated": 1,
                                        "logprobs": make_logprobs(prefill_token_id),
                                    }
                                }
                            }
                        ],
                    }
                else:
                    result_dict = {}
                    for req_id in captured_request_ids:
                        result_dict[req_id] = {
                            "token_ids": [decode_token_id],
                            "status": "COMPLETED",
                            "num_tokens_generated": 1,
                            "logprobs": make_logprobs(decode_token_id),
                        }
                    return {"status": "success", "result": result_dict}

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()
            mock_response.json = mock_json

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        mock_session_instance = AsyncMock()
        mock_session_instance.post = Mock(side_effect=create_mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_with_logprobs(self, mock_session):
        """Test non-streaming completion with logprobs returns OpenAI-compliant logprobs."""
        self._setup_engine_mock_with_logprobs(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            logprobs=1,
        )

        response = await request_handler.completions(request_data)
        data = json.loads(response.body.decode())

        self.assertIn("choices", data)
        choice = data["choices"][0]
        self.assertIn("logprobs", choice)

        lp = choice["logprobs"]
        self.assertIsNotNone(lp, "logprobs should not be None when logprobs=1")
        self.assertIn("tokens", lp)
        self.assertIn("token_logprobs", lp)
        self.assertIn("top_logprobs", lp)
        self.assertIn("text_offset", lp)
        self.assertIsInstance(lp["tokens"], list)
        self.assertIsInstance(lp["token_logprobs"], list)
        self.assertIsInstance(lp["top_logprobs"], list)
        self.assertIsInstance(lp["text_offset"], list)
        self.assertEqual(len(lp["tokens"]), len(lp["token_logprobs"]))
        self.assertEqual(len(lp["tokens"]), len(lp["top_logprobs"]))
        self.assertEqual(len(lp["tokens"]), len(lp["text_offset"]))

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_streaming_with_logprobs(self, mock_session):
        """Test streaming completion with logprobs includes text_offset in each chunk."""
        self._setup_engine_mock_with_logprobs(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            logprobs=1,
            stream=True,
        )

        response = await request_handler.completions(request_data)
        self.assertEqual(response.media_type, "text/event-stream")

        chunks = []
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk_str.startswith("data: "):
                data_part = chunk_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data_part))
                except json.JSONDecodeError:
                    continue

        logprobs_chunks = [
            c["choices"][0]["logprobs"]
            for c in chunks
            if c.get("choices") and c["choices"][0].get("logprobs") is not None
        ]
        self.assertGreater(
            len(logprobs_chunks), 0, "Expected at least one chunk with logprobs"
        )

        for lp in logprobs_chunks:
            self.assertIn("tokens", lp)
            self.assertIn("token_logprobs", lp)
            self.assertIn("top_logprobs", lp)
            self.assertIn(
                "text_offset", lp, "Streaming chunks must include text_offset"
            )
            n = len(lp["tokens"])
            self.assertEqual(len(lp["token_logprobs"]), n)
            self.assertEqual(len(lp["top_logprobs"]), n)
            self.assertEqual(len(lp["text_offset"]), n)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_streaming_logprobs_stop_string_alignment(self, mock_session):
        """Streaming + logprobs + stop string.

        The emitted text must be truncated at the stop string, and every chunk's
        logprobs arrays must be mutually length-consistent with monotonic
        offsets. Logprobs are emitted per generated token and are NOT trimmed to
        the truncated text (``text_offset`` may run past it).
        """
        self._setup_engine_mock_with_logprobs(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            logprobs=1,
            stream=True,
            stop=["how"],
        )

        response = await request_handler.completions(request_data)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if chunk_str.startswith("data: "):
                data_part = chunk_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data_part))
                except json.JSONDecodeError:
                    continue

        emitted_text = "".join(
            c["choices"][0].get("text", "") for c in chunks if c.get("choices")
        )
        # Emitted text is non-empty and truncated before the stop string.
        self.assertTrue(emitted_text.startswith("Hello"))
        self.assertNotIn("how", emitted_text)

        # Each chunk's logprobs arrays must be mutually length-consistent, and
        # offsets non-negative and monotonically non-decreasing across the
        # stream (they may exceed the truncated text).
        prev_offset = -1
        for c in chunks:
            if not c.get("choices"):
                continue
            lp = c["choices"][0].get("logprobs")
            if lp is None:
                continue
            n = len(lp["tokens"])
            self.assertEqual(len(lp["token_logprobs"]), n)
            self.assertEqual(len(lp["top_logprobs"]), n)
            self.assertEqual(len(lp["text_offset"]), n)
            for off in lp["text_offset"]:
                self.assertGreaterEqual(off, prev_offset)
                prev_offset = off

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_without_logprobs(self, mock_session):
        """Test non-streaming completion without logprobs returns logprobs=None."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
        )

        response = await request_handler.completions(request_data)
        data = json.loads(response.body.decode())

        choice = data["choices"][0]
        self.assertIsNone(choice["logprobs"])

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_with_echo(self, mock_session):
        """Test that echo=True prepends prompt text to completion."""
        self._setup_engine_mock(mock_session, 100, 200)

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            echo=True,
        )

        response = await request_handler.completions(request_data)
        data = json.loads(response.body.decode())

        choice = data["choices"][0]
        self.assertTrue(
            choice["text"].startswith("Hello, how are you?"),
            f"Echo response should start with prompt text, got: {choice['text'][:50]}",
        )

    def _setup_engine_mock_loglikelihood(
        self, mock_session, prompt_token_ids, forced_token_id=200
    ):
        """Mock the loglikelihood path: prefill COMPLETES, emitting prompt
        (echo) logprobs as a separate ``prompt_logprobs`` field alongside the
        single forced sampled token's ``logprobs``.

        Mirrors the engine contract for ``max_tokens=0`` + ``echo=True`` +
        ``logprobs`` (prompt_logprobs vs. sample logprobs split): the router
        keeps only the prompt logprobs and drops the forced token.
        """

        def make_prompt_logprobs():
            token_ids = list(prompt_token_ids)
            token_logprobs = [None] + [-0.3] * (len(prompt_token_ids) - 1)
            top_logprobs = [[] for _ in token_ids]
            return {
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        def make_sample_logprobs():
            return {
                "token_ids": [forced_token_id],
                "token_logprobs": [-0.5],
                "top_logprobs": [[]],
            }

        def create_mock_post(url, json=None):
            async def mock_json():
                if json and "prefill_batch" in json:
                    req_id = json["prefill_batch"][0]["req_id"]
                    return {
                        "status": "success",
                        "results": [
                            {
                                "result": {
                                    req_id: {
                                        "token_ids": [forced_token_id],
                                        "status": "COMPLETED",
                                        "num_tokens_generated": 1,
                                        "stop_reason": "length",
                                        "logprobs": make_sample_logprobs(),
                                        "prompt_logprobs": make_prompt_logprobs(),
                                    }
                                }
                            }
                        ],
                    }
                return {"status": "success", "result": {}}

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()
            mock_response.json = mock_json

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        mock_session_instance = AsyncMock()
        mock_session_instance.post = Mock(side_effect=create_mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_loglikelihood_max_tokens_zero(self, mock_session):
        """max_tokens=0 + echo + logprobs (loglikelihood path).

        The forced decode-step token/logprob must be excluded: only prompt
        logprobs are returned and completion_tokens must be 0.
        """
        prompt_token_ids = [1, 2, 3, 4, 5]
        self._setup_engine_mock_loglikelihood(
            mock_session, prompt_token_ids, forced_token_id=200
        )

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        response = await request_handler.completions(request_data)
        data = json.loads(response.body.decode())

        choice = data["choices"][0]
        lp = choice["logprobs"]
        self.assertIsNotNone(lp, "logprobs should be present for loglikelihood scoring")

        # Only the prompt tokens remain; the forced generated token (and its
        # logprob) must be dropped, so the count equals the prompt length.
        self.assertEqual(len(lp["tokens"]), len(prompt_token_ids))
        self.assertEqual(len(lp["token_logprobs"]), len(prompt_token_ids))
        self.assertEqual(len(lp["top_logprobs"]), len(prompt_token_ids))
        self.assertEqual(len(lp["text_offset"]), len(prompt_token_ids))
        # First prompt-token logprob is None (nothing predicts token 0).
        self.assertIsNone(lp["token_logprobs"][0])

        # No completion tokens should be counted for a loglikelihood request.
        self.assertEqual(data["usage"]["completion_tokens"], 0)

    async def test_config_endpoint(self):
        """Test the /v1/config endpoint returns model and config info."""
        self.mock_args.kv_cache_type = "BMC"
        self.mock_args.op_config = '{"attention_backend": "JIT"}'

        response = await request_handler.get_config()
        data = json.loads(response.body.decode())

        self.assertIn("model", data)
        self.assertIn("kv_cache_type", data)
        self.assertIn("attention_backend", data)
        self.assertEqual(data["model"], "facebook/opt-6.7b")
        self.assertEqual(data["kv_cache_type"], "BMC")
        self.assertEqual(data["attention_backend"], "JIT")


class EngineLogprobsBuilderTests(unittest.TestCase):
    """Numeric-contract tests for the engine's logprobs builders.

    These exercise the shared helpers used by *both* the iterative and the
    continuous-batch paths, so continuous-batch logprobs stay consistent.
    """

    def setUp(self):
        import torch  # noqa: F401

        self.torch = torch
        from pace.server.engine.server import ModelExecutor

        self.ModelExecutor = ModelExecutor

    def test_build_logprobs_object_numeric_shape(self):
        torch = self.torch
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0, -2.0]])
        row = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        tid = 2

        obj = self.ModelExecutor._build_logprobs_object([tid], row, logprobs_n=3)

        self.assertEqual(obj["token_ids"], [tid])
        self.assertEqual(len(obj["token_logprobs"]), 1)
        self.assertAlmostEqual(
            obj["token_logprobs"][0], float(row[0, tid].item()), places=5
        )
        # top_logprobs is [[ [tok_id, logprob], ... ]] with logprobs_n entries.
        self.assertEqual(len(obj["top_logprobs"][0]), 3)
        top_ids = [pair[0] for pair in obj["top_logprobs"][0]]
        self.assertEqual(top_ids[0], 0)  # highest-logit token
        for pair in obj["top_logprobs"][0]:
            self.assertIsInstance(pair[0], int)
            self.assertIsInstance(pair[1], float)

    def test_build_logprobs_object_no_topk(self):
        torch = self.torch
        row = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
        obj = self.ModelExecutor._build_logprobs_object([1], row, logprobs_n=0)
        self.assertEqual(obj["top_logprobs"], [[]])

    def test_build_prompt_logprobs_object(self):
        from types import SimpleNamespace

        torch = self.torch
        vocab = 5
        prompt_ids = [1, 3, 0, 4]
        # [1, seq, vocab] raw logits; logits at pos t predict token t+1.
        full_logits = torch.randn(1, len(prompt_ids), vocab)
        input_logprobs = torch.log_softmax(
            full_logits.squeeze(0), dim=-1, dtype=torch.float32
        )
        state = SimpleNamespace(
            prompt_token_ids=torch.tensor([prompt_ids]),
            logprobs_n=2,
            backend_state=SimpleNamespace(prompt_start=0),
        )
        obj = self.ModelExecutor()._prompt_logprobs_from_tensor(state, input_logprobs)

        self.assertEqual(obj["token_ids"], prompt_ids)
        # First prompt token has no predecessor -> None logprob, empty top.
        self.assertIsNone(obj["token_logprobs"][0])
        self.assertEqual(obj["top_logprobs"][0], [])
        # Remaining logprobs come from log_softmax(logits[i-1])[token_i].
        for i in range(1, len(prompt_ids)):
            self.assertAlmostEqual(
                obj["token_logprobs"][i],
                float(input_logprobs[i - 1, prompt_ids[i]].item()),
                places=5,
            )
            self.assertEqual(len(obj["top_logprobs"][i]), 2)


class ContinuousBatchLogprobsForwardingTests(unittest.IsolatedAsyncioTestCase):
    """The continuous-batch (mixed-step) router path must forward engine
    logprobs, so ``return_logprobs`` / echo behaves the same as the iterative
    path regardless of scheduler type."""

    def _fake_req(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            token_queue=asyncio.Queue(),
            status=RequestStatus.QUEUED,
            finish_reason="stop",
            req_stats=SimpleNamespace(
                generated_tokens_count=0,
                input_length=0,
                finished_at=0.0,
                prefill_finished_at=0.0,
            ),
        )

    async def _drain(self, queue):
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    async def test_prefill_step_forwards_logprobs_and_prompt_logprobs(self):
        from pace.server.router import engine_client

        req_id = "req-1"
        req = self._fake_req()
        sample_lp = {
            "token_ids": [200],
            "token_logprobs": [-0.5],
            "top_logprobs": [[]],
        }
        prompt_lp = {
            "token_ids": [1, 2, 3],
            "token_logprobs": [None, -0.3, -0.4],
            "top_logprobs": [[], [], []],
        }
        result = {
            "prefill_result": {
                req_id: {
                    "token_ids": [200],
                    "status": "PREFILL_COMPLETED",
                    "logprobs": sample_lp,
                    "prompt_logprobs": prompt_lp,
                }
            }
        }
        decode_queue = {}
        await engine_client.process_mixed_step_prefills(
            result,
            prefill_req_map={req_id: req},
            decode_queue=decode_queue,
            chunked_queue={},
            session=AsyncMock(),
            engine_url="",
            cleanup_callback=AsyncMock(),
        )

        items = await self._drain(req.token_queue)
        self.assertEqual(len(items), 1)
        payload = items[0]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["token_ids"], [200])
        self.assertEqual(payload["logprobs"], sample_lp)
        self.assertEqual(payload["prompt_logprobs"], prompt_lp)
        self.assertIn(req_id, decode_queue)

    async def test_decode_step_forwards_logprobs(self):
        from pace.server.router import engine_client

        req_id = "req-2"
        req = self._fake_req()
        sample_lp = {
            "token_ids": [42],
            "token_logprobs": [-1.2],
            "top_logprobs": [[[42, -1.2]]],
        }
        result = {
            "decode_result": {
                req_id: {
                    "token_ids": [42],
                    "status": "DECODING_IN_PROGRESS",
                    "logprobs": sample_lp,
                }
            }
        }
        await engine_client.process_mixed_step_decodes(
            result,
            decode_queue={req_id: req},
            session=AsyncMock(),
            engine_url="",
            cleanup_callback=AsyncMock(),
        )

        items = await self._drain(req.token_queue)
        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], dict)
        self.assertEqual(items[0]["logprobs"], sample_lp)

    async def test_decode_step_without_logprobs_streams_plain_tokens(self):
        from pace.server.router import engine_client

        req_id = "req-3"
        req = self._fake_req()
        result = {
            "decode_result": {
                req_id: {
                    "token_ids": [7, 8],
                    "status": "DECODING_IN_PROGRESS",
                }
            }
        }
        await engine_client.process_mixed_step_decodes(
            result,
            decode_queue={req_id: req},
            session=AsyncMock(),
            engine_url="",
            cleanup_callback=AsyncMock(),
        )

        items = await self._drain(req.token_queue)
        # No logprobs -> per-token plain lists, preserving streaming granularity.
        self.assertEqual(items, [[7], [8]])


if __name__ == "__main__":
    unittest.main()
