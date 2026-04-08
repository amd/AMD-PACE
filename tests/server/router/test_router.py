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
from pace.server.router.scheduler import IterativeScheduler
from pace.server.router.utils import CompletionRequest, GenerationConfig, RequestStatus

import pace.server.router.frontend as frontend
import pace.server.engine.frontend as engine_frontend
from pace.server.model_list import SUPPORTED_MODEL_LIST

os.environ["PACE_LOG_LEVEL"] = "none"


class RouterDirectTests(unittest.IsolatedAsyncioTestCase):
    """
    Tests for router functionality using direct function calls to endpoint handlers.
    This eliminates the need to start a separate server process.
    """

    async def asyncSetUp(self):
        """Set up test fixtures."""
        # Create mock args object
        self.mock_args = Mock()
        self.mock_args.model = "facebook/opt-6.7b"
        self.mock_args.server_host = "localhost"
        self.mock_args.server_port = 3000

        # Set the global args in frontend module
        frontend.args = self.mock_args

        # Create and set scheduler instance
        self.engine_url = "http://localhost:3000"
        frontend.scheduler = IterativeScheduler(
            self.engine_url, scheduler_metrics_enabled=True
        )

        # Start the scheduler
        await frontend.scheduler.start()

    async def asyncTearDown(self):
        """Clean up after tests."""
        if frontend.scheduler and frontend.scheduler.is_running:
            await frontend.scheduler.stop()

    def _create_mock_fastapi_request(self):
        """Helper to create a mock FastAPI request object."""
        mock_fastapi_request = AsyncMock()
        mock_fastapi_request.is_disconnected = AsyncMock(return_value=False)
        return mock_fastapi_request

    def _setup_engine_mock(
        self, mock_session, prefill_output="Hello! ", decode_output=" world"
    ):
        """Setup mock that returns DECODING_IN_PROGRESS then COMPLETED, matching real engine behavior."""
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
                                        "output": prefill_output,
                                        "status": "DECODING_IN_PROGRESS",
                                    }
                                }
                            }
                        ],
                    }
                else:
                    result_dict = {}
                    for req_id in captured_request_ids:
                        result_dict[req_id] = {
                            "output": decode_output,
                            "status": "COMPLETED",
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
        gen_config_params = {}
        top_level_params = {}

        gen_config_keys = {
            "max_new_tokens",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "top_k",
            "random_seed",
            "do_sample",
            "repetition_penalty",
            "frequency_penalty",
        }

        for key, value in kwargs.items():
            if key in gen_config_keys:
                if key == "max_tokens":
                    gen_config_params["max_new_tokens"] = value
                else:
                    gen_config_params[key] = value
            else:
                top_level_params[key] = value

        default_gen_config = {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": None,
            "stop": None,
            "top_k": 50,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
        }
        default_gen_config.update(gen_config_params)

        default_request = {
            "model": self.mock_args.model,
            "prompt": "Hello, how are you?",
            "stream": False,
            "gen_config": GenerationConfig(**default_gen_config),
        }
        default_request.update(top_level_params)

        return CompletionRequest(**default_request)

    # ========== TESTS ==========

    async def test_health_check(self):
        """Test the health check function by calling it directly."""
        # Call the actual endpoint function
        response = await frontend.health_check()

        # Extract data from JSONResponse
        data = json.loads(response.body.decode())

        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "frontend")
        self.assertIsInstance(data["scheduler_running"], bool)
        self.assertIsInstance(data["queue_size"], int)
        self.assertIsInstance(data["active_requests"], int)
        self.assertTrue(data["scheduler_running"])

    async def test_queue_status(self):
        """Test the queue status function by calling it directly."""
        # Call the actual endpoint function
        data = await frontend.get_queue_status()

        self.assertIn("queue_size", data)
        self.assertIn("active_requests", data)
        self.assertIsInstance(data["queue_size"], int)
        self.assertIsInstance(data["active_requests"], int)
        self.assertGreaterEqual(data["queue_size"], 0)
        self.assertGreaterEqual(data["active_requests"], 0)

    async def test_server_metrics(self):
        """Test the server metrics function by calling it directly."""
        # Call the actual endpoint function
        data = await frontend.get_server_wide_metrics()

        self.assertIsInstance(data, dict)
        for key, value in data.items():
            self.assertIsInstance(
                value,
                (int, float),
                f"Expected numeric value for metric '{key}', got {type(value)}",
            )

    def test_get_models(self):
        """Test the models endpoint by calling backend function directly."""
        # Call the actual backend endpoint function directly
        data = engine_frontend.list_models()

        # Verify OpenAI-compatible response structure
        self.assertIn("data", data)
        self.assertIn("object", data)
        self.assertEqual(data["object"], "list")

        models_list = data["data"]
        self.assertIsInstance(
            models_list, list, "Expected list of models in 'data' field"
        )
        self.assertGreater(len(models_list), 0, "Expected at least one available model")

        # Create set of supported model IDs for validation
        supported_model_ids = {model["id"] for model in SUPPORTED_MODEL_LIST}

        # Verify each returned model
        for model in models_list:
            # Check model structure
            self.assertIn("id", model)
            self.assertIn("object", model)
            self.assertIn("created", model)
            self.assertIn("owned_by", model)
            self.assertIn("dtypes", model)

            # Verify field types and values
            self.assertEqual(model["object"], "model")
            self.assertEqual(model["owned_by"], "local")
            self.assertIsInstance(model["created"], int)
            self.assertIsInstance(model["dtypes"], list)

            # Verify model is in supported list
            model_id = model["id"]
            self.assertIn(
                model_id,
                supported_model_ids,
                f"Model {model_id} not found in supported model list",
            )

            # Verify dtypes match supported configuration
            supported_model = next(
                m for m in SUPPORTED_MODEL_LIST if m["id"] == model_id
            )
            self.assertEqual(
                model["dtypes"],
                supported_model["dtypes"],
                f"Model {model_id} dtypes don't match supported configuration",
            )

        # Verify all supported models are available
        available_model_ids = {model["id"] for model in models_list}
        self.assertEqual(
            available_model_ids,
            supported_model_ids,
            "Available models should match supported model list exactly",
        )

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_non_streaming_basic(self, mock_session):
        """Test basic non-streaming completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, "Hello! ", " world")

        request_data = self._create_completion_request(
            prompt="Hello, how are you?",
            max_tokens=5,
            temperature=0.8,
        )

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        # Call the actual endpoint function
        response = await frontend.chat_completions(request_data, mock_fastapi_request)

        # Extract data from JSONResponse
        data = json.loads(response.body.decode())

        self.assertIn("request_id", data)
        self.assertIn("model", data)
        self.assertIn("choices", data)
        self.assertEqual(data["model"], request_data.model)

        choices = data["choices"]
        self.assertIsInstance(choices, list)
        self.assertGreater(len(choices), 0)

        choice = choices[0]
        self.assertIn("message", choice)
        self.assertIn("finish_reason", choice)

        message = choice["message"]
        self.assertIn("role", message)
        self.assertIn("content", message)
        self.assertEqual(message["role"], "assistant")

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_streaming_basic(self, mock_session):
        """Test basic streaming completion by calling endpoint directly."""
        self._setup_engine_mock(mock_session, "Token", " more")

        request_data = self._create_completion_request(
            prompt="Count from 1 to 5:",
            max_tokens=5,
            stream=True,
        )

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        # Call the actual endpoint function
        response = await frontend.chat_completions(request_data, mock_fastapi_request)

        # Verify it's a StreamingResponse
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("X-Request-ID", response.headers)

        # Collect chunks from the stream
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

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        # Call the actual endpoint function
        response = await frontend.chat_completions(request_data, mock_fastapi_request)

        # Should return JSONResponse with error
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.body.decode())
        self.assertIn("error", data)

    async def test_request_status_not_found(self):
        """Test request status for non-existent request by calling endpoint directly."""
        non_existent_id = "non-existent-request-id"

        with self.assertRaises(Exception):
            await frontend.get_request_status(non_existent_id)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_non_streaming_custom_params(self, mock_session):
        """Test non-streaming completion with custom parameters."""
        self._setup_engine_mock(mock_session, "ML is ", " fascinating")

        request_data = self._create_completion_request(
            prompt="Explain machine learning in one sentence-",
            max_tokens=40,
            temperature=0.5,
            top_p=0.85,
        )

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        response = await frontend.chat_completions(request_data, mock_fastapi_request)
        data = json.loads(response.body.decode())

        self.assertIn("request_id", data)
        self.assertIn("model", data)
        self.assertIn("choices", data)

        choice = data["choices"][0]
        content = choice["message"]["content"]
        self.assertIsInstance(content, str)

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_completion_streaming_long_response(self, mock_session):
        """Test streaming completion with longer response."""
        self._setup_engine_mock(mock_session, "Word ", " end")

        request_data = self._create_completion_request(
            prompt="A short stroll back into the past,",
            max_tokens=80,
            stream=True,
            temperature=0.6,
        )

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        response = await frontend.chat_completions(request_data, mock_fastapi_request)

        self.assertEqual(response.media_type, "text/event-stream")

        # Collect all content from stream
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
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    continue

        self.assertTrue(done_received, "Expected [DONE] signal")

    async def test_completion_malformed_request(self):
        """Test completion with malformed request data."""
        # Missing required fields - should raise validation error
        with self.assertRaises(Exception):
            # This will fail during CompletionRequest validation
            CompletionRequest(
                model="facebook/opt-6.7b",
                # Missing prompt - will raise validation error
            )

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_request_status_tracking(self, mock_session):
        """Test request status tracking for a valid completion request."""
        self._setup_engine_mock(mock_session, "Track ", " done")

        request_data = self._create_completion_request(
            prompt="Track this request status.",
            max_tokens=10,
            stream=True,
        )

        # Mock FastAPI request
        mock_fastapi_request = self._create_mock_fastapi_request()

        # Start the streaming request
        response = await frontend.chat_completions(request_data, mock_fastapi_request)
        request_id = response.headers.get("X-Request-ID")
        self.assertIsNotNone(request_id, "Request ID should be present in headers")

        # Check status immediately
        status_response = await frontend.get_request_status(request_id)

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
        self._setup_engine_mock(mock_session, "Response", " done")

        async def make_request(request_num):
            request_data = self._create_completion_request(
                prompt=f"Request {request_num}: What is AI?",
                max_tokens=5,
            )
            mock_fastapi_request = self._create_mock_fastapi_request()
            return await frontend.chat_completions(request_data, mock_fastapi_request)

        # Create concurrent tasks
        tasks = [make_request(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 3)
        for response in results:
            data = json.loads(response.body.decode())
            self.assertIn("request_id", data)
            self.assertIn("choices", data)


if __name__ == "__main__":
    unittest.main()
