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
from pace.server.router.scheduler import IterativeScheduler, PrefillFirstScheduler
from pace.server.router.utils import CompletionRequest, GenerationConfig, RequestStatus

# Import the actual endpoint functions from frontend
import pace.server.router.frontend as frontend
import pace.server.launcher  # noqa: F401

os.environ["PACE_LOG_LEVEL"] = "none"


class MockSessionFactory:
    """Factory class for creating mock aiohttp sessions with unified response structure."""

    @staticmethod
    def create_standard_mock(
        mock_session, prefill_output="Test", decode_output=" response"
    ):
        """
        Create a unified mock session that works for both scheduler types.
        Both schedulers use the same response format - the only difference is the request payload.

        Prefill response: {"status": "success", "results": [{"result": {"0": {...}}}]}
        Decode response: {"status": "success", "result": {"req_id": {...}}}
        """
        captured_request_ids = []

        def create_mock_post(url, json=None):
            async def mock_json_response():
                # Prefill request (has "prefill_batch" key)
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
                                        "status": "PREFILL_COMPLETED",
                                    }
                                }
                            }
                        ],
                    }
                # Decode request (has "is_decode" key)
                else:
                    # Return results for all captured request IDs
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
            mock_response.json = mock_json_response

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        mock_session_instance = AsyncMock()
        mock_session_instance.post = Mock(side_effect=create_mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance


class SchedulerDirectUnitTests(unittest.TestCase):
    """Unit tests for scheduler classes using direct function calls - synchronous tests only."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://test:3000")

                self.assertEqual(scheduler.engine_url, "http://test:3000")
                self.assertFalse(scheduler.is_running)
                self.assertEqual(len(scheduler.processing_tasks), 0)
                self.assertEqual(scheduler.get_queue_size(), 0)
                self.assertEqual(scheduler.get_active_requests_count(), 0)
                self.assertEqual(len(scheduler.ttft_intervals), 0)
                self.assertEqual(len(scheduler.processing_intervals), 0)

                # PrefillFirstScheduler-specific attributes (decode_queues: list of dicts per worker)
                if scheduler_class == PrefillFirstScheduler:
                    self.assertEqual(sum(len(q) for q in scheduler.decode_queues), 0)


class SchedulerDirectSyncUnitTests(unittest.TestCase):
    """Synchronous unit tests for scheduler classes using direct function calls."""

    def test_scheduler_queue_operations(self):
        """Test queue size and active request tracking for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000", True)

                # Initial state
                self.assertEqual(scheduler.get_queue_size(), 0)
                self.assertEqual(scheduler.get_active_requests_count(), 0)

                # Create mock request
                mock_req = Mock()
                mock_req.request_id = "test-123"
                mock_req.status = RequestStatus.QUEUED

                # Add to active requests
                scheduler.active_requests[mock_req.request_id] = mock_req
                self.assertEqual(scheduler.get_active_requests_count(), 1)

    def test_calculate_active_time(self):
        """Test active time calculation with overlapping intervals for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000", True)

                # Test empty intervals
                self.assertEqual(scheduler._calculate_active_time([]), 0.0)

                # Test single interval
                intervals = [(1.0, 3.0)]
                self.assertEqual(scheduler._calculate_active_time(intervals), 2.0)

                # Test non-overlapping intervals
                intervals = [(1.0, 2.0), (3.0, 5.0)]
                self.assertEqual(scheduler._calculate_active_time(intervals), 3.0)

                # Test overlapping intervals
                intervals = [(1.0, 3.0), (2.0, 4.0), (5.0, 6.0)]
                expected = 4.0  # (1.0-4.0) + (5.0-6.0) = 3.0 + 1.0 = 4.0
                self.assertEqual(scheduler._calculate_active_time(intervals), expected)

    def test_server_metrics_calculation(self):
        """Test server metrics calculation for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000", True)

                # Set up test data
                scheduler.ttft_intervals = [(0.0, 1.0), (2.0, 3.0)]
                scheduler.processing_intervals = [(0.0, 3.0), (4.0, 7.0)]
                scheduler.total_generated_tokens = 100
                scheduler.total_completed_requests = 2
                scheduler._calculate_server_metrics()
                # Calculate metrics
                updated_metrics = scheduler.server_metrics()

                # Verify calculations
                expected_ttft = 2.0 / 2  # total_time / num_intervals = 1.0
                expected_tpot = 6.0 / 100  # total_time / total_tokens = 0.06
                expected_rps = 2 / 6.0  # num_requests / total_time = 0.333...

                self.assertAlmostEqual(
                    updated_metrics["sched_session_ttft"], expected_ttft, places=6
                )
                self.assertAlmostEqual(
                    updated_metrics["sched_session_tpot"], expected_tpot, places=6
                )
                self.assertAlmostEqual(
                    updated_metrics["sched_requests_served_per_second"],
                    expected_rps,
                    places=6,
                )


class SchedulerDirectIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for scheduler functionality using direct function calls."""

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
            "model": "facebook/opt-6.7b",
            "prompt": "Hello, how are you?",
            "stream": False,
            "gen_config": GenerationConfig(**default_gen_config),
        }
        default_request.update(top_level_params)

        return CompletionRequest(**default_request)

    async def _setup_scheduler(self, scheduler_class):
        """Helper to set up a scheduler for testing."""
        # Create mock args object
        mock_args = Mock()
        mock_args.model = "facebook/opt-6.7b"
        mock_args.server_host = "localhost"
        mock_args.server_port = 3000

        # Set the global args in frontend module
        frontend.args = mock_args

        # Create and set scheduler instance
        engine_url = "http://localhost:3000"
        frontend.scheduler = scheduler_class(engine_url, scheduler_metrics_enabled=True)

        # Start the scheduler
        await frontend.scheduler.start()

    async def _teardown_scheduler(self):
        """Helper to tear down the scheduler."""
        if frontend.scheduler and frontend.scheduler.is_running:
            await frontend.scheduler.stop()

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_scheduler_request_lifecycle(self, mock_session):
        """Test complete request lifecycle through scheduler using direct calls."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                await self._setup_scheduler(scheduler_class)

                try:
                    # Setup unified mock for both scheduler types
                    MockSessionFactory.create_standard_mock(mock_session)

                    request_data = self._create_completion_request(
                        prompt="Test request lifecycle",
                        max_tokens=10,
                    )

                    # Mock FastAPI request
                    mock_fastapi_request = AsyncMock()
                    mock_fastapi_request.is_disconnected = AsyncMock(return_value=False)

                    # Call the actual endpoint function
                    response = await frontend.chat_completions(
                        request_data, mock_fastapi_request
                    )
                    data = json.loads(response.body.decode())

                    # Verify response structure
                    self.assertIn("request_id", data)
                    self.assertIn("model", data)
                    self.assertIn("choices", data)

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
                finally:
                    await self._teardown_scheduler()

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_scheduler_concurrent_request_handling(self, mock_session):
        """Test scheduler handling multiple concurrent requests using direct calls."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                await self._setup_scheduler(scheduler_class)

                try:
                    # Setup unified mock for both scheduler types
                    MockSessionFactory.create_standard_mock(
                        mock_session, "Response", " text"
                    )

                    async def submit_request(req_num):
                        request_data = self._create_completion_request(
                            prompt=f"Concurrent request {req_num}: What is AI?",
                            max_tokens=15,
                        )
                        mock_fastapi_request = AsyncMock()
                        mock_fastapi_request.is_disconnected = AsyncMock(
                            return_value=False
                        )
                        return await frontend.chat_completions(
                            request_data, mock_fastapi_request
                        )

                    # Create concurrent tasks
                    tasks = [submit_request(i) for i in range(3)]
                    results = await asyncio.gather(*tasks)

                    # Verify results
                    self.assertEqual(len(results), 3)
                    for response in results:
                        data = json.loads(response.body.decode())
                        self.assertIn("request_id", data)
                        self.assertIn("choices", data)
                finally:
                    await self._teardown_scheduler()

    @patch("pace.server.router.scheduler.aiohttp.ClientSession")
    async def test_scheduler_streaming_request_handling(self, mock_session):
        """Test scheduler handling of streaming requests using direct calls."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                await self._setup_scheduler(scheduler_class)

                try:
                    # Setup unified mock for both scheduler types
                    MockSessionFactory.create_standard_mock(
                        mock_session, "Token", " more"
                    )

                    request_data = self._create_completion_request(
                        prompt="Stream test: Count from 1 to 10",
                        max_tokens=30,
                        stream=True,
                    )

                    mock_fastapi_request = AsyncMock()
                    mock_fastapi_request.is_disconnected = AsyncMock(return_value=False)

                    response = await frontend.chat_completions(
                        request_data, mock_fastapi_request
                    )

                    # Verify it's a StreamingResponse
                    self.assertEqual(response.media_type, "text/event-stream")

                    # Collect streaming chunks
                    chunks = []
                    done_received = False

                    async for chunk in response.body_iterator:
                        chunk_str = (
                            chunk.decode() if isinstance(chunk, bytes) else chunk
                        )
                        if chunk_str.startswith("data: "):
                            data_part = chunk_str[6:].strip()
                            if data_part == "[DONE]":
                                done_received = True
                                break
                            try:
                                chunk_data = json.loads(data_part)
                                chunks.append(chunk_data)
                            except json.JSONDecodeError:
                                continue

                    # Verify streaming response
                    self.assertTrue(done_received, "Expected end of stream signal")
                    self.assertGreaterEqual(len(chunks), 0, "Expected streaming chunks")
                finally:
                    await self._teardown_scheduler()

    async def test_scheduler_server_metrics(self):
        """Test scheduler metrics collection and reporting using direct calls."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                await self._setup_scheduler(scheduler_class)

                try:
                    metrics = await frontend.get_server_wide_metrics()

                    self.assertIsInstance(
                        metrics, dict, "Expected metrics as dictionary"
                    )

                    # Verify metrics contain expected fields
                    for key, value in metrics.items():
                        self.assertIsInstance(
                            value,
                            (int, float),
                            f"Expected numeric value for metric '{key}', got {type(value)}",
                        )
                finally:
                    await self._teardown_scheduler()


class SchedulerDirectAsyncUnitTests(unittest.IsolatedAsyncioTestCase):
    """Async unit tests for scheduler classes using direct function calls."""

    async def test_scheduler_start_stop(self):
        """Test scheduler start and stop functionality for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000")

                try:
                    # Test start
                    await scheduler.start()
                    self.assertTrue(scheduler.is_running)
                    self.assertGreater(len(scheduler.processing_tasks), 0)

                    # Test stop
                    await scheduler.stop()
                    self.assertFalse(scheduler.is_running)
                finally:
                    if scheduler.is_running:
                        await scheduler.stop()

    async def test_request_status_tracking(self):
        """Test request status tracking for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000")

                # Test non-existent request
                status = await scheduler.get_request_status("non-existent")
                self.assertIsNone(status)

                # Test existing request
                mock_req = Mock()
                mock_req.request_id = "test-456"
                mock_req.status = RequestStatus.PROCESSING
                mock_req.req_stats = {"created_at": 1234567890.0}

                scheduler.active_requests["test-456"] = mock_req

                status = await scheduler.get_request_status("test-456")
                self.assertIsNotNone(status)
                self.assertEqual(status["status"], RequestStatus.PROCESSING)
                self.assertIn("message", status)
                self.assertIn("created_at", status)

    async def test_cleanup_request(self):
        """Test request cleanup functionality for both scheduler types."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class("http://localhost:3000")

                # Create mock request with stats
                mock_req = Mock()
                mock_req.request_id = "cleanup-test"
                mock_req.req_stats = {
                    "created_at": 1.0,
                    "prefill_finished_at": 2.0,
                    "finished_at": 5.0,
                    "generated_tokens_count": 10,
                    "end_wait_time": 1.5,
                }

                # Add to active requests
                scheduler.active_requests["cleanup-test"] = mock_req
                self.assertEqual(scheduler.get_active_requests_count(), 1)

                # Cleanup request
                await scheduler._cleanup_request(mock_req)

                # Verify cleanup
                self.assertEqual(scheduler.get_active_requests_count(), 0)
                self.assertNotIn("cleanup-test", scheduler.active_requests)

                # Verify stats were calculated
                self.assertIn("TTFT", mock_req.req_stats)
                self.assertIn("TPOT", mock_req.req_stats)
                self.assertEqual(mock_req.req_stats["TTFT"], 1.0)
                self.assertEqual(mock_req.req_stats["TPOT"], 0.4)

    async def test_decode_queue_management(self):
        """Test decode queue management specific to PrefillFirstScheduler (decode_queues per worker)."""
        scheduler = PrefillFirstScheduler("http://localhost:3000")
        # Single engine: use worker 0's decode queue
        worker_id = 0

        # Create mock request
        mock_req = Mock()
        mock_req.request_id = "decode-test"
        mock_req.status = RequestStatus.PROCESSING
        mock_req.req_sampling_params = Mock()
        mock_req.req_sampling_params.max_new_tokens = 100

        # Add to decode queue for worker 0
        scheduler.decode_queues[worker_id]["decode-test"] = mock_req
        self.assertEqual(len(scheduler.decode_queues[worker_id]), 1)

        # Remove from decode queue
        del scheduler.decode_queues[worker_id]["decode-test"]
        self.assertEqual(len(scheduler.decode_queues[worker_id]), 0)


class MultiInstanceSchedulerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for multi-instance engine URL assignment, queue isolation, and launcher."""

    URLS = [
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
    ]

    def _make_mock_request(self, req_id: str):
        mock_req = Mock()
        mock_req.request_id = req_id
        mock_req.assigned_engine_url = None
        mock_req.assigned_engine_index = 0
        return mock_req

    async def test_round_robin_distribution(self):
        """Round-robin cycles through engine URLs and submit_request assigns correctly."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class(list(self.URLS))

                self.assertEqual(scheduler.num_engines, 3)
                self.assertEqual(scheduler.engine_urls, self.URLS)
                self.assertEqual(scheduler.engine_url, self.URLS[0])

                # get_next_engine_url should cycle 0 -> 1 -> 2 -> 0 -> 1 -> 2
                for cycle in range(2):
                    for idx in range(3):
                        self.assertEqual(
                            scheduler.get_next_engine_url(), self.URLS[idx]
                        )

                # Reset index so submit_request starts from URL 0
                scheduler.current_engine_index = 0
                for i in range(6):
                    req = self._make_mock_request(f"rr-{i}")
                    await scheduler.submit_request(req)
                    expected_idx = i % 3
                    self.assertEqual(req.assigned_engine_url, self.URLS[expected_idx])
                    self.assertEqual(req.assigned_engine_index, expected_idx)

    async def test_per_worker_queue_isolation(self):
        """Each engine's queue only receives requests assigned to that engine."""
        for scheduler_class in [IterativeScheduler, PrefillFirstScheduler]:
            with self.subTest(scheduler=scheduler_class.__name__):
                scheduler = scheduler_class(list(self.URLS))

                for i in range(6):
                    req = self._make_mock_request(f"qi-{i}")
                    await scheduler.submit_request(req)

                self.assertEqual(scheduler.get_queue_size(), 6)

                for queue_idx in range(3):
                    q = scheduler.request_queues[queue_idx]
                    self.assertEqual(q.qsize(), 2)
                    for _ in range(q.qsize()):
                        queued_req = q.get_nowait()
                        self.assertEqual(queued_req.assigned_engine_index, queue_idx)
                        self.assertEqual(
                            queued_req.assigned_engine_url, self.URLS[queue_idx]
                        )

    @patch("pace.server.launcher.wait_for_server_ready", return_value=True)
    @patch("pace.server.launcher.subprocess.Popen")
    @patch("pace.server.launcher.psutil.cpu_count", return_value=96)
    @patch("pace.server.launcher.os.listdir", return_value=[])
    @patch(
        "sys.argv",
        [
            "launcher",
            "--server_port",
            "8000",
            "--num_engine_instances",
            "3",
            "--server_model",
            "test-model",
        ],
    )
    def test_launcher_multi_instance_startup_teardown(
        self, mock_listdir, mock_cpu_count, mock_popen, mock_ready
    ):
        """Launcher starts 3 engines on sequential ports and teardown terminates all."""
        mock_proc = Mock()
        mock_proc.wait = Mock(side_effect=KeyboardInterrupt)
        mock_popen.return_value = mock_proc

        import pace.server.launcher as launcher

        launcher.main()

        # 3 engine Popen calls + 1 router Popen call
        self.assertEqual(mock_popen.call_count, 4)

        engine_ports = []
        router_cmd = None
        for call in mock_popen.call_args_list:
            cmd = call[0][0]
            if "pace.server.engine.frontend" in cmd:
                port_idx = cmd.index("--port") + 1
                engine_ports.append(int(cmd[port_idx]))
            elif "pace.server.router.frontend" in cmd:
                router_cmd = cmd

        self.assertEqual(sorted(engine_ports), [8000, 8001, 8002])

        self.assertIn("--num_engine_instances", router_cmd)
        ni_idx = router_cmd.index("--num_engine_instances") + 1
        self.assertEqual(router_cmd[ni_idx], "3")

        # KeyboardInterrupt should trigger terminate on all 4 procs
        self.assertTrue(mock_proc.terminate.called)


if __name__ == "__main__":
    unittest.main()
