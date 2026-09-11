# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ContinuousBatchScheduler and mixed-step engine_client helpers.

Uses the same mock-based approach as test_scheduler.py: mock aiohttp sessions
return canned engine JSON so we can exercise the scheduler and engine_client
logic without a running engine.
"""

import os
import asyncio
import uuid
import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from pace.server.router.continuous_batch_scheduler import ContinuousBatchScheduler
from pace.server.router.engine_client import (
    build_mixed_step_payload,
    process_mixed_step_prefills,
    process_mixed_step_decodes,
    error_out_requests,
)
from pace.server.router.utils import RequestStatus, RequestStats

os.environ["PACE_LOG_LEVEL"] = "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(req_id=None, input_length=10):
    req = Mock()
    req.request_id = req_id or str(uuid.uuid4())
    req.status = RequestStatus.QUEUED
    req.finish_reason = None
    stats = RequestStats()
    stats.input_length = input_length
    req.req_stats = stats
    req.token_queue = AsyncMock()
    req.req = Mock()
    req.req.prompt = list(range(input_length))
    req.engine_gen_config = {"max_new_tokens": 50, "temperature": 0.0}
    req.assigned_engine_url = None
    req.assigned_engine_index = 0
    return req


def _mock_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    def _make_post_ctx(url, json=None):
        resp = AsyncMock()
        resp.status = 200
        resp.raise_for_status = Mock()
        resp.json = AsyncMock(return_value={"status": "success"})
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    session.post = Mock(side_effect=_make_post_ctx)
    return session


# ---------------------------------------------------------------------------
# Scheduler init & _schedule
# ---------------------------------------------------------------------------


class TestCBSInit(unittest.TestCase):

    def test_default_init(self):
        sched = ContinuousBatchScheduler("http://localhost:8000")
        self.assertEqual(sched.token_budget, 2048)
        self.assertEqual(sched.max_num_running_reqs, 128)
        self.assertEqual(sched.tokens_per_decode, 1)
        self.assertEqual(sched.max_prefill_chunk, 0)

    def test_multi_engine(self):
        urls = ["http://host:8000", "http://host:8001", "http://host:8002"]
        sched = ContinuousBatchScheduler(engine_url=urls)
        self.assertEqual(sched.num_engines, 3)
        self.assertEqual(len(sched.decode_queues), 3)
        self.assertEqual(len(sched.chunked_queues), 3)

    def test_admission_cap_clamp(self):
        sched = ContinuousBatchScheduler(
            token_budget=100,
            max_num_running_reqs=999,
            tokens_per_decode=10,
        )
        self.assertEqual(sched.max_num_running_reqs, 10)

    def test_tokens_per_decode_floor(self):
        sched = ContinuousBatchScheduler(tokens_per_decode=0)
        self.assertEqual(sched.tokens_per_decode, 1)


class TestScheduleMethod(unittest.IsolatedAsyncioTestCase):

    def _sched(self, **kw):
        defaults = dict(
            token_budget=100,
            max_num_running_reqs=10,
            tokens_per_decode=1,
        )
        defaults.update(kw)
        return ContinuousBatchScheduler("http://localhost:8000", **defaults)

    def test_empty_queues_returns_nothing(self):
        s = self._sched()
        self.assertEqual(s._schedule(num_running=0, worker_id=0), [])

    async def test_single_prefill_admitted(self):
        s = self._sched()
        req = _make_request(input_length=20)
        await s.submit_request(req)
        items = s._schedule(num_running=0, worker_id=0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], 0)
        self.assertGreater(items[0][2], 0)

    async def test_budget_limits_prefill_tokens(self):
        s = self._sched(token_budget=50)
        for _ in range(5):
            await s.submit_request(_make_request(input_length=30))
        items = s._schedule(num_running=0, worker_id=0)
        self.assertLessEqual(sum(c for _, _, c in items), 50)

    async def test_running_decodes_consume_budget(self):
        s = self._sched(token_budget=100, tokens_per_decode=10)
        for _ in range(3):
            await s.submit_request(_make_request(input_length=20))
        items = s._schedule(num_running=8, worker_id=0)
        self.assertLessEqual(sum(c for _, _, c in items), 20)

    async def test_max_running_caps_admission(self):
        s = self._sched(token_budget=1000, max_num_running_reqs=3)
        for _ in range(10):
            await s.submit_request(_make_request(input_length=5))
        items = s._schedule(num_running=2, worker_id=0)
        self.assertLessEqual(len(items), 1)

    async def test_chunked_continuations_scheduled_first(self):
        s = self._sched(token_budget=100, max_num_running_reqs=5)
        chunked_req = _make_request(input_length=200)
        s.chunked_queues[0][chunked_req.request_id] = (chunked_req, 50)
        await s.submit_request(_make_request(input_length=10))
        items = s._schedule(num_running=0, worker_id=0)
        self.assertEqual(items[0][0].request_id, chunked_req.request_id)

    async def test_max_prefill_chunk(self):
        s = self._sched(token_budget=500, max_prefill_chunk=20)
        await s.submit_request(_make_request(input_length=100))
        items = s._schedule(num_running=0, worker_id=0)
        self.assertLessEqual(items[0][2], 20)

    def test_decode_cost_exceeds_budget_asserts(self):
        s = self._sched(token_budget=10, tokens_per_decode=5)
        with self.assertRaises(AssertionError):
            s._schedule(num_running=3, worker_id=0)


# ---------------------------------------------------------------------------
# build_mixed_step_payload
# ---------------------------------------------------------------------------


class TestBuildMixedStepPayload(unittest.TestCase):

    def test_empty_work_items(self):
        payload, req_map = build_mixed_step_payload([])
        self.assertEqual(payload["prefill_batch"], [])
        self.assertEqual(req_map, {})

    def test_single_prefill(self):
        req = _make_request(input_length=5)
        payload, req_map = build_mixed_step_payload([(req, 0, 5)])
        self.assertEqual(len(payload["prefill_batch"]), 1)
        self.assertIn(req.request_id, req_map)
        self.assertEqual(payload["prefill_batch"][0]["prompt"], req.req.prompt)

    def test_continuation_sends_empty_prompt(self):
        req = _make_request(input_length=100)
        payload, _ = build_mixed_step_payload([(req, 50, 50)])
        self.assertEqual(payload["prefill_batch"][0]["prompt"], [])
        self.assertEqual(payload["chunk_info_map"][req.request_id], [50, 50])

    def test_sets_end_wait_time_once(self):
        req = _make_request()
        self.assertIsNone(req.req_stats.end_wait_time)
        build_mixed_step_payload([(req, 0, 5)])
        self.assertIsNotNone(req.req_stats.end_wait_time)
        first_ts = req.req_stats.end_wait_time
        build_mixed_step_payload([(req, 5, 5)])
        self.assertEqual(req.req_stats.end_wait_time, first_ts)


# ---------------------------------------------------------------------------
# process_mixed_step_prefills
# ---------------------------------------------------------------------------


class TestProcessMixedStepPrefills(unittest.IsolatedAsyncioTestCase):

    async def test_prefill_completed_moves_to_decode_queue(self):
        req = _make_request()
        rid = req.request_id
        decode_queue: Dict[str, Any] = {}
        result = {
            "prefill_result": {rid: {"status": "PREFILL_COMPLETED", "token_ids": [42]}}
        }
        await process_mixed_step_prefills(
            result,
            {rid: req},
            decode_queue,
            {},
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertIn(rid, decode_queue)
        self.assertEqual(req.req_stats.generated_tokens_count, 1)

    async def test_prefill_in_progress_updates_chunked_queue(self):
        req = _make_request()
        rid = req.request_id
        chunked_queue: Dict[str, Any] = {}
        result = {
            "prefill_result": {
                rid: {
                    "status": "PREFILL_IN_PROGRESS",
                    "num_computed_tokens": 50,
                    "actual_input_length": 200,
                }
            }
        }
        await process_mixed_step_prefills(
            result,
            {rid: req},
            {},
            chunked_queue,
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertIn(rid, chunked_queue)
        self.assertEqual(chunked_queue[rid], (req, 50))
        self.assertEqual(req.req_stats.input_length, 200)

    async def test_completed_during_prefill_finalizes(self):
        req = _make_request()
        rid = req.request_id
        result = {
            "prefill_result": {
                rid: {"status": "COMPLETED", "token_ids": [99], "stop_reason": "length"}
            }
        }
        await process_mixed_step_prefills(
            result,
            {rid: req},
            {},
            {},
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertEqual(req.status, RequestStatus.COMPLETED)
        self.assertEqual(req.finish_reason, "length")

    async def test_error_in_prefill(self):
        req = _make_request()
        rid = req.request_id
        result = {"prefill_result": {rid: {"error": "OOM"}}}
        await process_mixed_step_prefills(
            result,
            {rid: req},
            {},
            {},
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertEqual(req.status, RequestStatus.ERROR)


# ---------------------------------------------------------------------------
# process_mixed_step_decodes
# ---------------------------------------------------------------------------


class TestProcessMixedStepDecodes(unittest.IsolatedAsyncioTestCase):

    async def test_decode_in_progress_forwards_tokens(self):
        req = _make_request()
        rid = req.request_id
        decode_queue = {rid: req}
        result = {
            "decode_result": {
                rid: {"status": "DECODING_IN_PROGRESS", "token_ids": [10, 11, 12]}
            }
        }
        await process_mixed_step_decodes(
            result,
            decode_queue,
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertIn(rid, decode_queue)
        self.assertEqual(req.req_stats.generated_tokens_count, 3)

    async def test_decode_completed_removes_from_queue(self):
        req = _make_request()
        rid = req.request_id
        decode_queue = {rid: req}
        result = {
            "decode_result": {
                rid: {"status": "COMPLETED", "token_ids": [77], "stop_reason": "stop"}
            }
        }
        await process_mixed_step_decodes(
            result,
            decode_queue,
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertNotIn(rid, decode_queue)
        self.assertEqual(req.status, RequestStatus.COMPLETED)

    async def test_mixed_batch_selective_removal(self):
        reqs = {f"r-{i}": _make_request(req_id=f"r-{i}") for i in range(3)}
        decode_queue = dict(reqs)
        result = {
            "decode_result": {
                "r-0": {"status": "COMPLETED", "token_ids": [1]},
                "r-1": {"status": "DECODING_IN_PROGRESS", "token_ids": [2]},
                "r-2": {"status": "COMPLETED", "token_ids": [3]},
            }
        }
        await process_mixed_step_decodes(
            result,
            decode_queue,
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        self.assertNotIn("r-0", decode_queue)
        self.assertIn("r-1", decode_queue)
        self.assertNotIn("r-2", decode_queue)


# ---------------------------------------------------------------------------
# error_out_requests
# ---------------------------------------------------------------------------


class TestErrorOutRequests(unittest.IsolatedAsyncioTestCase):

    async def test_errors_all_requests(self):
        reqs = [_make_request() for _ in range(3)]
        await error_out_requests(
            reqs,
            _mock_session(),
            "http://e:8000",
            AsyncMock(),
        )
        for r in reqs:
            self.assertEqual(r.status, RequestStatus.ERROR)


# ---------------------------------------------------------------------------
# Full scheduler loop integration
# ---------------------------------------------------------------------------


class MixedStepMockFactory:

    @staticmethod
    def create_mock(mock_session_cls, complete_after_decodes=3):
        decode_count = [0]
        captured_prefill_ids = []

        def create_mock_post(url, json=None):
            prefill_result = {}
            decode_result = {}

            if json and "prefill_batch" in json:
                for pr in json["prefill_batch"]:
                    rid = pr["req_id"]
                    if rid not in captured_prefill_ids:
                        captured_prefill_ids.append(rid)
                    prefill_result[rid] = {
                        "status": "PREFILL_COMPLETED",
                        "token_ids": [42],
                    }

            for rid in list(captured_prefill_ids):
                if rid not in prefill_result:
                    decode_count[0] += 1
                    is_done = decode_count[0] >= complete_after_decodes
                    decode_result[rid] = {
                        "status": "COMPLETED" if is_done else "DECODING_IN_PROGRESS",
                        "token_ids": [99],
                        "stop_reason": "stop" if is_done else None,
                    }

            async def mock_json():
                return {
                    "status": "success",
                    "prefill_result": prefill_result,
                    "decode_result": decode_result,
                }

            resp = AsyncMock()
            resp.status = 200
            resp.raise_for_status = Mock()
            resp.json = mock_json
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        inst = AsyncMock()
        inst.post = Mock(side_effect=create_mock_post)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=None)
        mock_session_cls.return_value = inst


class TestCBSIntegration(unittest.IsolatedAsyncioTestCase):

    def _sched(self, **kw):
        defaults = dict(
            token_budget=2048,
            max_num_running_reqs=128,
            tokens_per_decode=1,
        )
        defaults.update(kw)
        return ContinuousBatchScheduler("http://localhost:8000", **defaults)

    @patch("pace.server.router.continuous_batch_scheduler.aiohttp.ClientSession")
    async def test_single_request_lifecycle(self, mock_session_cls):
        MixedStepMockFactory.create_mock(mock_session_cls, complete_after_decodes=2)
        sched = self._sched()
        await sched.start()
        try:
            req = _make_request(input_length=5)
            sched.active_requests[req.request_id] = req
            await sched.submit_request(req)
            for _ in range(20):
                await asyncio.sleep(0.05)
                if req.status in (RequestStatus.COMPLETED, RequestStatus.ERROR):
                    break
            self.assertTrue(mock_session_cls.return_value.post.called)
        finally:
            await sched.stop()

    @patch("pace.server.router.continuous_batch_scheduler.aiohttp.ClientSession")
    async def test_cancellation_drains(self, mock_session_cls):
        MixedStepMockFactory.create_mock(mock_session_cls, complete_after_decodes=999)
        sched = self._sched()
        await sched.start()
        try:
            req = _make_request(input_length=5)
            sched.active_requests[req.request_id] = req
            await sched.submit_request(req)
            await asyncio.sleep(0.1)
            sched.cancelled_requests_per_worker[0].add(req.request_id)
            await asyncio.sleep(0.2)
            self.assertNotIn(req.request_id, sched.decode_queues[0])
        finally:
            await sched.stop()


if __name__ == "__main__":
    unittest.main()
