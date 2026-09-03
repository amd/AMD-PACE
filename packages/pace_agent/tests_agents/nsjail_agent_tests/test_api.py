# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the FastAPI application."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from nsjail_agent.api import app
from nsjail_agent.models import ExecutionResult, ExecutionStatus, NetworkPolicy

# Dummy key for auth-gate tests only; not a production credential.
_TEST_API_KEY = "test-nsjail-api-key"


class _MockExecutor:
    """Canned executor that never calls nsjail."""

    def __init__(self):
        self.last_request = None

    async def execute(self, request):
        self.last_request = request
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            exit_code=0,
            stdout="mocked output",
            stderr="",
            wall_time_ms=1.0,
        )

    def execute_sync(self, request):
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.execute(request))


class TestAPI(unittest.TestCase):
    def setUp(self):
        # /execute is fail-closed, so the happy-path tests configure a key and
        # send it via _post(); the auth-gate tests below override it.
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_API_KEY
        os.environ.pop("NSJAIL_AGENT_ALLOW_ANONYMOUS", None)
        import nsjail_agent.api as api_module

        self.api_module = api_module
        self._client_ctx = TestClient(app)
        self.client = self._client_ctx.__enter__()
        self.mock_executor = _MockExecutor()
        api_module._executor = self.mock_executor
        self.client.mock_executor = self.mock_executor

    def tearDown(self):
        self.api_module._executor = None
        self._client_ctx.__exit__(None, None, None)
        os.environ.pop("NSJAIL_AGENT_API_KEY", None)
        os.environ.pop("NSJAIL_AGENT_ALLOW_ANONYMOUS", None)

    def _post(self, payload):
        return self.client.post(
            "/execute", json=payload, headers={"X-API-Key": _TEST_API_KEY}
        )

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_execute_returns_result(self):
        payload = {"code": "print('hi')", "language": "python"}
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("mocked output", data["stdout"])

    def test_openai_schema_endpoint(self):
        resp = self.client.get("/schema/openai-tool")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "execute_code")

    def test_execute_uses_server_owned_sandbox_policy(self):
        payload = {
            "code": "print('hi')",
            "language": "python",
            "sandbox": {"network": "loopback", "allow_proc": True, "uid": 0, "gid": 0},
        }

        resp = self._post(payload)

        self.assertEqual(resp.status_code, 200)
        sent_request = self.client.mock_executor.last_request
        self.assertEqual(sent_request.sandbox.network, NetworkPolicy.NONE)
        self.assertFalse(sent_request.sandbox.allow_proc)
        self.assertEqual(sent_request.sandbox.uid, 65534)
        self.assertEqual(sent_request.sandbox.gid, 65534)

    def test_execute_requires_api_key_when_configured(self):
        with patch.dict("os.environ", {"NSJAIL_AGENT_API_KEY": _TEST_API_KEY}):
            resp = self.client.post(
                "/execute", json={"code": "print('hi')", "language": "python"}
            )

        self.assertEqual(resp.status_code, 401)

    def test_execute_accepts_configured_api_key(self):
        with patch.dict("os.environ", {"NSJAIL_AGENT_API_KEY": _TEST_API_KEY}):
            resp = self.client.post(
                "/execute",
                json={"code": "print('hi')", "language": "python"},
                headers={"X-API-Key": _TEST_API_KEY},
            )

        self.assertEqual(resp.status_code, 200)

    def test_execute_hides_internal_errors(self):
        class _FailingExecutor:
            async def execute(self, request):
                raise RuntimeError("/internal/leaked-path in traceback")

        self.api_module._executor = _FailingExecutor()
        resp = self._post({"code": "print('hi')", "language": "python"})
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json()["detail"], "Execution failed")
        self.assertNotIn("/internal/leaked-path", resp.text)


if __name__ == "__main__":
    unittest.main()
