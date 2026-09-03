# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Regression tests for the nsjail_agent review findings.

Kept in one module so each guard has an obvious owner: the sandbox runs
untrusted LLM-generated code, so a silent regression in any of these is a
security regression, not a cosmetic one.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from nsjail_agent.api import app
import nsjail_agent.api as api_module
from nsjail_agent.config import _load_json_defaults
from nsjail_agent.local_executor import LocalExecutor
from nsjail_agent.models import (
    MAX_CODE_CHARS,
    MAX_FILES,
    MAX_RETURN_FILES,
    MAX_STDIN_CHARS,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Language,
    ResourceLimits,
    SandboxConfig,
    WorkspaceFile,
)
from nsjail_agent.sandbox import build_nsjail_cmd
from nsjail_agent.workspace import Workspace

_TEST_KEY = "test-nsjail-api-key"


def _flag(cmd: list[str], name: str) -> list[str]:
    return [cmd[i + 1] for i, value in enumerate(cmd) if value == name]


class TestResourceLimitFlags(unittest.TestCase):
    """nsjail's --time_limit is WALL clock; CPU time is a separate rlimit."""

    def _cmd(self, **limits):
        return build_nsjail_cmd(
            workspace_host_path=Path("/tmp/workspace"),
            language=Language.PYTHON,
            resources=ResourceLimits(**limits),
            config=SandboxConfig(),
        )

    def test_wall_and_cpu_map_to_distinct_flags(self):
        cmd = self._cmd(max_cpu_seconds=7, max_wall_seconds=90)
        self.assertEqual(_flag(cmd, "--time_limit"), ["90"])
        self.assertEqual(_flag(cmd, "--rlimit_cpu"), ["7"])

    def test_cpu_budget_is_not_used_as_the_wall_limit(self):
        """Regression: feeding max_cpu_seconds to --time_limit killed idle jobs early."""
        cmd = self._cmd(max_cpu_seconds=5, max_wall_seconds=300)
        self.assertNotEqual(_flag(cmd, "--time_limit"), ["5"])

    def test_security_critical_flags_are_present(self):
        cmd = self._cmd()
        self.assertIn("--disable_proc", cmd)
        self.assertEqual(_flag(cmd, "--user"), ["65534"])
        self.assertEqual(_flag(cmd, "--group"), ["65534"])
        for regrant in (
            "--disable_clone_newuser",
            "--disable_no_new_privs",
            "--keep_caps",
            "--keep_env",
        ):
            self.assertNotIn(regrant, cmd)


class TestPayloadCeilings(unittest.TestCase):
    """Request content is written to the host *before* nsjail starts, so the
    in-jail rlimits do not bound it."""

    def test_ordinary_request_is_accepted(self):
        self.assertEqual(ExecutionRequest(code="print(1)").code, "print(1)")

    def test_oversized_code_is_rejected(self):
        with self.assertRaises(Exception):
            ExecutionRequest(code="x" * (MAX_CODE_CHARS + 1))

    def test_oversized_stdin_is_rejected(self):
        with self.assertRaises(Exception):
            ExecutionRequest(code="p", stdin="x" * (MAX_STDIN_CHARS + 1))

    def test_too_many_files_rejected(self):
        files = [
            WorkspaceFile(name=f"f{i}.txt", content="x") for i in range(MAX_FILES + 1)
        ]
        with self.assertRaises(Exception):
            ExecutionRequest(code="p", files=files)

    def test_too_many_return_files_rejected(self):
        with self.assertRaises(Exception):
            ExecutionRequest(
                code="p",
                return_files=[f"f{i}.txt" for i in range(MAX_RETURN_FILES + 1)],
            )


class TestWorkspacePathContainment(unittest.TestCase):
    def test_traversal_and_absolute_names_rejected(self):
        for name in (
            "../etc/passwd",
            "../../root/.ssh/id_rsa",
            "/etc/passwd",
            "sub/../../escape.txt",
            "",
        ):
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    WorkspaceFile(name=name, content="x")

    def test_return_files_traversal_rejected(self):
        for name in ("../etc/passwd", "/etc/shadow"):
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    ExecutionRequest(code="p", return_files=[name])

    def test_nested_relative_paths_allowed(self):
        request = ExecutionRequest(
            code="p",
            files=[WorkspaceFile(name="data/in.csv", content="a,b")],
            return_files=["out/result.json"],
        )
        self.assertEqual(request.files[0].name, "data/in.csv")


class TestOutputFileTruncation(unittest.TestCase):
    """Sandboxed code may write files far larger than the output budget."""

    def test_read_file_truncates_at_max_bytes(self):
        with Workspace() as ws:
            ws.write_file("big.txt", "x" * 5000)
            content = ws.read_file("big.txt", max_bytes=1000)
        self.assertIsNotNone(content)
        self.assertIn("truncated", content)
        self.assertLess(len(content), 1200)

    def test_read_file_without_cap_returns_everything(self):
        with Workspace() as ws:
            ws.write_file("big.txt", "y" * 5000)
            self.assertEqual(ws.read_file("big.txt"), "y" * 5000)

    def test_collect_files_applies_the_cap(self):
        with Workspace() as ws:
            ws.write_file("a.txt", "a" * 3000)
            collected = ws.collect_files(["a.txt"], max_bytes=100)
        self.assertIn("truncated", collected["a.txt"])
        self.assertLess(len(collected["a.txt"]), 300)

    def test_returned_files_are_capped_end_to_end(self):
        executor = LocalExecutor()
        result = executor.execute_sync(
            ExecutionRequest(
                code="open('big.txt','w').write('z'*200000)",
                return_files=["big.txt"],
                resources=ResourceLimits(max_output_bytes=2048),
            )
        )
        self.assertLess(len(result.output_files["big.txt"]), 2200)

    def test_timeout_output_is_capped(self):
        """A runaway process is exactly the case that produces huge output."""
        executor = LocalExecutor()
        result = executor.execute_sync(
            ExecutionRequest(
                code="import time\nwhile True:\n    print('x'*1000)",
                resources=ResourceLimits(max_wall_seconds=2, max_output_bytes=1024),
            )
        )
        self.assertIs(result.status, ExecutionStatus.TIMEOUT)
        self.assertLess(len(result.stdout), 1300)


class TestMalformedSandboxConfig(unittest.TestCase):
    def test_invalid_json_raises_a_clear_error(self):
        bad = Path(tempfile.mkdtemp()) / "nsjail_sandbox.json"
        bad.write_text('{"workdir": "/sandbox",', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _load_json_defaults(bad)
        self.assertIn("Could not read sandbox config", str(ctx.exception))


class _MockExecutor:
    def __init__(self):
        self.last_request = None

    async def execute(self, request):
        self.last_request = request
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            exit_code=0,
            stdout="mocked",
            stderr="",
            wall_time_ms=1.0,
        )


class TestExecuteEndpointPolicy(unittest.TestCase):
    """/execute runs arbitrary code, so it must fail closed and own its limits."""

    def setUp(self):
        for var in ("NSJAIL_AGENT_API_KEY", "NSJAIL_AGENT_ALLOW_ANONYMOUS"):
            os.environ.pop(var, None)
        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()
        self.mock = _MockExecutor()
        api_module._executor = self.mock
        self.addCleanup(self._teardown)

    def _teardown(self):
        self._client_cm.__exit__(None, None, None)
        api_module._executor = None
        for var in ("NSJAIL_AGENT_API_KEY", "NSJAIL_AGENT_ALLOW_ANONYMOUS"):
            os.environ.pop(var, None)

    def _post(self, payload, headers=None):
        return self.client.post("/execute", json=payload, headers=headers or {})

    def test_disabled_without_api_key(self):
        """Regression: an unset key must not mean 'anyone may run code'."""
        resp = self._post({"code": "print(1)", "language": "python"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("NSJAIL_AGENT_API_KEY", resp.json()["detail"])

    def test_blank_api_key_is_treated_as_unset(self):
        os.environ["NSJAIL_AGENT_API_KEY"] = "   "
        self.assertEqual(
            self._post({"code": "print(1)", "language": "python"}).status_code, 503
        )

    def test_anonymous_requires_explicit_optin(self):
        os.environ["NSJAIL_AGENT_ALLOW_ANONYMOUS"] = "1"
        self.assertEqual(
            self._post({"code": "print(1)", "language": "python"}).status_code, 200
        )

    def test_rejects_missing_and_wrong_credentials(self):
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_KEY
        self.assertEqual(
            self._post({"code": "print(1)", "language": "python"}).status_code, 401
        )
        self.assertEqual(
            self._post(
                {"code": "print(1)", "language": "python"},
                {"Authorization": "Bearer wrong"},
            ).status_code,
            401,
        )

    def test_accepts_both_header_forms(self):
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_KEY
        for headers in (
            {"X-API-Key": _TEST_KEY},
            {"Authorization": f"Bearer {_TEST_KEY}"},
        ):
            with self.subTest(headers=list(headers)):
                resp = self._post({"code": "print(1)", "language": "python"}, headers)
                self.assertEqual(resp.status_code, 200)

    def test_clamps_client_requested_resources(self):
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_KEY
        resp = self._post(
            {
                "code": "print(1)",
                "language": "python",
                "resources": {
                    "max_wall_seconds": 600,
                    "max_memory_mb": 4096,
                    "max_pids": 512,
                    "max_fsize_mb": 512,
                    "max_cpu_seconds": 300,
                },
            },
            {"X-API-Key": _TEST_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        sent = self.mock.last_request.resources
        ceiling = ResourceLimits()
        self.assertEqual(sent.max_wall_seconds, ceiling.max_wall_seconds)
        self.assertEqual(sent.max_memory_mb, ceiling.max_memory_mb)
        self.assertEqual(sent.max_pids, ceiling.max_pids)

    def test_smaller_client_limits_are_preserved(self):
        """Clamping is a ceiling, not an override."""
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_KEY
        resp = self._post(
            {
                "code": "print(1)",
                "language": "python",
                "resources": {"max_wall_seconds": 5, "max_memory_mb": 32},
            },
            {"X-API-Key": _TEST_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.mock.last_request.resources.max_wall_seconds, 5)

    def test_sandbox_policy_stays_server_owned(self):
        os.environ["NSJAIL_AGENT_API_KEY"] = _TEST_KEY
        self._post(
            {
                "code": "print(1)",
                "language": "python",
                "sandbox": {"allow_proc": True, "uid": 0, "gid": 0},
            },
            {"X-API-Key": _TEST_KEY},
        )
        sent = self.mock.last_request.sandbox
        self.assertFalse(sent.allow_proc)
        self.assertEqual(sent.uid, 65534)

    def test_does_not_bind_all_interfaces_by_default(self):
        source = Path(api_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn('host="0.0.0.0"', source)


class TestMalformedToolArguments(unittest.TestCase):
    def test_agent_recovers_from_unparseable_tool_arguments(self):
        """A model emitting invalid JSON must not abort the whole run."""
        from nsjail_agent.agent import _tool_arguments_with_files

        with self.assertRaises(json.JSONDecodeError):
            _tool_arguments_with_files(
                {
                    "id": "c1",
                    "function": {"name": "execute_code", "arguments": "{broken"},
                },
                [],
            )


if __name__ == "__main__":
    unittest.main()
