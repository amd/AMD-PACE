# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Integration tests for CodeExecutor (requires nsjail on PATH; skipped otherwise)."""

from __future__ import annotations

import shutil
import unittest

from nsjail_agent.executor import CodeExecutor
from nsjail_agent.models import ExecutionRequest, ExecutionStatus, Language


@unittest.skipUnless(
    shutil.which("nsjail"), "nsjail not found on PATH — skipping integration tests"
)
class TestCodeExecutorIntegration(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutor()

    def test_python_hello_world(self):
        req = ExecutionRequest(code="print('hello')", language=Language.PYTHON)
        result = self.executor.execute_sync(req)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertIn("hello", result.stdout)

    def test_bash_echo(self):
        req = ExecutionRequest(code="echo hello_bash", language=Language.BASH)
        result = self.executor.execute_sync(req)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertIn("hello_bash", result.stdout)

    def test_nonzero_exit_is_error(self):
        req = ExecutionRequest(
            code="import sys; sys.exit(42)", language=Language.PYTHON
        )
        result = self.executor.execute_sync(req)
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.exit_code, 42)

    def test_stdin_passed_correctly(self):
        req = ExecutionRequest(
            code="import sys; print(sys.stdin.read().upper())",
            language=Language.PYTHON,
            stdin="hello",
        )
        result = self.executor.execute_sync(req)
        self.assertIn("HELLO", result.stdout)

    def test_return_files(self):
        req = ExecutionRequest(
            code="open('output.txt', 'w').write('42')",
            language=Language.PYTHON,
            return_files=["output.txt"],
        )
        result = self.executor.execute_sync(req)
        self.assertEqual(result.output_files.get("output.txt"), "42")


if __name__ == "__main__":
    unittest.main()
