# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for LocalExecutor (runs without nsjail — always available)."""

from __future__ import annotations

import unittest
import warnings

from nsjail_agent.local_executor import LocalExecutor, make_executor
from nsjail_agent.models import ExecutionRequest, ExecutionStatus, Language
from nsjail_agent.workspace import Workspace


class TestLocalExecutor(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.executor = LocalExecutor()

    def test_python_hello(self):
        result = self.executor.execute_sync(
            ExecutionRequest(code="print('hello')", language=Language.PYTHON)
        )
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertIn("hello", result.stdout)
        self.assertEqual(result.exit_code, 0)

    def test_nonzero_exit(self):
        result = self.executor.execute_sync(
            ExecutionRequest(code="import sys; sys.exit(42)", language=Language.PYTHON)
        )
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.exit_code, 42)

    def test_stdin_forwarded(self):
        code = "import sys; print(sys.stdin.read().strip().upper())"
        result = self.executor.execute_sync(
            ExecutionRequest(code=code, language=Language.PYTHON, stdin="hello")
        )
        self.assertIn("HELLO", result.stdout)

    def test_return_files(self):
        code = "open('out.txt','w').write('42')"
        result = self.executor.execute_sync(
            ExecutionRequest(
                code=code, language=Language.PYTHON, return_files=["out.txt"]
            )
        )
        self.assertEqual(result.output_files.get("out.txt"), "42")

    def test_execute_sync_in_workspace_preserves_files_between_calls(self):
        with Workspace() as ws:
            first = self.executor.execute_sync_in_workspace(
                ExecutionRequest(code="open('extracted.txt','w').write('persisted')"),
                ws,
            )
            second = self.executor.execute_sync_in_workspace(
                ExecutionRequest(code="print(open('extracted.txt').read())"),
                ws,
            )

        self.assertEqual(first.status, ExecutionStatus.SUCCESS)
        self.assertEqual(second.status, ExecutionStatus.SUCCESS)
        self.assertIn("persisted", second.stdout)

    def test_non_python_rejected(self):
        result = self.executor.execute_sync(
            ExecutionRequest(code="echo hi", language=Language.BASH)
        )
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertIn("Python only", result.stderr)


class TestMakeExecutor(unittest.TestCase):
    def test_make_executor_returns_something(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = make_executor()
        self.assertTrue(hasattr(ex, "execute_sync"))
        self.assertTrue(hasattr(ex, "execute"))


if __name__ == "__main__":
    unittest.main()
