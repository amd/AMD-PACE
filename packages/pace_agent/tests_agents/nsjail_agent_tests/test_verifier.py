# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for ResultVerifier."""

from __future__ import annotations

import unittest

from nsjail_agent.verifier import ResultVerifier, ScoredVerifier
from nsjail_agent.models import ExecutionResult, ExecutionStatus


def _make_result(**kwargs) -> ExecutionResult:
    defaults = dict(
        status=ExecutionStatus.SUCCESS,
        exit_code=0,
        stdout="",
        stderr="",
        wall_time_ms=100.0,
    )
    defaults.update(kwargs)
    return ExecutionResult(**defaults)


class TestResultVerifier(unittest.TestCase):
    def setUp(self):
        self.v = ResultVerifier()

    def test_success_on_zero_exit(self):
        r = self.v.verify(_make_result(exit_code=0, status=ExecutionStatus.SUCCESS))
        self.assertEqual(r.status, ExecutionStatus.SUCCESS)

    def test_error_on_nonzero_exit(self):
        r = self.v.verify(_make_result(exit_code=1, status=ExecutionStatus.SUCCESS))
        self.assertEqual(r.status, ExecutionStatus.ERROR)

    def test_oom_detected_in_stderr(self):
        r = self.v.verify(
            _make_result(
                exit_code=1, stderr="oom killer invoked", status=ExecutionStatus.SUCCESS
            )
        )
        self.assertEqual(r.status, ExecutionStatus.OOM)

    def test_security_violation_from_seccomp(self):
        r = self.v.verify(
            _make_result(
                exit_code=159,
                stderr="seccomp policy violation",
                status=ExecutionStatus.SUCCESS,
            )
        )
        self.assertEqual(r.status, ExecutionStatus.SECURITY_VIOLATION)

    def test_security_violation_from_nsjail_log(self):
        r = self.v.verify(
            _make_result(
                exit_code=159,
                status=ExecutionStatus.SUCCESS,
                metadata={"nsjail_log": "seccomp policy violation"},
            )
        )
        self.assertEqual(r.status, ExecutionStatus.SECURITY_VIOLATION)

    def test_oom_detected_from_nsjail_log(self):
        r = self.v.verify(
            _make_result(
                exit_code=1,
                status=ExecutionStatus.SUCCESS,
                metadata={"nsjail_log": "rlimit_as memory limit exceeded"},
            )
        )
        self.assertEqual(r.status, ExecutionStatus.OOM)

    def test_timeout_not_reclassified(self):
        r = self.v.verify(_make_result(status=ExecutionStatus.TIMEOUT, exit_code=-1))
        self.assertEqual(r.status, ExecutionStatus.TIMEOUT)


class TestScoredVerifier(unittest.TestCase):
    def test_score_success(self):
        v = ScoredVerifier()
        r = v.verify(_make_result(exit_code=0, status=ExecutionStatus.SUCCESS))
        self.assertEqual(r.metadata["score"], 1.0)

    def test_score_failure(self):
        v = ScoredVerifier()
        r = v.verify(_make_result(exit_code=1, status=ExecutionStatus.SUCCESS))
        self.assertEqual(r.metadata["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
