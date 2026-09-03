# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Post-execution result verification and status classification."""

from __future__ import annotations

import re

from .models import ExecutionResult, ExecutionStatus

# nsjail log patterns that indicate specific failure modes
_OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"oom",
        r"out of memory",
        r"memory limit",
        r"rlimit_as",
    ]
]

_SECURITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"seccomp",
        r"syscall.*denied",
        r"policy violation",
        r"killed.*signal 31",  # SIGSYS — seccomp kill
    ]
]


class ResultVerifier:
    """Classifies raw execution results and enriches them with a status code.

    Override :meth:`verify` to add custom post-processing (e.g., output
    validation, unit test scoring, or GAIA metric computation).
    """

    def verify(self, result: ExecutionResult) -> ExecutionResult:
        """Return a (possibly mutated) copy of *result* with the correct status."""
        if result.status == ExecutionStatus.TIMEOUT:
            return result

        status = self._classify(result)
        return result.model_copy(update={"status": status})

    # ------------------------------------------------------------------
    # Internal classification helpers
    # ------------------------------------------------------------------

    def _classify(self, result: ExecutionResult) -> ExecutionStatus:
        combined_output = (
            result.stdout + result.stderr + str(result.metadata.get("nsjail_log", ""))
        )

        if self._matches_any(_SECURITY_PATTERNS, combined_output):
            return ExecutionStatus.SECURITY_VIOLATION

        if self._matches_any(_OOM_PATTERNS, combined_output):
            return ExecutionStatus.OOM

        if result.exit_code == 0:
            return ExecutionStatus.SUCCESS

        # Non-zero exit code → generic error
        return ExecutionStatus.ERROR

    @staticmethod
    def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
        return any(p.search(text) for p in patterns)


class ScoredVerifier(ResultVerifier):
    """Extends :class:`ResultVerifier` with an output-scoring step.

    Subclass this and override :meth:`score` to implement task-specific
    evaluation (e.g., comparing output against a ground-truth answer).
    """

    def verify(self, result: ExecutionResult) -> ExecutionResult:
        result = super().verify(result)
        score = self.score(result)
        metadata = {**result.metadata, "score": score}
        return result.model_copy(update={"metadata": metadata})

    def score(self, result: ExecutionResult) -> float:
        """Override in subclasses. Returns a float in [0, 1]."""
        return 1.0 if result.status == ExecutionStatus.SUCCESS else 0.0
