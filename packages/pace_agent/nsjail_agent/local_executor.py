# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Local (unsandboxed) executor for development and testing.

WARNING: This executor runs code directly in the host Python environment with
NO isolation. Never use it with untrusted code in production.

Why a separate non-sandbox runtime exists:

- :class:`~nsjail_agent.executor.CodeExecutor` needs Linux + an nsjail binary.
  ``LocalExecutor`` lets examples, unit tests, and dev workflows run where nsjail
  is unavailable (macOS/Windows, or Linux without the binary configured).
- :func:`make_executor` falls back here only when nsjail is missing. GAIA's
  ``ExecutionAdapter`` rejects it unless ``--allow-local-executor`` is passed.

Production workloads must use :class:`CodeExecutor`. Multi-language support
(bash, JavaScript, Go) lives in the nsjail sandbox; ``LocalExecutor`` runs
**Python only**.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import sys
import time
import warnings

from .models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Language,
)
from .workspace import Workspace
from .verifier import ResultVerifier

logger = logging.getLogger(__name__)

# LocalExecutor is Python-only. Bash/JS/Go are supported by CodeExecutor (nsjail).
_PYTHON_CMD: list[str] = [sys.executable, "-u"]
_PYTHON_ENTRYPOINT = "main.py"

_WARNED = False


class LocalExecutor:
    """Runs Python code directly via subprocess — **no sandboxing whatsoever**.

    Drop-in replacement for :class:`~nsjail_agent.executor.CodeExecutor` on
    dev hosts without nsjail. Only ``Language.PYTHON`` is supported; request
    bash/JavaScript/Go through :class:`CodeExecutor` instead.

    Example::

        from nsjail_agent.local_executor import LocalExecutor
        executor = LocalExecutor()
        result = executor.execute_sync(ExecutionRequest(code="print('hi')"))
    """

    def __init__(self, verifier: ResultVerifier | None = None) -> None:
        self._verifier = verifier or ResultVerifier()
        _warn_once()

    # ------------------------------------------------------------------
    # Public API (mirrors CodeExecutor)
    # ------------------------------------------------------------------

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, request)

    def execute_sync(self, request: ExecutionRequest) -> ExecutionResult:
        return self._execute_sync(request)

    def execute_sync_in_workspace(
        self,
        request: ExecutionRequest,
        workspace: Workspace,
    ) -> ExecutionResult:
        return self._execute_sync_in_workspace(request, workspace)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_sync(self, request: ExecutionRequest) -> ExecutionResult:
        with Workspace() as ws:
            return self._execute_sync_in_workspace(request, ws)

    def _execute_sync_in_workspace(
        self,
        request: ExecutionRequest,
        ws: Workspace,
    ) -> ExecutionResult:
        for file in request.files:
            if file.content_base64 is not None:
                ws.write_binary_file(file.name, base64.b64decode(file.content_base64))
            else:
                ws.write_file(file.name, file.content)

        if request.language != Language.PYTHON:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=(
                    f"LocalExecutor supports Python only (got {request.language.value!r}). "
                    "Install/configure nsjail and use CodeExecutor for bash, JavaScript, "
                    "or Go."
                ),
                wall_time_ms=0.0,
            )

        entrypoint = _PYTHON_ENTRYPOINT
        ws.write_file(entrypoint, request.code)

        cmd = [*_PYTHON_CMD, str(ws.host_path / entrypoint)]
        logger.debug("LocalExecutor running: %s", " ".join(cmd))
        start = time.monotonic()

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(ws.host_path),
            )
            stdin_data = request.stdin.encode() if request.stdin else None

            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    input=stdin_data,
                    timeout=request.resources.max_wall_seconds,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                # Bound the wait: a process ignoring SIGKILL (uninterruptible
                # I/O) would otherwise hang the caller past its own timeout.
                try:
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    stdout_bytes, stderr_bytes = b"", b""
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    # A runaway process is exactly the case that produces huge
                    # output, so the cap applies here too.
                    stdout=_truncate(stdout_bytes, request.resources.max_output_bytes),
                    stderr=_truncate(stderr_bytes, request.resources.max_output_bytes),
                    wall_time_ms=(time.monotonic() - start) * 1000,
                )

        except (OSError, FileNotFoundError) as exc:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to launch interpreter: {exc}",
                wall_time_ms=0.0,
            )

        wall_ms = (time.monotonic() - start) * 1000
        max_bytes = request.resources.max_output_bytes

        raw = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            exit_code=proc.returncode,
            stdout=_truncate(stdout_bytes, max_bytes),
            stderr=_truncate(stderr_bytes, max_bytes),
            wall_time_ms=wall_ms,
            output_files=ws.collect_files(request.return_files, max_bytes=max_bytes),
        )
        return self._verifier.verify(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(data: bytes, max_bytes: int) -> str:
    if len(data) > max_bytes:
        data = data[:max_bytes] + b"\n[output truncated]"
    return data.decode("utf-8", errors="replace")


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "LocalExecutor runs code WITHOUT sandboxing. "
            "Use CodeExecutor (nsjail) for production workloads.",
            stacklevel=3,
        )
        _WARNED = True


def make_executor(prefer_sandbox: bool = True) -> "LocalExecutor | object":
    """Return a ``CodeExecutor`` when nsjail is available, else ``LocalExecutor``.

    The fallback is for development and tests only. GAIA rejects unsandboxed
    execution unless ``--allow-local-executor`` is set.
    """
    if prefer_sandbox:
        try:
            from .sandbox import check_nsjail_available
            from .executor import CodeExecutor

            check_nsjail_available()
            return CodeExecutor()
        except (RuntimeError, ImportError):
            logger.info("nsjail not available — falling back to LocalExecutor")
    return LocalExecutor()
