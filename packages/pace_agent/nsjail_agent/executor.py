# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""High-level code executor — orchestrates workspace, sandbox, and verification."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from typing import BinaryIO

from .models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from .sandbox import (
    build_nsjail_cmd,
    check_nsjail_available,
    _ENTRYPOINT_NAME,
)
from .verifier import ResultVerifier
from .workspace import Workspace

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Runs code inside an nsjail sandbox and returns a structured result.

    Example::

        executor = CodeExecutor()
        result = await executor.execute(
            ExecutionRequest(code="print('hello')", language=Language.PYTHON)
        )
        print(result.stdout)   # hello
    """

    def __init__(self, verifier: ResultVerifier | None = None) -> None:
        self._verifier = verifier or ResultVerifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute *request* inside nsjail and return the result.

        This is the primary async entry-point. It runs the blocking subprocess
        call in a thread-pool executor so it does not block the event loop.
        """
        check_nsjail_available()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, request)

    def execute_sync(self, request: ExecutionRequest) -> ExecutionResult:
        """Synchronous variant — use this from non-async contexts."""
        check_nsjail_available()
        return self._execute_sync(request)

    def execute_sync_in_workspace(
        self,
        request: ExecutionRequest,
        workspace: Workspace,
    ) -> ExecutionResult:
        """Synchronous execution inside an existing workspace.

        Use this for multi-turn agents where files created by one tool call
        should remain visible to later tool calls.
        """
        check_nsjail_available()
        return self._execute_sync_in_workspace(request, workspace)

    # ------------------------------------------------------------------
    # Internal implementation
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
                import base64

                ws.write_binary_file(file.name, base64.b64decode(file.content_base64))
            else:
                ws.write_file(file.name, file.content)

        # Write entrypoint
        entrypoint_name = _ENTRYPOINT_NAME[request.language]
        ws.write_file(entrypoint_name, request.code)

        max_bytes = request.resources.max_output_bytes

        with tempfile.TemporaryFile() as nsjail_log:
            log_fd = nsjail_log.fileno()

            # Build nsjail command — pass a writable fd so nsjail diagnostics
            # are captured separately from the user's stderr.
            nsjail_cmd = build_nsjail_cmd(
                workspace_host_path=ws.host_path,
                language=request.language,
                resources=request.resources,
                config=request.sandbox,
                log_fd=log_fd,
            )

            logger.debug("Launching nsjail: %s", " ".join(nsjail_cmd))
            start = time.monotonic()

            try:
                proc = subprocess.Popen(
                    nsjail_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(log_fd,),
                    close_fds=True,
                )

                stdin_data = request.stdin.encode() if request.stdin else None
                wall_limit = request.resources.max_wall_seconds

                try:
                    stdout_bytes, stderr_bytes = proc.communicate(
                        input=stdin_data,
                        timeout=wall_limit,
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout_bytes, stderr_bytes = proc.communicate()
                    wall_ms = (time.monotonic() - start) * 1000
                    nsjail_log_text = _read_log_file(nsjail_log, max_bytes)
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        exit_code=-1,
                        stdout=_decode_truncated(stdout_bytes, max_bytes),
                        stderr=_decode_truncated(stderr_bytes, max_bytes),
                        wall_time_ms=wall_ms,
                        metadata=(
                            {"nsjail_log": nsjail_log_text} if nsjail_log_text else {}
                        ),
                    )

            except OSError as exc:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Failed to launch nsjail: {exc}",
                    wall_time_ms=0.0,
                )

            wall_ms = (time.monotonic() - start) * 1000
            nsjail_log_text = _read_log_file(nsjail_log, max_bytes)

            raw_result = ExecutionResult(
                status=ExecutionStatus.SUCCESS,  # verifier may override
                exit_code=proc.returncode,
                stdout=_decode_truncated(stdout_bytes, max_bytes),
                stderr=_decode_truncated(stderr_bytes, max_bytes),
                wall_time_ms=wall_ms,
                output_files=ws.collect_files(
                    request.return_files, max_bytes=max_bytes
                ),
                metadata={"nsjail_log": nsjail_log_text} if nsjail_log_text else {},
            )

            return self._verifier.verify(raw_result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_truncated(data: bytes, max_bytes: int) -> str:
    """UTF-8 decode *data*, truncating at *max_bytes* with a warning suffix."""
    if len(data) > max_bytes:
        data = data[:max_bytes]
        suffix = b"\n[output truncated]"
        data = data + suffix
    return data.decode("utf-8", errors="replace")


def _read_log_file(log_file: BinaryIO, max_bytes: int) -> str:
    """Read nsjail diagnostics captured in a temporary file."""
    log_file.seek(0)
    return _decode_truncated(log_file.read(max_bytes + 1), max_bytes)
