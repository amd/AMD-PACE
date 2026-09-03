# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Optional FastAPI REST interface for remote nsjail execution.

PACE Agent and GAIA run code **in-process** via :class:`~nsjail_agent.executor.CodeExecutor`
(see ``ExecutionAdapter`` / ``sandbox_tool_agent``). This module is a standalone
remote-execution server for deployments that want HTTP instead of embedding the
executor in the agent process.

Start the server::

    uvicorn nsjail_agent.api:app --host 0.0.0.0 --port 8080

Or via the CLI helper::

    python -m nsjail_agent.api

Security defaults:

- CORS is **disabled** unless ``NSJAIL_AGENT_CORS_ORIGINS`` is set.
- Set ``NSJAIL_AGENT_API_KEY`` before exposing ``/execute`` outside localhost.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .executor import CodeExecutor
from .models import ExecutionRequest, ExecutionResult, ResourceLimits, SandboxConfig

logger = logging.getLogger(__name__)

_executor: CodeExecutor | None = None


def _max_concurrent_executions() -> int:
    """Parse ``NSJAIL_AGENT_MAX_CONCURRENT``; clamp invalid/zero values to a safe default."""
    raw = os.getenv("NSJAIL_AGENT_MAX_CONCURRENT", "4")
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid NSJAIL_AGENT_MAX_CONCURRENT=%r; using 4", raw)
        value = 4
    if value < 1:
        logger.warning("NSJAIL_AGENT_MAX_CONCURRENT=%r is < 1; using 1", raw)
        return 1
    return value


_execution_semaphore = asyncio.Semaphore(_max_concurrent_executions())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _executor
    _executor = CodeExecutor()
    logger.info("nsjail-agent-tool API started")
    yield
    _executor = None
    logger.info("nsjail-agent-tool API stopped")


app = FastAPI(
    title="nsjail-agent-tool",
    description="Secure code execution sandbox for AI agents, powered by nsjail.",
    version="0.1.0",
    lifespan=lifespan,
)


def _cors_allow_origins() -> list[str]:
    """Return explicit CORS origins from ``NSJAIL_AGENT_CORS_ORIGINS`` (comma-separated).

    Empty by default — cross-origin browser access is disabled unless configured.
    """
    raw = os.getenv("NSJAIL_AGENT_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_cors_origins = _cors_allow_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200 if the server is up."""
    return {"status": "ok"}


@app.post(
    "/execute",
    response_model=ExecutionResult,
    tags=["execution"],
    summary="Execute code in a sandboxed environment",
)
async def execute(
    request: ExecutionRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> ExecutionResult:
    """Run *request.code* inside nsjail and return the result.

    Sandbox policy is server-owned: request-provided sandbox settings are
    normalized before execution.
    """
    _authorize_execute(authorization, x_api_key)
    if _executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Executor not initialised",
        )
    try:
        async with _execution_semaphore:
            return await _executor.execute(_apply_server_policy(request))
    except Exception as exc:
        logger.exception("Unhandled error during execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Execution failed",
        ) from exc


@app.get("/schema/openai-tool", tags=["meta"])
async def openai_tool_schema() -> dict:
    """Return the OpenAI function-calling schema for the execute_code tool."""
    from .tool_adapter import OPENAI_TOOL_SCHEMA

    return OPENAI_TOOL_SCHEMA


def _apply_server_policy(request: ExecutionRequest) -> ExecutionRequest:
    """Keep sandbox *and* resource controls server-owned for remote requests.

    A client could otherwise ask for the top of every ``ResourceLimits`` range
    (600 s wall, 4 GB, 512 pids) and pin the host far longer than the server's
    own defaults intend, so resources are clamped to the server ceiling rather
    than taken on trust.
    """
    limits = _server_resource_ceiling()
    requested = request.resources
    clamped = ResourceLimits(
        max_cpu_seconds=min(requested.max_cpu_seconds, limits.max_cpu_seconds),
        max_wall_seconds=min(requested.max_wall_seconds, limits.max_wall_seconds),
        max_memory_mb=min(requested.max_memory_mb, limits.max_memory_mb),
        max_output_bytes=min(requested.max_output_bytes, limits.max_output_bytes),
        max_pids=min(requested.max_pids, limits.max_pids),
        max_fsize_mb=min(requested.max_fsize_mb, limits.max_fsize_mb),
    )
    return request.model_copy(update={"sandbox": SandboxConfig(), "resources": clamped})


def _server_resource_ceiling() -> ResourceLimits:
    """Per-request resource ceiling, overridable via ``NSJAIL_AGENT_MAX_*`` env vars."""
    defaults = ResourceLimits()
    overrides: dict[str, int] = {}
    for field, env_name in (
        ("max_cpu_seconds", "NSJAIL_AGENT_MAX_CPU_SECONDS"),
        ("max_wall_seconds", "NSJAIL_AGENT_MAX_WALL_SECONDS"),
        ("max_memory_mb", "NSJAIL_AGENT_MAX_MEMORY_MB"),
        ("max_output_bytes", "NSJAIL_AGENT_MAX_OUTPUT_BYTES"),
        ("max_pids", "NSJAIL_AGENT_MAX_PIDS"),
        ("max_fsize_mb", "NSJAIL_AGENT_MAX_FSIZE_MB"),
    ):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            overrides[field] = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default", env_name, raw)
    if not overrides:
        return defaults
    return defaults.model_copy(update=overrides)


def _auth_is_configured() -> bool:
    return bool((os.getenv("NSJAIL_AGENT_API_KEY") or "").strip())


def _anonymous_allowed() -> bool:
    """Explicit opt-out of auth, for loopback development only."""
    return (os.getenv("NSJAIL_AGENT_ALLOW_ANONYMOUS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _authorize_execute(
    authorization: str | None,
    x_api_key: str | None,
) -> None:
    expected = (os.getenv("NSJAIL_AGENT_API_KEY") or "").strip()
    if not expected:
        if _anonymous_allowed():
            return
        # Fail closed. /execute runs arbitrary code, so an unset (or blank) key
        # must refuse the request rather than serve it anonymously - a missing
        # env var in a deployment would otherwise be remote code execution.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Execution endpoint is disabled: set NSJAIL_AGENT_API_KEY to "
                "enable it, or set NSJAIL_AGENT_ALLOW_ANONYMOUS=1 to accept "
                "unauthenticated requests (loopback development only)."
            ),
        )

    bearer = f"Bearer {expected}"
    if _secrets_match(authorization, bearer) or _secrets_match(x_api_key, expected):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
    )


def _secrets_match(provided: str | None, expected: str) -> bool:
    """Constant-time string comparison for auth secrets."""
    if provided is None:
        return False
    a = provided.encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Loopback by default: this server executes arbitrary code, so exposing it
    # on every interface has to be a deliberate act, not the default.
    host = os.getenv("NSJAIL_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("NSJAIL_AGENT_PORT", "8080"))
    if host not in ("127.0.0.1", "::1", "localhost") and not _auth_is_configured():
        raise SystemExit(
            f"Refusing to listen on {host} without NSJAIL_AGENT_API_KEY set: "
            "/execute runs arbitrary code. Set an API key, or bind 127.0.0.1."
        )
    uvicorn.run("nsjail_agent.api:app", host=host, port=port, reload=False)
