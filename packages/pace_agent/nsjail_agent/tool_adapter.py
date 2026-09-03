# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Framework adapters that expose the executor as a callable agent tool.

Each adapter converts the nsjail executor into the tool-calling contract
expected by a specific agent framework (LangChain/LangGraph, OpenAI, CAMEL).

Related wrappers (not defined here):

- :class:`nsjail_agent.agent.WorkspaceExecutor` + :class:`ExecutionAgent` keep one
  workspace across tool calls and parse OpenAI ``tool_calls`` in the agent loop.
- :class:`gaia_planner.adapters.ExecutionAdapter` wraps :class:`OpenAIToolAdapter`
  with a persistent task workspace and preprocesses ``execute_code`` arguments via
  ``_prepare_execute_arguments`` (Python import preamble, last-expression echo).
"""

from __future__ import annotations

import json
from typing import Any

from typing import Protocol, runtime_checkable

from .models import ExecutionRequest, ExecutionResult, Language


@runtime_checkable
class AnyExecutor(Protocol):
    """Structural type accepted by all adapters (CodeExecutor or LocalExecutor)."""

    def execute_sync(self, request: ExecutionRequest) -> ExecutionResult: ...
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _coerce_tool_input(
    tool_input: str | dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse OpenAI tool arguments; return ``(payload, None)`` or ``(None, error_text)``."""
    if isinstance(tool_input, dict):
        return tool_input, None
    try:
        parsed = json.loads(tool_input)
    except json.JSONDecodeError as exc:
        return None, f"[tool error] invalid JSON arguments: {exc.msg} (char {exc.pos})"
    if not isinstance(parsed, dict):
        return None, "[tool error] tool arguments must be a JSON object"
    return parsed, None


def _build_request(tool_input: dict[str, Any]) -> ExecutionRequest:
    """Convert a generic tool-call payload dict into an ExecutionRequest."""
    return ExecutionRequest(
        code=tool_input["code"],
        language=Language(tool_input.get("language", "python")),
        stdin=tool_input.get("stdin", ""),
        files=tool_input.get("files", []),
        return_files=tool_input.get("return_files", []),
    )


def _result_to_text(result: ExecutionResult) -> str:
    """Render an ExecutionResult as a plain-text string suitable for LLM consumption."""
    parts: list[str] = [f"[status: {result.status.value}]"]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.output_files:
        for name, content in result.output_files.items():
            if content.startswith("[base64:"):
                marker = content.splitlines()[0]
                parts.append(
                    f"FILE {name}: {marker} "
                    "(binary file returned; print textual analysis/results to stdout)"
                )
            else:
                parts.append(f"FILE {name}:\n{content}")
    nsjail_log = result.metadata.get("nsjail_log")
    if nsjail_log and result.status.value != "success":
        parts.append(f"NSJAIL LOG:\n{nsjail_log}")
    parts.append(
        f"(exit_code={result.exit_code}, wall_time={result.wall_time_ms:.0f}ms)"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI / function-calling schema
# ---------------------------------------------------------------------------

OPENAI_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Executes code in a secure nsjail sandbox and returns stdout, stderr, "
            "and optional output files. Supports Python, Bash, JavaScript, and Go. "
            "Use it to inspect attached files, verify available imports/commands, "
            "and try alternative parsers before concluding a dependency is missing. "
            "STATELESS EXECUTION: every call runs in a brand-new process. Variables, "
            "imports, and any in-memory state from previous calls are GONE - referencing "
            "them raises NameError. Files on disk persist across calls only when the "
            "caller supplies a shared workspace (ExecutionAgent and GAIA ExecutionAdapter "
            "do this); standalone adapter use creates a fresh workspace per call. Task "
            "attachments seeded by the host are available from the first call. Write each "
            "call as ONE self-contained script that re-imports what it needs, re-reads "
            "any files, and print()s the final result (bare expressions produce no output)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to execute.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "bash", "javascript", "go"],
                    "description": "Programming language of the code.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Optional text to pipe to stdin.",
                },
                "return_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace file paths to return after execution.",
                },
            },
            "required": ["code"],
        },
    },
}


class OpenAIToolAdapter:
    """Wraps the executor so it can be called from an OpenAI tool-calling loop."""

    def __init__(self, executor: AnyExecutor | None = None) -> None:
        if executor is None:
            from .local_executor import make_executor

            executor = make_executor()
        self._executor = executor

    @property
    def schema(self) -> dict[str, Any]:
        return OPENAI_TOOL_SCHEMA

    def __call__(self, tool_input: str | dict[str, Any]) -> str:
        """Synchronous call — safe to use from a standard OpenAI tool-call handler."""
        parsed, err = _coerce_tool_input(tool_input)
        if err is not None:
            return err
        request = _build_request(parsed)
        result = self._executor.execute_sync(request)
        return _result_to_text(result)

    async def acall(self, tool_input: str | dict[str, Any]) -> str:
        """Async variant for use with async OpenAI clients."""
        parsed, err = _coerce_tool_input(tool_input)
        if err is not None:
            return err
        request = _build_request(parsed)
        result = await self._executor.execute(request)
        return _result_to_text(result)


# ---------------------------------------------------------------------------
# LangChain / LangGraph adapter
# ---------------------------------------------------------------------------


def make_langchain_tool(executor: AnyExecutor | None = None):  # type: ignore[return]
    """Return a LangChain ``Tool`` that runs code in nsjail.

    Requires ``langchain-core`` to be installed.
    """
    try:
        from langchain_core.tools import tool as lc_tool  # type: ignore
    except ImportError as exc:
        raise ImportError("Install langchain-core: pip install langchain-core") from exc

    if executor is None:
        from .local_executor import make_executor

        executor = make_executor()
    _executor = executor

    @lc_tool
    def execute_code(code: str, language: str = "python", stdin: str = "") -> str:
        """Execute code in a secure nsjail sandbox.

        Args:
            code: Source code to run.
            language: One of python, bash, javascript, go.
            stdin: Optional stdin data.

        Returns:
            Combined stdout/stderr output as a string.
        """
        request = _build_request({"code": code, "language": language, "stdin": stdin})
        result = _executor.execute_sync(request)
        return _result_to_text(result)

    return execute_code


# ---------------------------------------------------------------------------
# CAMEL adapter
# ---------------------------------------------------------------------------


class CAMELToolAdapter:
    """Adapter for the CAMEL agent framework's OpenAIFunction interface.

    CAMEL expects tools to expose ``.openai_tool_schema`` and be callable.
    """

    def __init__(self, executor: AnyExecutor | None = None) -> None:
        if executor is None:
            from .local_executor import make_executor

            executor = make_executor()
        self._executor = executor

    @property
    def openai_tool_schema(self) -> dict[str, Any]:
        return OPENAI_TOOL_SCHEMA

    def __call__(self, **kwargs: Any) -> str:
        request = _build_request(kwargs)
        result = self._executor.execute_sync(request)
        return _result_to_text(result)

    # CAMEL also supports async calls via __acall__
    async def __acall__(self, **kwargs: Any) -> str:
        request = _build_request(kwargs)
        result = await self._executor.execute(request)
        return _result_to_text(result)
