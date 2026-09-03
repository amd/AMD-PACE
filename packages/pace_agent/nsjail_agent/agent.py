# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Reusable LLM-driven execution-agent loop.

Vendored under ``packages/pace_agent/nsjail_agent`` alongside ``gaia_planner`` (see
``docs/nsjail-agent.md``). A future ``pace_agent/vendor/`` layout may relocate these
packages without changing this module's API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from pace_agent.toolloop import assistant_message as _assistant_message_dict

from .models import WorkspaceFile
from .tool_adapter import OPENAI_TOOL_SCHEMA, OpenAIToolAdapter
from .workspace import Workspace

ToolCallback = Callable[[Any], None]
ToolResultCallback = Callable[[Any, str], None]


DEFAULT_SYSTEM_PROMPT = """You are an expert problem-solving assistant.
You have access to a secure code execution environment via the execute_code tool.
Use it to:
- Perform calculations and data analysis
- Parse and process files
- Verify intermediate reasoning steps
- Generate and test hypotheses

When you call execute_code, write complete scripts that print the values you
need to inspect; bare Python expressions do not produce stdout in script mode.
Before saying a dependency, command, or file format is unavailable, verify it
from inside execute_code by checking imports, command paths, and error output.
If one parser fails, try another installed library, command-line utility, or
standard-library approach before giving a final answer.

Always reason step by step. On the very last line of your response write ONLY:
FINAL ANSWER: <answer>
"""


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the reusable execution-agent loop."""

    model: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 8
    force_first_tool: bool = True
    force_tool_every_turn: bool = True


@dataclass(frozen=True)
class AgentRunResult:
    """Result returned by :class:`ExecutionAgent`."""

    answer: str
    messages: list[dict[str, Any]] = field(default_factory=list)


class WorkspaceExecutor:
    """Executor wrapper that keeps one workspace across multiple tool calls."""

    def __init__(self, wrapped_executor: Any, workspace: Workspace) -> None:
        self._wrapped_executor = wrapped_executor
        self._workspace = workspace

    def execute_sync(self, request: Any) -> Any:
        if hasattr(self._wrapped_executor, "execute_sync_in_workspace"):
            return self._wrapped_executor.execute_sync_in_workspace(
                request, self._workspace
            )
        return self._wrapped_executor.execute_sync(request)


class ExecutionAgent:
    """Execution agent driven by PACE Agent ``LLMClient.chat`` or an OpenAI SDK client.

    Prefer passing a PACE Agent :class:`~pace_agent.llm.client.LLMClient` (or any object
    exposing ``chat(messages, tools=..., tool_choice=...)``). An OpenAI SDK client
    (``client.chat.completions.create``) is still accepted for lightweight examples.
    """

    def __init__(self, *, client: Any, executor: Any, config: AgentConfig) -> None:
        self._client = client
        self._executor = executor
        self._config = config

    def run(
        self,
        question: str,
        *,
        files: list[WorkspaceFile] | None = None,
        on_tool_call: ToolCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> AgentRunResult:
        files = files or []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user", "content": question + self._file_note(files)},
        ]

        with Workspace() as task_workspace:
            adapter = OpenAIToolAdapter(
                executor=WorkspaceExecutor(self._executor, task_workspace),
            )

            for turn in range(self._config.max_turns):
                should_force_tool = self._config.force_tool_every_turn or (
                    self._config.force_first_tool and turn == 0
                )
                tool_choice: str | dict[str, Any] = (
                    {"type": "function", "function": {"name": "execute_code"}}
                    if should_force_tool
                    else "auto"
                )

                turn_msg = self._chat_turn(messages, tool_choice=tool_choice)
                tool_calls = turn_msg.get("tool_calls") or []
                content = turn_msg.get("content") or ""

                if not tool_calls:
                    if should_force_tool:
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Call execute_code. Do not answer directly during tool "
                                    "iterations."
                                ),
                            }
                        )
                        continue
                    return AgentRunResult(answer=content, messages=messages)

                messages.append(_assistant_message_dict(turn_msg))

                for tool_call in tool_calls:
                    callback_obj = _tool_call_callback_view(tool_call)
                    if on_tool_call is not None:
                        on_tool_call(callback_obj)
                    try:
                        arguments = _tool_arguments_with_files(tool_call, files)
                    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                        # Models do emit malformed tool arguments. Hand the error
                        # back as the tool result so the loop can recover instead
                        # of aborting the whole run.
                        output = f"[tool error] could not parse tool arguments: {exc}"
                    else:
                        output = adapter(arguments)
                    if on_tool_result is not None:
                        on_tool_result(callback_obj, output)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(tool_call),
                            "content": output,
                        }
                    )

            final_turn = self._chat_turn(messages, tool_choice="none")
            answer = (
                final_turn.get("content") or "[max_turns reached without final answer]"
            )
            return AgentRunResult(answer=answer, messages=messages)

    def _chat_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
    ) -> dict[str, Any]:
        """One model turn via PACE Agent ``LLMClient.chat`` or an OpenAI-compatible client."""
        tools = [OPENAI_TOOL_SCHEMA]
        if _uses_pace_agent_chat_api(self._client):
            return self._client.chat(messages, tools=tools, tool_choice=tool_choice)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return _assistant_message_dict(response.choices[0].message)

    @staticmethod
    def _file_note(files: list[WorkspaceFile]) -> str:
        if not files:
            return ""
        names = ", ".join(file.name for file in files)
        return f"\n\nAttached files are available in the sandbox working directory: {names}."


def _uses_pace_agent_chat_api(client: Any) -> bool:
    """True when *client* exposes PACE Agent-style ``chat(messages, ...)`` (not OpenAI SDK)."""
    chat = getattr(client, "chat", None)
    if chat is None or not callable(chat):
        return False
    return not hasattr(chat, "completions")


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(tool_call.id)


def _tool_call_callback_view(tool_call: Any) -> Any:
    """SDK-shaped view of a normalized or SDK tool call for legacy callbacks."""
    if not isinstance(tool_call, dict):
        return tool_call
    fn = tool_call.get("function") or tool_call
    if isinstance(fn, dict):
        name = fn.get("name", "")
        arguments = fn.get("arguments", "{}")
    else:
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", "{}")
    return SimpleNamespace(
        id=tool_call.get("id"),
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_arguments_with_files(
    tool_call: Any, files: list[WorkspaceFile]
) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or tool_call
        raw = (
            fn.get("arguments")
            if isinstance(fn, dict)
            else tool_call.get("arguments", "{}")
        )
    else:
        raw = tool_call.function.arguments
    arguments = json.loads(raw)
    if files:
        arguments["files"] = [
            *arguments.get("files", []),
            *[file.model_dump() for file in files],
        ]
    return arguments
