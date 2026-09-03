# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""GAIA sample-task agent example using the vendored nsjail executor.

Requires an installed ``pace-agent`` package (so ``nsjail_agent`` imports) and
endpoint credentials via the environment (``LLM_GATEWAY_KEY`` /
``LLM_GATEWAY_BASE_URL`` / ``LLM_GATEWAY_MODEL``, or the OpenAI equivalents).
nsjail is used when available; otherwise ``LocalExecutor`` is used with a
warning (dev only).

Usage (from the repo root, after ``pip install -e ./packages/pace_agent``)::

    python -m pace_agent.examples.gaia_agent_example --max-tasks 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from pace_agent.datasets import read_json_or_jsonl
from openai import APIStatusError, OpenAI

from nsjail_agent.agent import AgentConfig, ExecutionAgent
from nsjail_agent.local_executor import make_executor
from nsjail_agent.models import WorkspaceFile

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "gaia"
# Task list lives with the other PACE Agent example configs; attachments stay under fixtures/.
_PACE_AGENT_DIST_ROOT = Path(__file__).resolve().parents[2]
TASKS_PATH = _PACE_AGENT_DIST_ROOT / "configs" / "gaia_sample_tasks.json"

GAIA_SYSTEM_PROMPT = """You are an expert problem-solving assistant participating in the GAIA benchmark.
You have access to a secure Python execution environment via the execute_code tool.
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


def _build_client() -> OpenAI:
    """Build an OpenAI-compatible client from the environment (no hardcoded endpoint)."""
    api_key = os.environ.get("LLM_GATEWAY_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("LLM_GATEWAY_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", ""
    )
    if not api_key:
        raise SystemExit(
            "ERROR: Set LLM_GATEWAY_KEY or OPENAI_API_KEY before running this example."
        )
    if not base_url:
        raise SystemExit(
            "ERROR: Set LLM_GATEWAY_BASE_URL or OPENAI_BASE_URL before running this example."
        )
    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if os.environ.get("LLM_GATEWAY_KEY"):
        client_kwargs["default_headers"] = {
            "Ocp-Apim-Subscription-Key": os.environ["LLM_GATEWAY_KEY"],
        }
    return OpenAI(**client_kwargs)


def load_sample_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    """Load GAIA sample tasks from JSON."""
    data = read_json_or_jsonl(path)
    for task in data:
        if not isinstance(task.get("task_id"), str) or not isinstance(
            task.get("question"), str
        ):
            raise ValueError(
                f"Each task in {path} requires string task_id and question fields"
            )
    return data


def _print_tool_call_code(tool_call) -> None:
    """Print generated code from an execute_code tool call before running it."""
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        print(f"  [tool call arguments]\n{tool_call.function.arguments}")
        return

    language = arguments.get("language", "python")
    code = arguments.get("code", "")
    if code:
        print(f"\n  [generated code: {language}]")
        print("  " + "-" * 56)
        print(code)
        print("  " + "-" * 56)


def _print_tool_result(tool_call, output: str) -> None:
    if "[status: success]" in output and len(output) > 120:
        print(f"  [tool call] {tool_call.function.name}  ->  {output[:120]}...")
    else:
        print(f"  [tool call] {tool_call.function.name}  ->  {output}")


def _workspace_files_for_task(task_files: list[str]) -> list[WorkspaceFile]:
    workspace_files: list[WorkspaceFile] = []
    for filename in task_files:
        path = FIXTURES_DIR / Path(filename).name
        if not path.exists():
            print(f"  [warning] task file not found: {path}")
            continue
        workspace_files.append(
            WorkspaceFile(
                name=path.name,
                content="",
                content_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
            )
        )
    return workspace_files


def _format_gateway_error(exc: APIStatusError, model: str) -> str:
    message = (
        f"[gateway error] HTTP {exc.status_code} while using model {model!r}: {exc}"
    )
    if exc.status_code == 426:
        message += (
            "\nSet LLM_GATEWAY_MODEL or pass --model with a current deployment/model "
            "name, then rerun the script."
        )
    return message


def solve_gaia_task(
    question: str,
    *,
    client: OpenAI,
    executor: Any,
    model: str,
    task_files: list[str] | None = None,
    max_turns: int = 8,
    force_first_tool: bool = True,
    force_tool_every_turn: bool = True,
) -> str:
    """Run the agent loop for a single GAIA task and return the full response."""
    workspace_files = _workspace_files_for_task(task_files or [])
    agent = ExecutionAgent(
        client=client,
        executor=executor,
        config=AgentConfig(
            model=model,
            system_prompt=GAIA_SYSTEM_PROMPT,
            max_turns=max_turns,
            force_first_tool=force_first_tool,
            force_tool_every_turn=force_tool_every_turn,
        ),
    )
    try:
        result = agent.run(
            question,
            files=workspace_files,
            on_tool_call=_print_tool_call_code,
            on_tool_result=_print_tool_result,
        )
    except APIStatusError as exc:
        return _format_gateway_error(exc, model)
    return result.answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GAIA-style sample tasks with tool calls."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_GATEWAY_MODEL") or os.environ.get("OPENAI_MODEL"),
        help="Model id (or set LLM_GATEWAY_MODEL / OPENAI_MODEL).",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit the number of sample tasks to run.",
    )
    parser.add_argument(
        "--task-id", help="Run a single sample task by id, e.g. gaia_015."
    )
    parser.add_argument(
        "--allow-direct-first-answer",
        action="store_true",
        help="Allow the model to answer without a first execute_code tool call.",
    )
    parser.add_argument(
        "--allow-auto-tool-choice",
        action="store_true",
        help="Allow tool_choice='auto' during tool-capable turns instead of forcing execute_code.",
    )
    return parser.parse_args()


def extract_final_answer(response: str) -> str | None:
    """Pull the 'FINAL ANSWER: ...' line out of the model's response."""
    for line in reversed(response.splitlines()):
        line = line.strip()
        if line.upper().startswith("FINAL ANSWER:"):
            return line.split(":", 1)[1].strip()
    return None


if __name__ == "__main__":
    args = parse_args()
    if not args.model:
        print(
            "ERROR: Pass --model or set LLM_GATEWAY_MODEL / OPENAI_MODEL.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = _build_client()
    executor = make_executor()
    print(f"Using model: {args.model}")

    correct = 0
    sample_tasks = load_sample_tasks()
    max_tasks = args.max_tasks if args.max_tasks is not None else len(sample_tasks)
    tasks = (
        [task for task in sample_tasks if task["task_id"] == args.task_id]
        if args.task_id
        else sample_tasks[:max_tasks]
    )
    if not tasks:
        print("No tasks selected.")
        sys.exit(0)

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Task {task['task_id']}: {task['question']}")
        print("-" * 60)

        full_response = solve_gaia_task(
            task["question"],
            client=client,
            executor=executor,
            model=args.model,
            task_files=task.get("files", []),
            max_turns=args.max_turns,
            force_first_tool=not args.allow_direct_first_answer,
            force_tool_every_turn=not (
                args.allow_auto_tool_choice or args.allow_direct_first_answer
            ),
        )
        answer = extract_final_answer(full_response) or full_response.strip()

        expected = task.get("expected", "")
        match = expected and (expected.lower() in answer.lower())
        if match:
            correct += 1
        status = "✓" if match else "✗"

        print(f"Answer   : {answer}")
        if expected:
            print(f"Expected : {expected}  {status}")

    print(f"\n{'=' * 60}")
    print(f"Score: {correct}/{len(tasks)} ({100 * correct // len(tasks)}%)")
