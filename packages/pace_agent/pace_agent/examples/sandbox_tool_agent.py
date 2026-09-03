# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PACE Agent example: an OpenAI tool-calling agent that runs code in a sandbox.

This is the PACE Agent-native template for integrating the AgenticAI ``nsjail-agent-tool``
executor (and the ``gaia-planner`` loop it powers). The agent is a plain
``run(llm, task)`` tool-calling loop — no LangGraph — driven by :func:`pace_agent.toolloop.run_tool_loop`.

Two integration points are shown:

- **execute_code** is registered as a PACE Agent op with ``side_effect_free=False`` because a
  real sandbox mutates a workspace / spawns a subprocess and is non-deterministic. PACE Agent
  records its output and **reuses it on replay instead of re-executing it**, so
  ``pace-agent optimize`` never re-runs the sandbox. Its wall time is attributed (bucket
  ``tool_io``) but, correctly, never claimed as CPU-optimizable.

- the executor is **pluggable** via :func:`set_executor`. Out of the box a tiny deterministic
  in-process executor keeps the example runnable/offline; wire the real sandbox with::

      from nsjail_agent import CodeExecutor, ExecutionRequest, Language
      from pace_agent.examples import sandbox_tool_agent as sta
      ex = CodeExecutor()
      sta.set_executor(lambda code, language="python", stdin="": ex.execute_sync(
          ExecutionRequest(code=code, language=Language(language), stdin=stdin)
      ).stdout)
"""

import io
import contextlib

from pace_agent.registry import op as _register_op
from pace_agent.toolloop import parse_arguments, run_tool_loop

NAME = "sandbox_tool_agent"
KIND = "standard"
QUESTION = "Compute the sum of the first 100 positive integers using code."

# The OpenAI tool schema the model sees. Mirrors nsjail-agent-tool's OPENAI_TOOL_SCHEMA so a
# real integration can drop in that exact schema unchanged.
EXECUTE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute code in a secure sandbox and return its stdout. Supports Python. "
            "Stateless: each call is a fresh process. Write one self-contained script and "
            "print() the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code to execute."},
                "language": {
                    "type": "string",
                    "enum": ["python"],
                    "description": "Programming language (only python in this example).",
                },
            },
            "required": ["code"],
        },
    },
}


def _default_executor(code, language="python", stdin=""):
    """Deterministic, dependency-free in-process fallback so the example runs offline.

    NOT a security boundary — it merely `exec`s trusted example code to produce stdout.
    Replace it with the real nsjail sandbox via :func:`set_executor` for any real use.
    """
    import sys

    if language != "python":
        return f"[unsupported language: {language}]"
    buf = io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        with contextlib.redirect_stdout(buf):
            # nosec - trusted example code only; real use goes through nsjail
            exec(compile(code, "<sandbox>", "exec"), {"__name__": "__main__"})
    # noqa: BLE001 - surface the error to the model like a sandbox would
    except Exception as exc:
        return f"[error] {type(exc).__name__}: {exc}\n{buf.getvalue()}"
    finally:
        sys.stdin = old_stdin
    return buf.getvalue() or "[no stdout]"


_EXECUTOR = _default_executor


def set_executor(fn):
    """Install the executor backing ``execute_code`` (e.g. the real nsjail sandbox).

    ``fn(code, language="python", stdin="") -> str`` returns the run's textual output.
    """
    global _EXECUTOR
    _EXECUTOR = fn


@_register_op("execute_code", bucket="tool_io", io_pattern=True)
def _execute_code(code, language="python", stdin=""):
    """Run ``code`` through the configured executor and return its textual output."""
    return _EXECUTOR(code, language=language, stdin=stdin)


# side-effecting + non-deterministic: replay reuses the recorded output, never re-runs it.
_execute_code.pace_agent_tool.side_effect_free = False
type(_execute_code.pace_agent_tool).side_effect_free = False


_SYS = (
    "You are a careful problem solver. Prefer running code with the execute_code tool over "
    "reasoning about what code would output. When you have the answer, reply with a final "
    "line 'FINAL ANSWER: <value>' and do not call any more tools."
)


def _dispatch(name, arguments):
    """Route an LLM tool call by ``name`` to the matching PACE Agent-registered tool."""
    from pace_agent.registry import get_tool

    args = parse_arguments(arguments)
    if name == "execute_code":
        code = args.get("code", "")
        language = args.get("language", "python")
        return str(get_tool("execute_code").run("naive", code, language=language))
    return f"[unknown tool: {name}]"


def run(llm, task=None):
    """Run the tool-calling agent loop and return the final answer string."""
    query = (task or {}).get("query") or QUESTION
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": query},
    ]
    max_turns = int((task or {}).get("max_turns", 6))
    answer, _messages, events = run_tool_loop(
        llm,
        messages,
        tools=[EXECUTE_CODE_TOOL],
        dispatch=_dispatch,
        max_turns=max_turns,
        final_prompt="Give your best final answer now, ending with 'FINAL ANSWER: <value>'.",
    )
    return answer
