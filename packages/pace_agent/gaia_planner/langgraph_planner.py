# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

"""LangGraph engine for the GAIA planner.

``LangGraphGaiaPlanner`` is a drop-in alternative to the hand-rolled loop in
:class:`gaia_planner.planner.GaiaPlanner`, expressed as an explicit LangGraph
``StateGraph`` (the same style as ``pace_agent.examples.webvoyager``). It subclasses
``GaiaPlanner`` and reuses *all* of its behavior — the tool schemas, tool dispatch
(``_call_tool``), the FINAL-ANSWER acceptance / placeholder / empty nudges, the
context-budget trimming, and the ``PlannerRunResult`` shape — so runs recorded and
scored through PACE Agent are identical regardless of engine.

Graph shape (one ``agent`` LLM turn per super-step, mirroring the loop's turns)::

    agent ─(tool_calls)→ tools ─┐
      │                         ├─(turn<max)→ agent
      │                         └─(turn≥max)→ finalize → END
      ├─(FINAL ANSWER, verified / already nudged)──────→ END
      ├─(FINAL ANSWER, not verified)──(verify nudge)───→ agent
      ├─(turn≥max, nudged)──────────────────→ finalize → END
      └─(nudge, turn<max)───────────────────→ agent

``finalize`` is the loop's post-loop forced answer (one ``tool_choice="none"`` call).

Verification gate (langgraph-only; on by default, disable with ``GAIA_ENFORCE_VERIFY=0``):
when the model emits a FINAL ANSWER *without* having just verified by execution — i.e. its
last tool action was not a successful ``execute_code`` — it is sent back once with a nudge to
confirm the result in the sandbox. This keeps the LLM to planning and pushes deterministic
verification onto the tools, while leaving genuinely non-code-verifiable answers to pass
through after a single pass. The loop engine does not apply this gate.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from .models import GaiaTask, PlannerRunResult
from .planner import (
    GaiaPlanner,
    _assistant_message,
    _final_answer_value,
    _is_placeholder_answer,
    _result,
    _select_nudge,
    _truncate_text,
)

# Tools that gather *new* evidence rather than verify an existing result. If one of these was
# the model's last action before finalizing, it did not just verify by execution.
_EVIDENCE_TOOLS = frozenset(
    {
        "browse_web",
        "fetch_url",
        "provision",
        "analyze_video",
        "analyze_audio",
        "analyze_image",
    }
)

# Enforced only when the model finalizes WITHOUT having just verified by execution. Keep this
# general (recompute / re-parse / assert shape) rather than task-specific, and let genuinely
# non-code-verifiable answers pass through unchanged after a single pass.
VERIFY_BY_EXECUTION_NUDGE = (
    "You produced a FINAL ANSWER without verifying it by execution. Before finalizing, "
    "confirm the result with the sandbox: run an execute_code check that re-derives or "
    "validates the answer from the evidence you already gathered (recompute the value, "
    "re-parse the source, or assert its shape/type/units match what the question asks). If "
    "the check disagrees, fix your approach and answer again. Only if the answer genuinely "
    "cannot be verified with code (e.g. a purely qualitative fact already established by a "
    "tool result) should you restate the same line exactly:\nFINAL ANSWER: <answer>"
)


def _llm_is_verifying(tool_events: list[dict[str, Any]]) -> bool:
    """True if the model's most recent tool action was a *successful* ``execute_code`` call,
    i.e. it verified (or computed) via the sandbox right before finalizing.

    Scans tool history backwards: the first evidence-gathering tool (browse/fetch/analyze)
    means the answer's immediate provenance was not a code check; a successful ``execute_code``
    means it was; ``execute_code`` that errored (or no tools at all) counts as not verified.
    """
    for event in reversed(tool_events or []):
        if not isinstance(event, dict):
            continue
        name = event.get("name")
        if name in _EVIDENCE_TOOLS:
            return False
        if name == "execute_code":
            return _execute_code_succeeded(event.get("output") or "")
    return False


def _execute_code_succeeded(output: str) -> bool:
    """True only when an ``execute_code`` result actually reports success.

    The sandbox renders ``[status: success|error|timeout|...]``, but a tool call
    the planner could not dispatch comes back as a JSON envelope
    (``{"status": "error", ...}``) and an unparseable argument list as
    ``[tool error] ...``. Treating those as success would let an *unverified*
    answer through the verify gate, so anything that is not an explicit success
    counts as not verified.
    """
    text = output.lstrip()
    if text.startswith("[status:"):
        return text.startswith("[status: success]")
    if text.startswith("[tool error]"):
        return False
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "status" in payload:
            return payload["status"] == "success"
    # No explicit failure signal: assume the call ran (the historical default).
    return True


class _PlannerState(TypedDict):
    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    web_state: dict[str, int]
    turn: int
    raw_answer: str
    done: bool
    # The assistant message for a turn that requested tool calls (consumed by the
    # tools node), or None. Always cleared before the graph ends.
    pending: Any
    # Set once we have required an unverified final answer to make one sandbox verification
    # pass, so the enforcement fires at most once per task (no verify/answer loop).
    verify_nudged: bool


class LangGraphGaiaPlanner(GaiaPlanner):
    """GAIA planner whose control flow is a LangGraph ``StateGraph`` instead of a loop."""

    def run(self, task: GaiaTask) -> PlannerRunResult:
        from langgraph.graph import END, StateGraph

        from pace_agent.node_counter import _record, reset
        from pace_agent.node_counter import timings as node_timings

        system_prompt = self.config.system_prompt or self._default_prompt()
        max_turns = self.config.max_turns
        # When the model finalizes without having just verified by execution, require one
        # sandbox verification pass. Disable with GAIA_ENFORCE_VERIFY=0.
        # Closed-book runs advertise no tools and have no sandbox, so nudging the
        # model to "verify by running execute_code" would demand something it
        # cannot do. run_gaia sets GAIA_ENFORCE_VERIFY=0 for this, but the
        # standalone `python -m gaia_planner --no-tools` path must self-disable.
        enforce_verify = (
            os.getenv("GAIA_ENFORCE_VERIFY", "1") != "0" and not self.config.no_tools
        )

        def _timed(name, fn):
            """Wrap a graph node to sample its wall-clock time (same as webvoyager.py)."""

            def wrapped(state):
                t = time.perf_counter()
                try:
                    return fn(state)
                finally:
                    _record(name, time.perf_counter() - t)

            return wrapped

        with self._execution_cm(task) as execution:
            # Closed-book mode: no tools built, none advertised; the graph never reaches
            # the tools node because the model is offered no tools to call.
            tools = None if execution is None else self._build_tools(execution)

            def agent(state: _PlannerState) -> dict[str, Any]:
                messages = list(state["messages"])
                self._enforce_context_budget(messages)
                turn = state["turn"] + 1
                response = self._agent_call(messages, tools, tool_choice="auto")
                message = response.choices[0].message
                if not message.tool_calls:
                    raw_answer = message.content or ""
                    messages.append({"role": "assistant", "content": raw_answer})
                    answer_value = _final_answer_value(raw_answer)
                    if answer_value and not _is_placeholder_answer(answer_value):
                        # Force a sandbox verification pass ONLY when the model finalized
                        # without just verifying by execution (and we have not already asked
                        # for one). If it verified — or verification is disabled — accept.
                        needs_verify = (
                            enforce_verify
                            and not state.get("verify_nudged", False)
                            and not _llm_is_verifying(state["tool_events"])
                        )
                        if not needs_verify:
                            return {
                                "messages": messages,
                                "turn": turn,
                                "raw_answer": raw_answer,
                                "done": True,
                                "pending": None,
                            }
                        messages.append(
                            {"role": "user", "content": VERIFY_BY_EXECUTION_NUDGE}
                        )
                        return {
                            "messages": messages,
                            "turn": turn,
                            "pending": None,
                            "done": False,
                            "verify_nudged": True,
                        }
                    # No/blank/placeholder FINAL ANSWER — push the model to keep going.
                    nudge = _select_nudge(answer_value, self.config.no_tools)
                    messages.append({"role": "user", "content": nudge})
                    return {
                        "messages": messages,
                        "turn": turn,
                        "pending": None,
                        "done": False,
                    }

                messages.append(_assistant_message(message))
                return {
                    "messages": messages,
                    "turn": turn,
                    "pending": message,
                    "done": False,
                }

            def tools_node(state: _PlannerState) -> dict[str, Any]:
                message = state["pending"]
                messages = list(state["messages"])
                events = list(state["tool_events"])
                web_state = dict(state["web_state"])
                for tool_call in message.tool_calls:
                    output = self._call_tool(tool_call, task, execution, web_state)
                    stored = _truncate_text(output, self.config.max_tool_output_chars)
                    events.append(
                        {
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "output": stored,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": stored,
                        }
                    )
                return {
                    "messages": messages,
                    "tool_events": events,
                    "web_state": web_state,
                    "pending": None,
                }

            def finalize(state: _PlannerState) -> dict[str, Any]:
                messages = list(state["messages"])
                raw_answer = self._force_final_answer(messages, tools)
                return {"messages": messages, "raw_answer": raw_answer, "done": True}

            def route_agent(state: _PlannerState) -> str:
                if state["done"]:
                    return "end"
                if state["pending"] is not None:
                    return "tools"
                # A nudged turn (no answer, no tools). Out of turns -> force a final answer.
                return "finalize" if state["turn"] >= max_turns else "agent"

            def route_tools(state: _PlannerState) -> str:
                # The just-dispatched turn's tools are already applied; mirror the loop's
                # "run the last turn's tools, then finalize" behavior.
                return "finalize" if state["turn"] >= max_turns else "agent"

            graph = StateGraph(_PlannerState)
            graph.add_node("agent", _timed("agent", agent))
            graph.add_node("tools", _timed("tools", tools_node))
            graph.add_node("finalize", _timed("finalize", finalize))
            graph.set_entry_point("agent")
            graph.add_conditional_edges(
                "agent",
                route_agent,
                {
                    "tools": "tools",
                    "finalize": "finalize",
                    "agent": "agent",
                    "end": END,
                },
            )
            graph.add_conditional_edges(
                "tools", route_tools, {"agent": "agent", "finalize": "finalize"}
            )
            graph.add_edge("finalize", END)
            app = graph.compile()

            reset()  # per-task node timings (module-global store when no capture is active)
            out = app.invoke(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._task_prompt(task)},
                    ],
                    "tool_events": [],
                    "web_state": {"fail_streak": 0},
                    "turn": 0,
                    "raw_answer": "",
                    "done": False,
                    "pending": None,
                    "verify_nudged": False,
                },
                config={"recursion_limit": max_turns * 4 + 20},
            )
            per_node = node_timings()

        return _result(
            task,
            out["raw_answer"],
            out["messages"],
            out["tool_events"],
            per_node=per_node,
        )
