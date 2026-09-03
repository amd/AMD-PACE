# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Deterministic OpenAI-style tool-calling loop.

Many real agents (e.g. the AgenticAI ``gaia-planner`` and the ``nsjail-agent-tool``
executor loop) are not LangGraph graphs but a hand-rolled
``client.chat.completions.create(tools=...)`` loop: the model either answers or emits
``tool_calls``, each tool result is appended, and the loop repeats. ``run_tool_loop`` is
the PACE Agent-native form of exactly that loop, driven through a PACE Agent ``LLMClient`` so the
whole run records and replays deterministically:

- every model turn goes through ``llm.chat(...)`` — captured on the trajectory tape;
- every tool call is dispatched to a PACE Agent op — captured (and, for side-effecting tools
  like a sandbox, marked ``side_effect_free=False`` so replay reuses the recorded output
  instead of re-executing it).

An agent module only has to expose the tool schemas and a ``dispatch(name, arguments)``
that routes to its PACE Agent ops; ``pace-agent profile`` / ``optimize`` / ``report`` then work on it
with no framework-specific code.
"""

import json


def assistant_message(msg):
    """Rebuild an OpenAI assistant message (content + tool_calls) for the running history.

    Accepts either:

    - the normalized PACE Agent ``llm.chat`` dict
      ``{"content", "tool_calls": [{"id", "name", "arguments"}, ...]}``; or
    - an OpenAI SDK ``ChatCompletionMessage`` (``.content``, ``.tool_calls`` with
      ``.function.name`` / ``.function.arguments``).

    Shared by ``run_tool_loop``, gaia_planner, and nsjail_agent so the three tool loops
    do not each maintain their own copy of this conversion.
    """
    out = {"role": "assistant"}
    if isinstance(msg, dict):
        if msg.get("content"):
            out["content"] = msg["content"]
        calls = msg.get("tool_calls") or []
        if calls:
            out["tool_calls"] = [_normalize_tool_call(c) for c in calls]
        return out

    if getattr(msg, "content", None):
        out["content"] = msg.content
    calls = getattr(msg, "tool_calls", None) or []
    if calls:
        out["tool_calls"] = [_normalize_tool_call(c) for c in calls]
    return out


def _normalize_tool_call(call):
    """Map one tool call (dict or SDK object) to the OpenAI wire shape."""
    if isinstance(call, dict):
        if "function" in call:
            return {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
        return {
            "id": call.get("id"),
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        }
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }


def run_tool_loop(
    llm,
    messages,
    tools,
    dispatch,
    *,
    max_turns=12,
    tool_choice="auto",
    final_prompt=None,
    on_event=None,
):
    """Run a bounded tool-calling loop and return ``(answer_text, messages, events)``.

    Parameters
    ----------
    llm : PACE Agent ``LLMClient`` (or compatible) exposing
        ``chat(messages, tools, tool_choice) -> {"content", "tool_calls"}``.
    messages : initial chat messages (system + user); copied, not mutated in place.
    tools : list of OpenAI tool schemas advertised to the model.
    dispatch : ``callable(name: str, arguments: str) -> str`` running one tool call and
        returning its text output. Route this to PACE Agent ops so the CPU/tool work is captured.
    max_turns : hard cap on model turns before forcing a final no-tool answer.
    final_prompt : optional user nudge appended before the forced final answer.
    on_event : optional ``callable(event: dict)`` called for each tool result.
    """
    messages = list(messages)
    events = []
    for _turn in range(max_turns):
        msg = llm.chat(messages, tools=tools, tool_choice=tool_choice)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or ""), messages, events
        messages.append(assistant_message(msg))
        for call in calls:
            output = dispatch(call["name"], call.get("arguments") or "{}")
            event = {
                "id": call.get("id"),
                "name": call["name"],
                "arguments": call.get("arguments"),
                "output": output,
            }
            events.append(event)
            if on_event is not None:
                on_event(event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": output,
                }
            )
    # ran out of turns: force one final answer with tools disabled
    if final_prompt:
        messages.append({"role": "user", "content": final_prompt})
    msg = llm.chat(messages, tools=tools, tool_choice="none")
    return (msg.get("content") or ""), messages, events


def parse_arguments(arguments):
    """Best-effort parse of a tool-call ``arguments`` JSON string into a dict."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
