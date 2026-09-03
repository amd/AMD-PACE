# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Benchmark-agnostic helpers shared by PACE Agent example runners.

Keeps gateway preflight, tool-failure detection, and console token summaries out of
per-suite scripts (GAIA, future datasets) so they stay one place to harden.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def preflight_chat_gateway(
    client: object,
    model: str,
    *,
    skip_env: str = "PACE_AGENT_SKIP_PREFLIGHT",
) -> bool:
    """Fail fast if the gateway/model route is unreachable.

    Issues one tiny chat completion before a task loop. Returns ``False`` when the
    check is skipped via ``skip_env`` (or legacy ``GAIA_SKIP_PREFLIGHT``). Raises
    ``RuntimeError`` when the route is down/misconfigured.
    """
    if os.getenv(skip_env) == "1" or os.getenv("GAIA_SKIP_PREFLIGHT") == "1":
        return False
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            tool_choice="none",
        )
        _ = resp.choices[0].message
    except Exception as exc:  # noqa: BLE001 - convert to a clear, fail-fast abort
        raise RuntimeError(
            f"Gateway preflight failed for model {model!r}: {type(exc).__name__}: {exc}\n"
            "The LLM gateway/model route looks unavailable — aborting before running tasks. "
            "Verify --model (is the id currently served?), --base-url, and --api-key. "
            f"Set {skip_env}=1 (or GAIA_SKIP_PREFLIGHT=1) to bypass this check."
        ) from exc
    return True


def count_tool_failures(tool_events: object) -> tuple[int, dict[str, int]]:
    """Count tool calls that returned an error, broken down by tool name.

    Adapters signal failure differently: nsjail-style tools prefix output with
    ``[status: error]``, while JSON-returning tools embed ``"status": "error"``.
    Both are detected from the recorded ``output`` string.
    """
    total = 0
    by_tool: dict[str, int] = {}
    for ev in tool_events or []:
        if not isinstance(ev, dict):
            continue
        out = ev.get("output")
        if not isinstance(out, str):
            continue
        blob = out.lstrip()
        if blob.startswith("[status: error]") or '"status": "error"' in out:
            name = ev.get("name") or ev.get("tool") or "unknown"
            total += 1
            by_tool[name] = by_tool.get(name, 0) + 1
    return total, by_tool


def component_token_lines(by_component: object, fmt: Callable[[Any], str]) -> list[str]:
    """Render a planner-vs-tools token breakdown as console lines."""
    if not isinstance(by_component, dict) or not by_component:
        return []
    width = max(len(name) for name in by_component)
    lines = ["  tokens by component:"]
    for name, m in by_component.items():
        backend = m.get("backend")
        suffix = f"  ({backend})" if backend and backend != "gateway" else ""
        lines.append(
            f"    {name.ljust(width)} : {str(m.get('llm_calls') or 0).rjust(3)} calls  "
            f"ctx {fmt(m.get('context_tokens')).rjust(9)}  "
            f"out {fmt(m.get('output_tokens')).rjust(7)}{suffix}"
        )
    return lines


def print_task_token_stats(rec: dict) -> None:
    """Print one task's token statistics to the console (best-effort)."""
    try:
        m = rec.get("metrics") or {}
        q = (rec.get("query") or "").replace("\n", " ")
        if len(q) > 100:
            q = q[:97] + "..."
        out = m.get("output_tokens_total")
        rea = m.get("reasoning_tokens_total")

        def _n(v):
            return "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))

        lines = [
            "",
            "─" * 72,
            f"task done: {q}",
            f"  answer: {rec.get('answer')!r}  (completed={rec.get('completed')})",
            "  token statistics:",
            f"    llm calls              : {_n(m.get('llm_calls'))} ({m.get('api') or '?'} API)",
            f"    context tokens (total) : {_n(m.get('context_tokens_total'))}"
            f"  (max/turn {_n(m.get('context_tokens_max'))})",
            f"    cached tokens          : {_n(m.get('cached_tokens_total'))}",
            f"    output tokens          : {_n(out)}",
            f"    reasoning tokens       : {_n(rea)}",
            f"    work tokens (out-rea)  : {_n(m.get('work_tokens_total'))}",
            "  reasoning tokens details:",
        ]
        if rea is None:
            lines.append("    not reported by this backend (no /responses usage).")
        elif out:
            pct = 100.0 * rea / out
            lines.append(
                f"    {_n(rea)} of {_n(out)} output tokens spent thinking ({pct:.1f}%)"
            )
            if rea == 0:
                lines.append(
                    "    -> model emitted no separate 'thinking' tokens for this task."
                )
        else:
            lines.append(f"    {_n(rea)} (no output tokens recorded)")
        lines.extend(component_token_lines(m.get("by_component"), _n))
        lines.append("─" * 72)
        print("\n".join(lines), flush=True)
    except Exception:  # noqa: BLE001 - console logging must never abort a run
        pass


def write_profile_like_result(
    pace_agent_out: Path,
    *,
    cmd: str,
    agent: str,
    engine: str | None,
    model: str | None,
    status: object,
    per_query: list[dict],
    aggregate_extra: dict | None = None,
) -> None:
    """Write ``result.json`` + ``env.json`` in the profiler's multi-query shape."""
    from pace_agent.env import capture_env
    from pace_agent.rundir import RunDir

    def _sum(key: str) -> float:
        return sum(q.get(key) or 0 for q in per_query)

    aggregate = {
        "n_queries": len(per_query),
        "degraded_queries": sum(1 for q in per_query if q.get("degraded")),
        "total_tool_failures": _sum("tool_failures"),
        "e2e_median_total_s": _sum("e2e_median"),
        "total_llm_calls": _sum("llm_calls"),
        "total_nodes_traversed": _sum("nodes_traversed"),
        "total_context_tokens": _sum("context_tokens_total"),
        "total_cached_tokens": _sum("cached_tokens_total"),
        "total_output_tokens": _sum("output_tokens_total"),
        "total_reasoning_tokens": _sum("reasoning_tokens_total"),
        "total_non_model_cpu_s": 0.0,
    }
    if aggregate_extra:
        aggregate.update(aggregate_extra)

    rd = RunDir(str(pace_agent_out))
    rd.write_result(
        {
            "cmd": cmd,
            "agent": agent,
            "engine": engine,
            "model": model,
            "status": status,
            "per_query": per_query,
            "aggregate": aggregate,
        }
    )
    rd.write_env(capture_env())
