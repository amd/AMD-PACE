# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import inspect
import statistics
import time
import warnings

from pace_agent.capture import CaptureContext, recording
from pace_agent.errors import ReplayDivergence
from pace_agent.registry import get_tool, set_tool_hook
from pace_agent.watchdog import beat


def _arg_sig(v):
    """Structural signature of one argument (type + shape/len), for replay validation.
    Deliberately structural, not a content hash: it catches a divergent op/arg shape
    without false-positiving on benign float differences."""
    shape = getattr(v, "shape", None)
    if shape is not None:
        return ("array", tuple(shape), str(getattr(v, "dtype", "")))
    if isinstance(v, (str, bytes, list, tuple, dict)):
        return (type(v).__name__, len(v))
    return (type(v).__name__,)


def _sig(args, kw):
    return (
        tuple(_arg_sig(a) for a in args),
        tuple((k, _arg_sig(kw[k])) for k in sorted(kw)),
    )


def _llm_consumption(llm):
    return getattr(llm, "_pos", 0), len(getattr(llm, "_tape", None) or [])


def call_agent(agent, llm, task):
    if hasattr(agent, "run"):
        try:
            params = inspect.signature(agent.run).parameters
        except (TypeError, ValueError):
            params = {}
        if len(params) >= 2 or "task" in params:
            return agent.run(llm, task)
        return agent.run(llm)
    # B5: a plain LangGraph agent (compiled `graph`, a `graph(llm)`/`build(llm)` factory)
    # is driven by PACE Agent — the node-timing callback is injected, zero agent-side code.
    graph = _resolve_graph(agent, llm)
    if graph is not None:
        from pace_agent.langgraph import with_pace_agent

        return graph.invoke(_graph_inputs(agent, task), config=with_pace_agent())
    raise TypeError(
        f"agent {getattr(agent, '__name__', agent)!r} exposes neither run(llm, task) "
        "nor a compiled `graph` / `build(llm)` / `graph(llm)` factory"
    )


def _resolve_graph(agent, llm):
    g = getattr(agent, "graph", None)
    if g is not None:
        if hasattr(g, "invoke"):
            return g
        if callable(g):
            return g(llm)
    build = getattr(agent, "build", None)
    if callable(build):
        return build(llm)
    return None


def _graph_inputs(agent, task):
    fn = getattr(agent, "inputs", None)
    if callable(fn):
        return fn(task)
    return task if isinstance(task, dict) else {}


class Player:
    def replay(self, traj, agent, llm, tier, n, validate=True):
        """Replay the recorded run `n` times, timing `tier` (str, or callable(op)->tier).

        On the first iteration the replay cursor is validated against the recorded streams:
        each tool call must match the recorded op/bucket/arg-signature in order, with no
        extra/missing calls, and the LLM tape must be fully consumed. The first divergence
        raises a precise ReplayDivergence (never a downstream IndexError).
        """
        e2e_samples = []
        per_tool = {}
        task = traj.meta.get("task")
        n_tools = len(traj.tools)
        for it in range(n):
            check = validate and it == 0
            llm.replay_from(traj.llm_tape)
            state = {"i": 0, "times": []}

            def hook(op, _tier, args, kw):
                i = state["i"]
                if check and i >= n_tools:
                    raise ReplayDivergence(
                        f"agent made tool call #{i} {op.name}(...) beyond the "
                        f"{n_tools} recorded calls"
                    )
                rec = traj.tools[i]
                if check:
                    if op.name != rec.op or op.bucket != rec.bucket:
                        raise ReplayDivergence(
                            f"replay divergence at call #{i}: expected "
                            f"{rec.op}[{rec.bucket}], got {op.name}[{op.bucket}]"
                        )
                    if _sig(args, kw) != _sig(rec.args, rec.kw):
                        raise ReplayDivergence(
                            f"replay divergence at call #{i} ({op.name}): argument "
                            "signature changed since recording"
                        )
                state["i"] += 1
                # never re-execute a side-effecting op on replay; reuse recorded output+time
                if not op.side_effect_free:
                    state["times"].append((rec.op, rec.naive_time))
                    return rec.output
                # `tier` may be a str, or a callable(op_name)->tier for per-op dispatch
                use = tier(rec.op) if callable(tier) else tier
                t = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    op.dispatch(use, *rec.args, **rec.kw)
                dt = time.perf_counter() - t
                state["times"].append((rec.op, dt))
                beat(f"replayed tool {rec.op}")
                return rec.output

            beat(f"replay iteration {it + 1}/{n}")
            with recording(CaptureContext()):  # hook visible to worker threads
                set_tool_hook(hook)
                t0 = time.perf_counter()
                try:
                    call_agent(agent, llm, task)
                finally:
                    set_tool_hook(None)
            if check:
                if state["i"] != n_tools:
                    raise ReplayDivergence(
                        f"replay consumed {state['i']} of {n_tools} recorded tool calls "
                        "(agent made fewer calls than recorded)"
                    )
                pos, tape_len = _llm_consumption(llm)
                if pos != tape_len:
                    raise ReplayDivergence(
                        f"replay consumed {pos} of {tape_len} recorded LLM responses"
                    )
            e2e_samples.append(time.perf_counter() - t0 + traj.llm_time)
            for name, dt in state["times"]:
                per_tool.setdefault(name, []).append(dt)
        return {
            "e2e": e2e_samples,
            "per_tool": per_tool,
            "contract": {
                "tool_calls_expected": n_tools,
                "tool_calls_consumed": n_tools,
                "llm_calls_expected": len(traj.llm_tape),
                "validated": validate,
            },
        }

    def isolate(self, traj, op_name, tier="optimized", repeat=5):
        op = get_tool(op_name)
        calls = [o for o in traj.tools if o.op == op_name]
        if (
            not op.side_effect_free
        ):  # never re-run a side-effecting op; use recorded time
            rec = sum(c.naive_time for c in calls)
            return {
                "op": op_name,
                "bucket": op.bucket,
                "io_pattern": op.io_pattern,
                "n_calls": len(calls),
                "naive_time": rec,
                "candidate_time": rec,
                "optimized_time": rec,
                "recoverable": 0.0,
            }
        naive, cand = 0.0, 0.0
        for c in calls:
            naive += _median(lambda: op.dispatch("naive", *c.args, **c.kw), repeat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cand += _median(lambda: op.dispatch(tier, *c.args, **c.kw), repeat)
        return {
            "op": op_name,
            "bucket": op.bucket,
            "io_pattern": op.io_pattern,
            "n_calls": len(calls),
            "naive_time": naive,
            "candidate_time": cand,
            "optimized_time": cand,  # legacy alias
            "recoverable": naive - cand,
        }


def _median(fn, repeat):
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)
