# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""One-line LangGraph profiling.

``profile_invoke(graph, inputs)`` profiles a single run of ANY compiled LangGraph — node
timing + shim attribution (+ an optional trajectory) — with zero PACE Agent-specific code in the
agent. ``instrument(graph)`` returns a drop-in proxy so existing ``graph.invoke(...)`` call
sites are profiled unchanged. ``with_pace_agent(config)`` enriches a LangGraph config with the
node-timing callback (merging, not replacing, existing callbacks/tags).

Boundary (honest by design): this always gives node timing and attribution for any graph;
recording a *replayable* trajectory additionally needs the graph's tools to be registered
``PaceAgentTool``s (or decorated with ``@pace_agent.op``) and its LLM to be a PACE Agent client so the
tape is captured. Unattributed work stays visible, never guessed. LangGraph is pinned
(``langgraph==1.2.9``) so the proxied API surface is stable.
"""

import time

from pace_agent import node_counter
from pace_agent.capture import CaptureContext, capturing
from pace_agent.node_counter import PaceAgentNodeCounter
from pace_agent.registry import set_tool_hook
from pace_agent.replay.trajectory import ToolCall, Trajectory


def with_pace_agent(config=None, *, run_name=None, counter=None):
    """Return a LangGraph config dict with PACE Agent's node-timing callback added (merged)."""
    config = dict(config or {})
    callbacks = list(config.get("callbacks") or [])
    callbacks.append(counter if counter is not None else PaceAgentNodeCounter())
    config["callbacks"] = callbacks
    if run_name is not None:
        tags = list(config.get("tags") or [])
        tags.append(f"pace_agent:{run_name}")
        config["tags"] = tags
    return config


def _run_summary(ctx, result, e2e):
    return {
        "result": result,
        "e2e": e2e,
        "node_timings": node_counter.timings_of(ctx.node_samples, ctx.node_order),
        "nodes_traversed": ctx.node_count(),
        "tool_calls": list(ctx.tool_calls),
        "shim_totals": {k: v for k, v in ctx.shim_totals.items() if k != "model"},
        "capture": ctx,
    }


def profile_invoke(
    graph,
    inputs,
    *,
    config=None,
    run_name=None,
    record=None,
    record_tier="naive",
    agent_path=None,
    task=None,
    return_run=False,
):
    """Profile one ``graph.invoke(inputs)``.

    Opens a run-scoped CaptureContext, injects the node-timing callback (merged into
    ``config``), and returns the graph result (or a run handle if ``return_run``). When
    ``record`` (a run dir) is given, also records tool I/O and writes a trajectory.
    """
    from pace_agent import shims

    ctx = CaptureContext(run_id=run_name or "")
    counter = PaceAgentNodeCounter(ctx)  # binds ctx to node worker threads
    cfg = with_pace_agent(config, run_name=run_name, counter=counter)
    shims.install_shims()
    with capturing(ctx):
        if record is not None:

            def hook(tool, tier, args, kw):
                t = time.perf_counter()
                with shims.tagged(tool.bucket):
                    out = tool.dispatch(record_tier or tier, *args, **kw)
                dt = time.perf_counter() - t
                ctx.tool_calls.append(
                    ToolCall(tool.name, tool.bucket, tool.io_pattern, args, kw, out, dt)
                )
                return out

            set_tool_hook(hook)
        t0 = time.perf_counter()
        try:
            result = graph.invoke(inputs, config=cfg)
        finally:
            if record is not None:
                set_tool_hook(None)
        e2e = time.perf_counter() - t0

    run = _run_summary(ctx, result, e2e)
    if record is not None:
        traj = Trajectory(
            tools=list(ctx.tool_calls),
            llm_tape=list(ctx.llm_tape),
            shim_buckets=run["shim_totals"],
            e2e=e2e,
            run_metrics={
                "nodes_traversed": ctx.node_count(),
                "tool_calls": len(ctx.tool_calls),
            },
            meta={
                "agent": agent_path,
                "task": task,
                "result": result,
                "replay_mode": "deterministic",
            },
        )
        traj.save(record)
        run["trajectory"] = traj
    return run if return_run else result


class _Instrumented:
    """Drop-in graph proxy: profiles invoke/stream/ainvoke/astream/batch/abatch and records
    each run's node timing/attribution into ``pace_agent_runs``; everything else delegates.
    """

    def __init__(self, graph):
        self._graph = graph
        self.pace_agent_runs = []

    def _prep(self, config):
        ctx = CaptureContext()
        return ctx, with_pace_agent(config, counter=PaceAgentNodeCounter(ctx))

    def invoke(self, inputs, config=None, **kw):
        run = profile_invoke(self._graph, inputs, config=config, return_run=True)
        self._log(run)
        return run["result"]

    def stream(self, inputs, config=None, **kw):
        ctx, cfg = self._prep(config)
        t0 = time.perf_counter()
        with capturing(ctx):
            for chunk in self._graph.stream(inputs, config=cfg, **kw):
                yield chunk
        self._log(_run_summary(ctx, None, time.perf_counter() - t0))

    async def ainvoke(self, inputs, config=None, **kw):
        ctx, cfg = self._prep(config)
        t0 = time.perf_counter()
        with capturing(ctx):
            result = await self._graph.ainvoke(inputs, config=cfg, **kw)
        self._log(_run_summary(ctx, result, time.perf_counter() - t0))
        return result

    async def astream(self, inputs, config=None, **kw):
        ctx, cfg = self._prep(config)
        t0 = time.perf_counter()
        with capturing(ctx):
            async for chunk in self._graph.astream(inputs, config=cfg, **kw):
                yield chunk
        self._log(_run_summary(ctx, None, time.perf_counter() - t0))

    def batch(self, inputs, config=None, **kw):
        ctx, cfg = self._prep(config)
        t0 = time.perf_counter()
        with capturing(ctx):
            result = self._graph.batch(inputs, config=cfg, **kw)
        self._log(_run_summary(ctx, result, time.perf_counter() - t0))
        return result

    async def abatch(self, inputs, config=None, **kw):
        ctx, cfg = self._prep(config)
        t0 = time.perf_counter()
        with capturing(ctx):
            result = await self._graph.abatch(inputs, config=cfg, **kw)
        self._log(_run_summary(ctx, result, time.perf_counter() - t0))
        return result

    def _log(self, run):
        self.pace_agent_runs.append({k: v for k, v in run.items() if k != "result"})

    def __getattr__(self, name):
        return getattr(self._graph, name)


def instrument(graph):
    """Wrap a compiled LangGraph so its entry points are transparently profiled."""
    return _Instrumented(graph)
