# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import time

from pace_agent import shims
from pace_agent.capture import CaptureContext, recording
from pace_agent.registry import set_tool_hook
from pace_agent.replay.player import call_agent
from pace_agent.replay.trajectory import ToolCall, Trajectory
from pace_agent.watchdog import beat


class Recorder:
    def record(self, agent, llm, agent_path=None, task=None, ctx=None):
        ctx = ctx if ctx is not None else CaptureContext()
        tools = ctx.tool_calls

        def hook(op, tier, args, kw):
            t = time.perf_counter()
            with shims.tagged(op.bucket):  # suppress shim double-count inside ops
                out = op.dispatch("naive", *args, **kw)
            dt = time.perf_counter() - t
            tools.append(ToolCall(op.name, op.bucket, op.io_pattern, args, kw, out, dt))
            beat(f"tool {op.name} returned")
            return out

        tape = ctx.llm_tape
        llm.record_into(tape)
        shims.install_shims()  # idempotent, process-wide (not un-patched per run)
        t0 = time.perf_counter()
        with recording(ctx):  # publish ctx to worker threads (LangGraph executor)
            set_tool_hook(hook)  # bound onto ctx
            try:
                result = call_agent(agent, llm, task)
            finally:
                set_tool_hook(None)
        e2e = time.perf_counter() - t0
        shim_buckets = {k: v for k, v in ctx.shim_totals.items() if k != "model"}
        metrics = dict(llm.metrics()) if hasattr(llm, "metrics") else {}
        metrics.update(nodes_traversed=ctx.node_count(), tool_calls=len(tools))
        return Trajectory(
            tools=tools,
            llm_tape=tape,
            llm_time=sum(llm.calls),
            e2e=e2e,
            shim_buckets=shim_buckets,
            run_metrics=metrics,
            meta={
                "agent": agent_path,
                "task": task,
                "result": result,
                "replay_mode": "deterministic",
                "identity": _identity(agent_path),
            },
        )


def _identity(agent_path):
    """Provenance for the strict replay contract: versions + agent source digest, so replay
    can detect that the trajectory no longer matches the agent/environment."""
    import hashlib
    import os
    import platform

    from pace_agent import __version__

    info = {"pace_agent": __version__, "python": platform.python_version()}
    if agent_path and os.path.isfile(agent_path):
        with open(agent_path, "rb") as f:
            info["agent_source_digest"] = hashlib.sha256(f.read()).hexdigest()[:16]
    return info
