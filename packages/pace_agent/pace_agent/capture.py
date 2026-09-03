# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Run-scoped capture context.

A single graph/agent run accumulates its node timings, tool calls, shim totals and LLM
tape into one ``CaptureContext``. Consumers (node_counter, recorder, shims, the registry
tool hook) resolve the current context via ``current_capture()``.

Threading note: LangGraph executes nodes on a worker-thread pool, and ``contextvars`` set
on the main thread do NOT propagate to those workers. Two mechanisms bridge that:

- ``recording(ctx)`` (recorder path) publishes a *process-global* context that worker
  threads can find. Concurrent recording is process-isolated (the parallel runner uses
  processes), so a process-global is safe here.
- ``profile_invoke`` binds the context to the ``PaceAgentNodeCounter`` instance, which re-sets
  the thread-local contextvar inside each node's worker thread (see node_counter). That
  keeps concurrent in-process ``profile_invoke`` calls isolated (each has its own counter).

``current_capture()`` prefers the thread-local contextvar, then the process-global.
"""

import contextlib
import contextvars
from dataclasses import dataclass, field


@dataclass
class CaptureContext:
    run_id: str = ""
    node_samples: dict = field(default_factory=dict)  # node name -> [seconds, ...]
    node_order: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)  # replay.trajectory.ToolCall
    shim_totals: dict = field(default_factory=dict)  # bucket -> seconds
    llm_tape: list = field(default_factory=list)
    tool_hook: object = None  # active registry hook for this run (see registry)

    def record_node(self, name, dt):
        if name not in self.node_samples:
            self.node_samples[name] = []
            self.node_order.append(name)
        self.node_samples[name].append(dt)

    def add_shim(self, bucket, dt):
        self.shim_totals[bucket] = self.shim_totals.get(bucket, 0.0) + dt

    def node_count(self):
        return sum(len(v) for v in self.node_samples.values())


_active = contextvars.ContextVar("pace_agent_capture", default=None)
_recording = None  # process-global recording context (worker-thread visible)


def active_capture():
    """The thread-local capture context, or None."""
    return _active.get()


def current_capture():
    """The capture in effect: thread-local contextvar first, then the process-global
    recording context (so LangGraph worker threads still find it)."""
    return _active.get() or _recording


@contextlib.contextmanager
def capturing(ctx=None):
    """Activate ``ctx`` on the thread-local contextvar for the duration of the block."""
    ctx = ctx if ctx is not None else CaptureContext()
    token = _active.set(ctx)
    try:
        yield ctx
    finally:
        _active.reset(token)


@contextlib.contextmanager
def recording(ctx):
    """Recorder scope: publish ``ctx`` on both the contextvar (main thread) and the
    process-global (worker threads)."""
    global _recording
    prev, token = _recording, _active.set(ctx)
    _recording = ctx
    try:
        yield ctx
    finally:
        _recording = prev
        _active.reset(token)


def enter_thread(ctx):
    """Bind ``ctx`` to the current (worker) thread's contextvar; returns a token for
    ``exit_thread`` (or None if already active). Used by PaceAgentNodeCounter."""
    if ctx is None or _active.get() is ctx:
        return None
    return _active.set(ctx)


def exit_thread(token):
    if token is not None:
        try:
            _active.reset(token)
        except ValueError:
            pass  # reset attempted from a different thread/context — leave it
