# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import time

from pace_agent.stats import summarize

try:  # langgraph ships langchain-core; subclass so the callback manager doesn't skip us
    from langchain_core.callbacks.base import BaseCallbackHandler as _CallbackBase
except Exception:  # pragma: no cover - only when langchain-core is absent
    _CallbackBase = object

_samples = {}  # name -> [seconds, ...]  (process-global fallback when not capturing)
_order = []


def _target():
    """Node timings live on the current CaptureContext when capturing (per-run isolation),
    else on the module globals."""
    from pace_agent.capture import current_capture

    ctx = current_capture()
    if ctx is not None:
        return ctx.node_samples, ctx.node_order
    return _samples, _order


def reset():
    _samples.clear()
    _order.clear()


def _record(name, dt):
    from pace_agent.watchdog import beat

    beat(f"node {name} finished")
    samples, order = _target()
    if name not in samples:
        samples[name] = []
        order.append(name)
    samples[name].append(dt)


def count():
    samples, _ = _target()
    return sum(len(v) for v in samples.values())


def _fmt(samples, order):
    out = {}
    for n in order:
        s = summarize([x * 1e3 for x in samples[n]])  # ms
        s["count"] = len(samples[n])
        s["total_ms"] = sum(samples[n]) * 1e3
        out[n] = s
    return out


def timings():
    samples, order = _target()
    return _fmt(samples, order)


def timings_of(samples, order):
    """Format node timings from an explicit (samples, order) pair (e.g. a CaptureContext),
    for callers that read after the capture has been deactivated."""
    return _fmt(samples, order)


class PaceAgentNodeCounter(_CallbackBase):
    """LangGraph/LangChain callback: samples each node's execution time (per run).

    When constructed with a ``capture`` context, it binds that context to the worker thread
    running each node (LangGraph's executor doesn't inherit contextvars), so tool calls and
    shims inside the node attribute to the right run — and concurrent runs stay isolated.
    """

    # explicit so the callback manager never skips chain events (a plain __getattr__ would
    # make `ignore_chain` truthy and silently drop every node event)
    ignore_chain = False
    ignore_llm = False
    ignore_agent = False
    ignore_retriever = False
    ignore_chat_model = False
    ignore_custom_event = False
    ignore_retry = False
    raise_error = False
    run_inline = True

    def __init__(self, capture=None):
        try:
            super().__init__()
        except Exception:  # pragma: no cover - object base takes no args
            pass
        self._starts = {}
        self._capture = capture

    def on_chain_start(self, serialized, inputs, *, run_id=None, **kw):
        name = (kw.get("metadata") or {}).get("langgraph_node")
        if name:
            from pace_agent import capture
            from pace_agent.watchdog import beat

            beat(f"node {name} started")
            token = capture.enter_thread(self._capture)
            self._starts[run_id] = (name, time.perf_counter(), token)

    def on_chain_end(self, outputs, *, run_id=None, **kw):
        rec = self._starts.pop(run_id, None)
        if rec:
            _record(rec[0], time.perf_counter() - rec[1])
            if rec[2] is not None:
                from pace_agent import capture

                capture.exit_thread(rec[2])
