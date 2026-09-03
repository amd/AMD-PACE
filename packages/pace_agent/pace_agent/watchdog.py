# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""A stall watchdog for a running task.

The thing worth aborting is a task that has *stopped making progress*, not one that is merely slow:
a hard per-task deadline would kill a long GAIA question that is working through its steps while
still letting a wedged one hold a worker until the deadline. So this measures the gap *between*
`beat()` calls — a node entered or finished, a tool call returned, the model answered — and fires
only when that gap exceeds the limit, however long the task itself runs.

The check runs on `SIGALRM`, so a stall surfaces as an ordinary exception at the agent's own call
site and its `finally` blocks still close the browser. Two consequences worth knowing: it only
works in the main thread of the process running the task, and the timer is disarmed the moment it
fires so that cleanup is never interrupted twice. Recording progress is only a timestamp write,
which is what makes it safe to call from LangGraph's node threads.

What that main-thread requirement means in practice: a linear graph — every agent here — runs its
nodes on the calling thread, so a wedged node is interrupted where it hangs (tested). A graph that
fans out runs its branches on a pool thread instead, and a branch wedged there is only caught once
it returns; enforcing the limit through a hang like that would take a watchdog outside the process.
"""

import os
import signal
import threading
import time
from contextlib import contextmanager

from pace_agent.errors import ConfigError, TaskStalled

DEFAULT_STALL_S = 600.0
MAX_TICK_S = 30.0
ENV_VAR = "PACE_AGENT_MAX_TIMEOUT"

_state = None


def stall_limit_s():
    """How long a task may make no progress, in seconds; 0 disables the watchdog."""
    raw = os.environ.get(ENV_VAR)
    if raw is None or not str(raw).strip():
        return DEFAULT_STALL_S
    try:
        seconds = float(raw)
    except ValueError:
        raise ConfigError(f"{ENV_VAR} must be a number of seconds, got {raw!r}")
    if seconds < 0:
        raise ConfigError(f"{ENV_VAR} must not be negative, got {raw!r}")
    return seconds


def armable():
    return hasattr(signal, "SIGALRM") and threading.current_thread() is (
        threading.main_thread()
    )


def beat(what=None):
    """Note that the task made progress, optionally labelling it for the stall message."""
    st = _state
    if st is None:
        return
    st["last"] = time.monotonic()
    if what is not None:
        st["what"] = what


def _tick(signum, frame):
    st = _state
    if st is None:
        return
    idle = time.monotonic() - st["last"]
    if idle < st["limit"]:
        return
    signal.setitimer(
        signal.ITIMER_REAL, 0
    )  # let cleanup run without a second interruption
    raise TaskStalled(
        f"{st['label']} made no progress for {idle:.0f}s "
        f"(last progress: {st['what']}); increase or disable the limit with "
        f"{ENV_VAR}=<seconds>, 0 to disable"
    )


@contextmanager
def watch(label="task", limit=None):
    """Raise `TaskStalled` once `limit` seconds pass with no `beat()`."""
    global _state
    limit = stall_limit_s() if limit is None else limit
    if not limit or not armable():
        yield None
        return

    previous_state = _state
    previous_handler = signal.signal(signal.SIGALRM, _tick)
    _state = {
        "limit": limit,
        "last": time.monotonic(),
        "what": "task start",
        "label": label,
    }
    tick = max(0.01, min(limit / 4.0, MAX_TICK_S))
    # arming this timer replaces any enclosing one, which must be given back below
    previous_timer = signal.setitimer(signal.ITIMER_REAL, tick, tick)
    try:
        yield limit
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        _state = previous_state
        if previous_timer[0]:
            if previous_state is not None:
                # this watch ran, so the enclosing one must not read it as one long stall
                previous_state["last"] = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
