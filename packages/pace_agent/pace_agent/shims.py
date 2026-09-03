# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import contextlib
import contextvars
import functools
import json
import pickle
import time

_current = contextvars.ContextVar("pace_agent_bucket", default=None)
_acc = {}
_installed = []


@contextlib.contextmanager
def tagged(bucket):
    tok = _current.set(bucket)
    try:
        yield
    finally:
        _current.reset(tok)


def _accumulate(fn, bucket):
    @functools.wraps(fn)
    def inner(*a, **k):
        if (
            _current.get() is not None
        ):  # explicit tag > shim (counted by op/model timing)
            return fn(*a, **k)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            # attribute to the current run's capture (per-run isolation); else module acc
            from pace_agent.capture import current_capture

            ctx = current_capture()
            if ctx is not None:
                ctx.add_shim(bucket, dt)
            else:
                _acc[bucket] = _acc.get(bucket, 0.0) + dt

    return inner


def _targets():
    """(module, attr, bucket) for every third-party lib PACE Agent uses whose time we want
    auto-attributed. Import-guarded: a missing optional lib is skipped, never fatal."""
    targets = [
        (json, "dumps", "serialization"),
        (json, "loads", "serialization"),
        (pickle, "dumps", "serialization"),
        (pickle, "loads", "serialization"),
    ]
    try:  # msgpack: the trajectory codec's serialization
        import msgpack

        targets += [
            (msgpack, "packb", "serialization"),
            (msgpack, "unpackb", "serialization"),
        ]
    except ImportError:
        pass
    return targets


def install_shims():
    """Patch shims once, process-wide (idempotent). Under concurrency one run must never
    un-patch another's, so accumulation is per-run via the active capture, not by
    install/uninstall cycling."""
    if _installed:
        return
    for mod, attr, bucket in _targets():
        orig = getattr(mod, attr, None)
        if orig is None:
            continue
        _installed.append((mod, attr, orig))
        setattr(mod, attr, _accumulate(orig, bucket))


def uninstall_shims():
    for mod, attr, orig in _installed:
        setattr(mod, attr, orig)
    _installed.clear()


def shim_totals():
    return {
        k: v for k, v in _acc.items() if k != "model"
    }  # model authoritative via llm timing
