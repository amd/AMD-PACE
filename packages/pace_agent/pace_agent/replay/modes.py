# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Replay-mode registry.

A trajectory records how it should be replayed in ``meta["replay_mode"]`` (default
``"deterministic"``). optimize looks the mode up here instead of importing a workload/
example, so replay is driven by recorded data — never a forward reference from core to
``pace_agent.examples``. Core registers the deterministic op-replay mode; the browser-snapshot
mode is registered by its module (b6).

Handler contract: ``handler(trajectory, run_dir, opts) -> dict`` where ``opts`` carries
``impl``, ``num_runs`` and ``coverage``.
"""

_MODES = {}
# built-in modes registered on demand (avoids importing selenium unless actually needed)
_LAZY = {"browser-snapshot": "pace_agent.replay.browser_snapshot"}


def register_replay_mode(name, handler):
    _MODES[name] = handler


def get_replay_mode(name):
    if name not in _MODES and name in _LAZY:
        import importlib

        importlib.import_module(_LAZY[name])
    try:
        return _MODES[name]
    except KeyError:
        known = sorted(_MODES) or ["(none)"]
        raise KeyError(f"no replay mode '{name}' registered; known: {known}") from None


def replay_modes():
    return sorted(_MODES)
