# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Validation for a run config's ``characterize`` block, import-light so ``config`` can
reach it: a misspelled strategy must fail at load, not after ten live runs."""

from pace_agent.characterize.select import AUTO, STRATEGIES
from pace_agent.characterize.signature import ACTION, GRANULARITIES
from pace_agent.errors import ConfigError

DEFAULTS = {
    "strategy": AUTO,
    "granularity": ACTION,
    "min_success": 1,
    "keep_trajectories": False,
}


def resolve(block):
    """Normalized keyword arguments for ``characterize``; raises on a malformed block."""
    out = {**DEFAULTS, **(block or {})}
    unknown = sorted(set(out) - set(DEFAULTS))
    if unknown:
        raise ConfigError(
            f"unknown characterize option(s) {unknown}; expected any of {sorted(DEFAULTS)}"
        )
    for key, allowed in (("strategy", STRATEGIES), ("granularity", GRANULARITIES)):
        if out[key] not in allowed:
            raise ConfigError(
                f"unknown characterize.{key} {out[key]!r}; expected one of {list(allowed)}"
            )
    try:
        out["min_success"] = int(out["min_success"])
    except (TypeError, ValueError):
        out["min_success"] = 0
    if out["min_success"] < 1:
        raise ConfigError(
            "characterize.min_success must be an integer of at least 1, got "
            f"{(block or {}).get('min_success')!r}"
        )
    out["keep_trajectories"] = bool(out["keep_trajectories"])
    return out
