# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from pace_agent.telemetry import perf, uprof


def available():
    return {
        "perf": perf.available(),
        "uprof": uprof.available(),
    }


def collect(mode, argv, config="cpi"):
    if mode in ("off", None):
        return {"source": "off"}
    if mode == "uprof":
        return {"source": "uprof", **uprof.collect(argv, config=config)}
    if not perf.available():
        # don't shell out to a missing `perf` (FileNotFoundError); report unavailable
        return {"source": mode, "available": False, "reason": "perf not installed"}
    result = {"source": mode}
    result.update(perf.counters(argv))
    return result
