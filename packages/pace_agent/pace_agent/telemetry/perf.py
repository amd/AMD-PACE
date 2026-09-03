# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import shutil
import subprocess

_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
]


def available():
    return shutil.which("perf") is not None


def perf_stat(argv, events=None, timeout=600):
    events = events or _EVENTS
    # --no-big-num: keep values as plain digits so the `-x ,` CSV parse doesn't split a
    # locale-grouped number (e.g. "1,234,567") across columns
    cmd = [
        "perf",
        "stat",
        "--no-big-num",
        "-x",
        ",",
        "-e",
        ",".join(events),
        "--",
    ] + list(argv)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return _parse(proc.stderr)


def _parse(text):
    out = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        val, event = parts[0], parts[2]
        if val in ("<not supported>", "<not counted>", ""):
            continue
        try:
            out[event] = float(val)
        except ValueError:
            continue
    return out


def counters(argv, **kw):
    raw = perf_stat(argv, **kw)
    cyc, ins = raw.get("cycles"), raw.get("instructions")
    hw = {"cycles": cyc, "instructions": ins, "ipc": (ins / cyc) if cyc else None}
    cache = {}
    if raw.get("cache-references"):
        cache["cache_miss_rate"] = raw.get("cache-misses", 0) / raw["cache-references"]
    if raw.get("L1-dcache-loads"):
        cache["l1_dcache_miss_rate"] = (
            raw.get("L1-dcache-load-misses", 0) / raw["L1-dcache-loads"]
        )
    if raw.get("LLC-loads"):
        cache["llc_miss_rate"] = raw.get("LLC-load-misses", 0) / raw["LLC-loads"]
    return {"hw": hw, "cache": cache, "raw": raw}
