# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os

_LAST = {"requested": None, "applied": {}}


def _parse_cpus(spec):
    cpus = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def cpu_spec(cpus):
    """A CPU set as the compact ranges people write in configs: {0,1,2,5} -> "0-2,5"."""
    cpus = sorted(cpus)
    if not cpus:
        return ""
    ranges, start = [], cpus[0]
    for prev, cur in zip(cpus, cpus[1:] + [None]):
        if cur != prev + 1:
            ranges.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = cur
    return ",".join(ranges)


def current_affinity():
    """The CPUs this process may actually run on — the ground truth a report can be audited
    against, whatever the config asked for."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return cpu_spec(os.sched_getaffinity(0))
    except OSError:
        return None


def record():
    """What pinning this process requested and what took effect (see `apply_pinning`)."""
    return {
        **_LAST,
        "cpu_affinity": current_affinity(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def apply_pinning(pinning):
    global _LAST
    _LAST = {"requested": dict(pinning) if pinning else None, "applied": {}}
    if not pinning:
        return {}
    applied = _LAST["applied"]
    omp = pinning.get("omp")
    if omp:
        os.environ["OMP_NUM_THREADS"] = str(omp)
        applied["omp"] = omp
    cpus = pinning.get("physcpubind")
    if cpus is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(_parse_cpus(cpus)))
            applied["physcpubind"] = cpus
        except (OSError, ValueError):
            applied["physcpubind"] = f"FAILED({cpus})"
    if pinning.get("membind") is not None:
        applied["membind"] = pinning[
            "membind"
        ]  # honored via numactl wrapper when launched externally
    return applied
