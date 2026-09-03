# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import platform
import re
import subprocess

from pace_agent import pinning


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _cpu_model():
    for line in _run(["lscpu"]).splitlines():
        if line.startswith("Model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _numa():
    out = _run(["lscpu"])
    nodes = {}
    for line in out.splitlines():
        m = re.match(r"NUMA node(\d+) CPU\(s\):\s*(.+)", line)
        if m:
            nodes[f"node{m.group(1)}"] = m.group(2).strip()
    return nodes


def _git_sha():
    return _run(["git", "rev-parse", "--short", "HEAD"]).strip() or None


def capture_env():
    mem = ""
    for line in _run(["free", "-h"]).splitlines():
        if line.lower().startswith("mem"):
            mem = line.split()[1]
    paranoid = None
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            paranoid = int(f.read().strip())
    except Exception:
        pass
    return {
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "numa_nodes": _numa(),
        "mem_total": mem,
        "governor": _run(
            [
                "bash",
                "-c",
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null",
            ]
        ).strip()
        or None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "perf_event_paranoid": paranoid,
        "git_sha": _git_sha(),
        # what this process asked for vs. the affinity it actually holds
        "pinning": pinning.record(),
    }
