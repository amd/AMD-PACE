# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import sys
import unittest

from pace_agent import telemetry
from pace_agent.telemetry import perf, uprof

_PY = sys.executable
_WORK = [
    _PY,
    "-c",
    "import numpy as np;a=np.random.rand(900,900);[(a@a).sum() for _ in range(2)]",
]


def _perf_perm():
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return int(f.read().strip()) <= 0
    except Exception:
        return False


class TestTelemetry(unittest.TestCase):
    def test_available_shape(self):
        a = telemetry.available()
        self.assertLessEqual({"perf", "uprof"}, set(a))

    def test_perf_counters(self):
        if not perf.available() or not _perf_perm():
            self.skipTest("perf CPU counters unavailable (paranoid>0 or no perf)")
        c = perf.counters(
            _WORK, timeout=60
        )  # unit test: don't inherit the 600s default
        if not (c["hw"]["cycles"] and c["hw"]["instructions"]):
            self.skipTest(
                "perf ran but collected no counters (caps/container restrictions)"
            )
        self.assertGreater(c["hw"]["ipc"], 0)

    def test_uprof_gated(self):
        if not uprof.available():
            self.skipTest("uProf not installed")
        busy = [
            _PY,
            "-c",
            "import numpy as np,time;t=time.time();a=np.random.rand(1400,1400)\n"
            "while time.time()-t<5:(a@a).sum()",
        ]
        r = uprof.collect(busy, config="cpi", duration=3, timeout=60)
        if not r.get("available"):
            self.skipTest(f"uProf collect unavailable in env: {r.get('reason')}")
        if not r["metrics"].get("IPC"):
            self.skipTest("uProf collected but produced no IPC metric in this env")
        self.assertGreater(r["metrics"].get("IPC", 0), 0)
