# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import os
import unittest

from pace_agent.bench.tools import bench_tools
from pace_agent.config import load_bench_config

_CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "bench_tools.json")


def _cfg():
    return load_bench_config(_CFG)


class TestBenchTools(unittest.TestCase):
    def test_bench_ops_naive_runs(self):
        rows = bench_tools(_cfg(), tiers=("naive",), repeat=2, warmup=1)
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["status"], ("ok", "skip"))
            if r["status"] == "ok":
                self.assertGreaterEqual(r["tiers"]["naive"]["median"], 0)

    def test_bench_config_is_cpu_compute_only(self):
        cfg = _cfg()
        for io_op in ["dom", "file_read", "state_merge"]:
            self.assertNotIn(io_op, cfg.tools)

    def test_at_least_some_ops_ok(self):
        rows = bench_tools(_cfg(), tiers=("naive",), repeat=2)
        ok = [r for r in rows if r["status"] == "ok"]
        self.assertGreaterEqual(len(ok), 10)
