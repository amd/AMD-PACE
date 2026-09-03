# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import os
import shutil
import tempfile
import unittest

from pace_agent.config import BenchConfig, RunConfig, load_config
from pace_agent.errors import ConfigError


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_run_config(self):
        p = os.path.join(self.tmp, "run.json")
        with open(p, "w") as f:
            json.dump(
                {"workload": "webvoyager", "impl": "pace", "num_runs": 5, "junk": 1}, f
            )
        cfg = load_config(p)
        self.assertIsInstance(cfg, RunConfig)
        self.assertEqual(cfg.workload, "webvoyager")
        self.assertEqual(cfg.num_runs, 5)

    def _write(self, name, payload):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            json.dump(payload, f)
        return p

    def test_task_defaults_fill_in_only_what_a_task_omits(self):
        cfg = load_config(
            self._write(
                "run.json",
                {
                    "task_defaults": {"capture_snapshots": False, "max_iter": 15},
                    "tasks": [{"query": "q1"}, {"query": "q2", "max_iter": 40}],
                },
            )
        )
        self.assertEqual(
            cfg.tasks[0], {"query": "q1", "capture_snapshots": False, "max_iter": 15}
        )
        self.assertEqual(cfg.tasks[1]["max_iter"], 40)

    def test_scoring_block_is_validated_at_load(self):
        cfg = load_config(self._write("ok.json", {"scoring": {"gold_key": "expected"}}))
        self.assertEqual(cfg.scoring["gold_key"], "expected")
        with self.assertRaises(ConfigError):
            load_config(self._write("bad.json", {"scoring": "gold"}))

    def test_bench_config(self):
        p = os.path.join(self.tmp, "bench.json")
        with open(p, "w") as f:
            json.dump({"tools": {"quantize": {"size": 10}}}, f)
        cfg = load_config(p)
        self.assertIsInstance(cfg, BenchConfig)
        self.assertEqual(cfg.tools["quantize"]["size"], 10)
