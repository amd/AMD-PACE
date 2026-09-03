# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import json
import shutil
import tempfile
import os
import unittest

from pace_agent import get_tool
from pace_agent.cli import _resolve_agent
from pace_agent.optimize import optimize
from pace_agent.profile.attribution import attribute
from pace_agent.profile.profiler import import_agent
from pace_agent.replay import Recorder
from pace_agent.rundir import RunDir


class StubLLM:
    def __init__(self, endpoint=""):
        self.endpoint = endpoint
        self._tape = None
        self._rec = False
        self._pos = 0
        self.calls = []

    def record_into(self, tape):
        self._tape, self._rec, self._pos = tape, True, 0

    def replay_from(self, tape):
        self._tape, self._rec, self._pos = list(tape), False, 0

    def complete(self, m):
        if self._tape is not None and not self._rec:
            r = self._tape[self._pos]
            self._pos += 1
            return r
        self.calls.append(0.01)
        if self._rec:
            self._tape.append("ANSWER")
        return "ANSWER"


_AGENT_SRC = """
from pace_agent import get_tool
def run(llm):
    op = get_tool("vector_search")  # a real optimized candidate exists
    args, kw = op.bench_inputs({"N": 200, "dim": 16, "top_k": 5})
    op.run("naive", *args, **kw)
    return llm.complete([{"role": "user", "content": "hi"}])
"""


class ShimAgent:
    def __init__(self):
        self.endpoint = ""

    def run(self, llm):
        json.dumps(
            {"payload": list(range(20000))}
        )  # un-instrumented serialization (not an op)
        get_tool("tokenize").run(
            "naive", "x y z"
        )  # op -> attributed exactly, shim suppressed
        return llm.complete([{"role": "user", "content": "hi"}])


class TestReviewFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_file_path_agent_profile_optimize(self):
        p = os.path.join(self.tmp, "my_agent.py")
        with open(p, "w") as f:
            f.write(_AGENT_SRC)
        agent = import_agent(p)
        traj = Recorder().record(agent, StubLLM(), os.path.abspath(p))
        d = os.path.join(self.tmp, "run")
        RunDir(d).write_result({"cmd": "profile"})
        traj.save(d)
        r = optimize(d, impl="optimized", num_runs=2)
        self.assertIn("e2e_pace", r)  # legacy alias kept one release
        self.assertEqual(r["candidate_impl"], "optimized")
        self.assertEqual(r["coverage"]["candidate_calls"], 1)
        self.assertTrue(r["parity_all_pass"])

    def test_resolve_agent_accepts_backslash_paths(self):
        spec = r"C:\agents\my_agent"
        self.assertEqual(_resolve_agent(spec), os.path.abspath(spec))

    def test_shims_attribute_uninstrumented_work(self):
        traj = Recorder().record(ShimAgent(), StubLLM(), "shimtest")
        # §7.13 layer 2 is LIVE, not dead
        self.assertGreater(traj.shim_buckets.get("serialization", 0), 0)
        a = attribute(traj)
        self.assertGreater(a["buckets"]["serialization"], 0)
