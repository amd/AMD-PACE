# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Offline E2E of the full optimize loop against the optimized tiers.

Records the deterministic `agent_ops` fixture with an offline StubLLM, then runs
`optimize` over the recorded tape (no browser, no live LLM). Asserts the invariants the
optimized-tier work set out to guarantee:

  * the loop is valid and parity holds for every covered op, and
  * the pure (no-optional-lib) optimized tiers are always covered.

Timing/speedup thresholds are gated behind PACE_AGENT_PERF_ASSERT so the structural invariants
stay deterministic in CI.
"""

import os
import shutil
import tempfile
import unittest

from pace_agent.capture import CaptureContext
from pace_agent.optimize import optimize
from pace_agent.profile.profiler import import_agent
from pace_agent.registry import get_tool
from pace_agent.replay import Recorder
from pace_agent.rundir import RunDir
from tests.stubs import StubLLM

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e", "agent_ops.py")

# pure tiers (no optional accelerator lib) -> always usable, so they must always show up as
# covered candidates regardless of the environment
PURE_OPS = {"embed", "rerank", "dedup", "ast_parse"}
# optional-library tiers -> covered only when their accelerator library imports
LIB_OPS = {"json_serialize", "json_deserialize", "html_parse", "regex_extract"}


class TestE2EOptimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp(prefix="pace_agent_e2e_test_")
        agent = import_agent(AGENT)
        ctx = CaptureContext()
        traj = Recorder().record(
            agent,
            StubLLM(replies=["ANSWER: done"]),
            AGENT,
            task={"query": "hello"},
            ctx=ctx,
        )
        RunDir(cls.out).write_result(
            {"cmd": "profile", "agent": AGENT, "e2e": {"median": traj.e2e}}
        )
        traj.save(cls.out)
        cls.recorded_ops = {o.op for o in traj.tools}
        cls.rep = optimize(cls.out, impl="optimized", num_runs=3)
        cls.covered = set(cls.rep.get("coverage", {}).get("covered_ops", []))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_optimize_valid_and_parity(self):
        self.assertTrue(self.rep.get("valid"), "optimize result not valid")
        self.assertTrue(
            self.rep.get("parity_all_pass"), "parity failed for a covered op"
        )

    def test_fixture_exercises_all_optimized_ops(self):
        # guards against the fixture silently drifting away from the optimized set
        self.assertTrue(PURE_OPS <= self.recorded_ops)
        self.assertTrue(LIB_OPS <= self.recorded_ops)

    def test_pure_tiers_are_covered(self):
        missing = PURE_OPS - self.covered
        self.assertFalse(
            missing, f"pure optimized tiers not covered: {sorted(missing)}"
        )

    def test_lib_tiers_covered_when_accelerator_available(self):
        for name in LIB_OPS:
            available = get_tool(name).optimized_available()
            with self.subTest(op=name, available=available):
                if available:
                    self.assertIn(name, self.covered)

    def test_e2e_projection_present(self):
        self.assertIsNotNone(self.rep.get("e2e_speedup"))
        self.assertGreater(self.rep["e2e"]["reference"]["median"], 0.0)

    @unittest.skipUnless(
        os.environ.get("PACE_AGENT_PERF_ASSERT"),
        "timing-sensitive; set PACE_AGENT_PERF_ASSERT=1 to enable",
    )
    def test_e2e_speedup_is_a_win(self):
        self.assertGreater(self.rep.get("e2e_speedup"), 1.0)


if __name__ == "__main__":
    unittest.main()
