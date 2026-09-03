# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Real-agent E2E of the full optimize loop.

Where test_e2e_optimize drives a synthetic op-checklist fixture, this records the *real*
example agents that invoke the PACE Agent ops as part of their actual work (RAG retrieval,
tool-result parsing) with an offline StubLLM, then runs `optimize` over each recording and
asserts the honest invariants:

  * the replay contract holds and parity passes for every covered op, and
  * every op whose optimized tier is genuinely available in this environment is covered
    (ops backed by a missing optional library are honestly *not* counted as covered).

Requires langgraph (the example agents' graph runtime); skips cleanly when it is absent,
the same way the browser tests skip without Chrome.
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

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - env without the graph runtime
    _HAS_LANGGRAPH = False

_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pace_agent",
    "examples",
)

# real example agent -> the PACE Agent optimized ops it drives through the registry hook
AGENT_OPS = {
    "large_corpus_rag": {"embed", "dedup"},  # pure tiers -> always candidates
    "customer_support": {"dedup"},  # pure tier
    "tool_agent": {"json_deserialize", "regex_extract"},  # optional-lib tiers
}


def _available(ops):
    """Subset of `ops` whose optimized tier is a genuine (non-fallback) candidate here."""
    return {name for name in ops if get_tool(name).optimized_available()}


@unittest.skipUnless(_HAS_LANGGRAPH, "langgraph not installed (example-agent runtime)")
class TestRealAgentOptimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dirs = []
        cls.runs = {}
        for name in AGENT_OPS:
            path = os.path.join(_EXAMPLES, f"{name}.py")
            out = tempfile.mkdtemp(prefix=f"pace_agent_e2e_{name}_")
            cls._dirs.append(out)
            agent = import_agent(path)
            ctx = CaptureContext()
            traj = Recorder().record(
                agent,
                StubLLM(replies=["ANSWER: done"]),
                path,
                task={"query": "hello"},
                ctx=ctx,
            )
            RunDir(out).write_result(
                {"cmd": "profile", "agent": path, "e2e": {"median": traj.e2e}}
            )
            traj.save(out)
            cls.runs[name] = {
                "rep": optimize(out, impl="optimized", num_runs=2),
                "recorded_ops": {o.op for o in traj.tools},
            }

    @classmethod
    def tearDownClass(cls):
        for d in cls._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def test_agents_actually_invoke_their_ops(self):
        # guards against an example drifting away from the ops this test claims it exercises
        for name, ops in AGENT_OPS.items():
            with self.subTest(agent=name):
                self.assertLessEqual(ops, self.runs[name]["recorded_ops"])

    def test_optimize_valid_and_parity_for_real_agents(self):
        for name in AGENT_OPS:
            rep = self.runs[name]["rep"]
            with self.subTest(agent=name):
                self.assertTrue(
                    rep.get("valid"), f"{name}: {rep.get('invalid_reason')}"
                )
                self.assertTrue(rep.get("parity_all_pass"), f"{name} parity failed")

    def test_available_ops_are_covered(self):
        # every op with a real (non-fallback) optimized tier must be re-timed as a candidate;
        # ops backed by a missing optional lib are honestly excluded, never phantom-covered
        for name, ops in AGENT_OPS.items():
            covered = set(
                self.runs[name]["rep"].get("coverage", {}).get("covered_ops", [])
            )
            with self.subTest(agent=name):
                self.assertLessEqual(_available(ops), covered)
                # a fallback op (missing lib) must NOT be reported as covered
                self.assertFalse((ops - _available(ops)) & covered)

    def test_e2e_projection_present(self):
        for name in AGENT_OPS:
            rep = self.runs[name]["rep"]
            with self.subTest(agent=name):
                self.assertIsNotNone(rep.get("e2e_speedup"))
                self.assertGreater(rep["e2e"]["reference"]["median"], 0.0)


if __name__ == "__main__":
    unittest.main()
