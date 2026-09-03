# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Unit tests for every PACE Agent optimized tier.

Grouped by how the accelerator is provided, not by when it was added:
  pure (no optional dep):    vector_search, bm25, quantize, dedup, embed,
                             rerank, ast_parse
  library-backed (optional): json_serialize / json_deserialize (orjson),
                             html_parse (selectolax), regex_extract (re2)

Plus the honesty hook (`optimized_available` -> is_fallback) that makes a
missing accelerator library a *labeled* naive fallback rather than a phantom tier.
"""

import os
import unittest
import warnings

import numpy as np

from pace_agent import get_tool
from pace_agent.stats import time_call
from pace_agent.tools import serialization as _ser

# Optimized tiers with no optional dependency: always a genuine candidate.
PURE_OPS = [
    "vector_search",
    "bm25",
    "quantize",
    "dedup",
    "embed",
    "rerank",
    "ast_parse",
]

# Library-backed tiers -> module-level availability flag in pace_agent.tools.serialization.
LIB_OPS = {
    "json_serialize": "_HAS_ORJSON",
    "json_deserialize": "_HAS_ORJSON",
    "html_parse": "_HAS_SELECTOLAX",
    "regex_extract": "_HAS_RE2",
}

# Small inputs keep parity fast; parity is size-independent so the large bench
# defaults (e.g. vector_search N=10000) would only slow the test down.
SPECS = {
    "vector_search": {"N": 256, "dim": 64, "top_k": 10},
    "bm25": {"N_docs": 200, "doclen": 20, "qlen": 5, "top_k": 10},
    "quantize": {"size": 1000},
    "dedup": {"n": 600},
    "embed": {"n": 64, "dim": 64, "toklen": 16},
    "rerank": {"n": 256, "dim": 64, "top_k": 10},
    "ast_parse": {"n": 50},
    "json_serialize": {"n": 200},
    "json_deserialize": {"n": 200},
    "html_parse": {"n": 120},
    "regex_extract": {"n": 600},
}


def _quiet():
    warnings.simplefilter("ignore")


class TestPureOptimizedParity(unittest.TestCase):
    """Pure optimized tiers must be output-equivalent to naive and never a fallback."""

    def test_parity(self):
        for name in PURE_OPS:
            op = get_tool(name)
            args, kw = op.bench_inputs(SPECS[name])
            with warnings.catch_warnings():
                _quiet()
                self.assertTrue(op.parity(args, kw), f"{name} parity failed")

    def test_pure_tiers_are_real_candidates(self):
        # no optional dep -> always a genuine candidate, never a labeled fallback
        for name in PURE_OPS:
            info = get_tool(name).implementations()["optimized"]
            self.assertFalse(info.is_fallback, f"{name} should be a real candidate")
            self.assertEqual(info.kind, "candidate", name)


class TestDedup(unittest.TestCase):
    def test_order_preserving_and_equivalent(self):
        op = get_tool("dedup")
        items = ["a", "b", "a", "c", "b", "d", "a"]
        self.assertEqual(op.dispatch("optimized", items), ["a", "b", "c", "d"])
        self.assertEqual(op.dispatch("optimized", items), op.dispatch("naive", items))

    def test_unhashable_falls_back(self):
        op = get_tool("dedup")
        items = [["x"], ["x"], ["y"]]  # lists are unhashable
        self.assertEqual(op.dispatch("optimized", items), [["x"], ["y"]])


class TestEmbed(unittest.TestCase):
    def test_allclose_to_naive(self):
        op = get_tool("embed")
        args, kw = op.bench_inputs(SPECS["embed"])
        a = op.dispatch("naive", *args, **kw)
        b = op.dispatch("optimized", *args, **kw)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b, atol=1e-6, rtol=1e-6))


class TestRerank(unittest.TestCase):
    def test_topk_matches_naive(self):
        op = get_tool("rerank")
        args, kw = op.bench_inputs(SPECS["rerank"])
        in_, sn = op.dispatch("naive", *args, **kw)
        io, so = op.dispatch("optimized", *args, **kw)
        self.assertEqual(list(in_), list(io))  # distinct random scores => same order
        self.assertTrue(np.allclose(np.sort(sn), np.sort(so), atol=1e-4))

    def test_parity_uses_custom_comparator(self):
        op = get_tool("rerank")
        args, kw = op.bench_inputs({"n": 32, "dim": 8, "top_k": 5})
        calls = []

        def comparator(reference, candidate):
            calls.append((reference, candidate))
            return False

        self.assertFalse(op.parity(args, kw, comparator=comparator))
        self.assertEqual(len(calls), 1)


class TestAstParse(unittest.TestCase):
    def test_node_count_equal(self):
        op = get_tool("ast_parse")
        args, kw = op.bench_inputs(SPECS["ast_parse"])
        self.assertEqual(
            op.dispatch("naive", *args, **kw), op.dispatch("optimized", *args, **kw)
        )


class TestLibOptimized(unittest.TestCase):
    """Library-backed tiers: parity when the lib is present; honest fallback when not."""

    def test_parity_holds_either_way(self):
        # parity must pass whether or not the accelerator is installed (fallback = naive)
        for name in LIB_OPS:
            op = get_tool(name)
            args, kw = op.bench_inputs(SPECS[name])
            with warnings.catch_warnings():
                _quiet()
                self.assertTrue(op.parity(args, kw), f"{name} parity failed")

    def test_availability_reported_honestly(self):
        # implementations() must mirror the actual lib availability
        for name, flag in LIB_OPS.items():
            info = get_tool(name).implementations()["optimized"]
            present = getattr(_ser, flag)
            self.assertEqual(info.is_fallback, not present, name)
            self.assertEqual(info.kind, "candidate" if present else "reference", name)

    def test_honest_fallback_when_lib_absent(self):
        # force each accelerator "absent" and assert: labeled fallback + correct output
        for name, flag in LIB_OPS.items():
            op = get_tool(name)
            args, kw = op.bench_inputs(SPECS[name])
            saved = getattr(_ser, flag)
            try:
                setattr(_ser, flag, False)
                info = op.implementations()["optimized"]
                self.assertTrue(
                    info.is_fallback, f"{name} should be fallback when absent"
                )
                # dispatch must still work (falls back to naive), never crash
                out_opt = op.dispatch("optimized", *args, **kw)
                out_naive = op.dispatch("naive", *args, **kw)
                if name == "json_serialize":
                    import json

                    self.assertEqual(json.loads(out_opt), json.loads(out_naive))
                else:
                    self.assertEqual(out_opt, out_naive)
            finally:
                setattr(_ser, flag, saved)


class TestOptimizedSpeed(unittest.TestCase):
    """Timing-sensitive; opt-in via PACE_AGENT_PERF_ASSERT=1 (skipped by default)."""

    def test_optimized_faster(self):
        if not os.environ.get("PACE_AGENT_PERF_ASSERT"):
            self.skipTest("timing-sensitive; set PACE_AGENT_PERF_ASSERT=1 to enable")
        # clearest wins: vectorized numpy tiers and the algorithmic O(n^2)->O(n) ones.
        # vectorized ops use their large bench defaults; the rest use the parity specs.
        for name in ["vector_search", "quantize", "dedup", "rerank"]:
            spec = {} if name in ("vector_search", "quantize") else SPECS[name]
            op = get_tool(name)
            args, kw = op.bench_inputs(spec)
            with warnings.catch_warnings():
                _quiet()
                tn = min(
                    time_call(lambda *a, **k: op.run("naive", *a, **k), args, kw, 3)
                )
                to = min(
                    time_call(lambda *a, **k: op.run("optimized", *a, **k), args, kw, 3)
                )
            self.assertLess(to, tn, f"{name}: optimized {to:.5f}s !< naive {tn:.5f}s")


if __name__ == "__main__":
    unittest.main()
