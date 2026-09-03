# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest

from pace_agent import get_tool, list_tools, register_tool, PaceAgentTool, load_plugins


class TestRegistry(unittest.TestCase):
    def test_ops_registered(self):
        names = {o.name for o in list_tools()}
        for expected in [
            "vector_search",
            "bm25",
            "quantize",
            "dom",
            "tokenize",
            "chunk",
        ]:
            self.assertIn(expected, names)

    def test_get_op_unknown(self):
        with self.assertRaises(KeyError):
            get_tool("does_not_exist")

    def test_tiers_and_parity(self):
        op = get_tool("vector_search")
        args, kw = op.bench_inputs({"N": 200, "dim": 16, "top_k": 5})
        idx, scores = op.run("naive", *args, **kw)
        self.assertEqual(len(idx), 5)
        self.assertTrue(op.parity(args, kw))

    def test_io_pattern_flag(self):
        self.assertIs(get_tool("dom").io_pattern, True)
        self.assertIs(get_tool("vector_search").io_pattern, False)

    def test_external_registration(self):
        @register_tool("custom_double", "context")
        class _D(PaceAgentTool):
            def naive(self, x):
                return [v * 2 for v in x]

        self.assertIn("custom_double", {o.name for o in list_tools()})
        self.assertEqual(get_tool("custom_double").run("naive", [1, 2]), [2, 4])

    def test_load_plugins_noop(self):
        load_plugins([])


class TestOptimizedAvailability(unittest.TestCase):
    """The `optimized_available()` honesty hook on the PaceAgentTool contract."""

    def test_default_available_true(self):
        self.assertTrue(PaceAgentTool().optimized_available())

    def test_override_gated_by_availability(self):
        # an op that overrides optimized but reports it unavailable => labeled fallback
        @register_tool("_avail_probe", "context")
        class _Probe(PaceAgentTool):
            def naive(self, x):
                return x

            def optimized(self, x):
                return x

            def optimized_available(self):
                return False

        try:
            info = get_tool("_avail_probe").implementations()["optimized"]
            self.assertTrue(info.is_fallback)
            self.assertEqual(info.kind, "reference")
        finally:
            from pace_agent.registry import TOOLS

            TOOLS.pop("_avail_probe", None)
