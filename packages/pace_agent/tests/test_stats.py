# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import statistics
import unittest

from pace_agent.stats import bootstrap_ci, measure, speedup_ci, summarize


class TestStats(unittest.TestCase):
    def test_percentile_gating_by_sample_size(self):
        self.assertNotIn("p95", summarize([1.0] * 10))  # n < 20 -> no tail
        s20 = summarize([float(i) for i in range(20)])
        self.assertIn("p95", s20)
        self.assertNotIn("p99", s20)  # 20 <= n < 100 -> p95 only
        s100 = summarize([float(i) for i in range(100)])
        self.assertIn("p95", s100)
        self.assertIn("p99", s100)  # n >= 100 -> both

    def test_bootstrap_ci_brackets_median_and_is_deterministic(self):
        xs = [float(i) for i in range(50)]
        ci = bootstrap_ci(xs, seed=0)
        self.assertIsNotNone(ci)
        self.assertLessEqual(ci[0], statistics.median(xs))
        self.assertLessEqual(statistics.median(xs), ci[1])
        self.assertEqual(ci, bootstrap_ci(xs, seed=0))

    def test_bootstrap_ci_needs_two_samples(self):
        self.assertIsNone(bootstrap_ci([1.0]))

    def test_speedup_ci(self):
        ci = speedup_ci([2.0] * 30, [1.0] * 30, seed=0)
        self.assertIsNotNone(ci)
        self.assertAlmostEqual(ci[0], 2.0, places=6)
        self.assertAlmostEqual(ci[1], 2.0, places=6)

    def test_measure_samples_and_warmup(self):
        calls = {"a": 0, "b": 0}

        def mk(k):
            def f():
                calls[k] += 1

            return f

        out = measure({"a": mk("a"), "b": mk("b")}, samples=5, warmup=2)
        self.assertEqual(len(out["a"]), 5)
        self.assertEqual(len(out["b"]), 5)
        self.assertEqual(calls["a"], 7)  # 2 warmup + 5 sampled
