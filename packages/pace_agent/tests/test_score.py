# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import unittest

from pace_agent import score
from pace_agent.errors import ConfigError


class TestMatchers(unittest.TestCase):
    def test_strict_ignores_only_formatting(self):
        self.assertTrue(score.match_strict("1700", "$1,700"))
        self.assertTrue(score.match_strict("  Yes ", "yes"))
        self.assertTrue(score.match_strict("a,b", "a, b"))  # a list stays two items
        self.assertFalse(score.match_strict("the answer is 17", "17"))

    def test_loose_accepts_an_answer_in_a_sentence(self):
        self.assertTrue(score.match_loose("the answer is 17", "17"))
        self.assertTrue(score.match_loose("about 42.0 dollars", "42"))
        self.assertFalse(score.match_loose("no idea", "17"))

    def test_loose_containment_can_over_credit(self):
        # documented consequence of substring matching: read the answers before quoting accuracy
        self.assertTrue(score.match_loose("111.25", "11.2"))
        self.assertFalse(score.match_strict("111.25", "11.2"))


class TestScore(unittest.TestCase):
    def test_missing_expected_answer_is_unscored_not_wrong(self):
        for gold in (None, "", "   "):
            r = score.score("17", gold)
            self.assertIsNone(r["match_strict"], gold)
            self.assertIsNone(r["match_loose"], gold)

    def test_no_answer_against_an_expected_answer_is_a_miss(self):
        # a crashed or stalled task must stay in the accuracy denominator
        r = score.score(None, "17")
        self.assertEqual((r["match_strict"], r["match_loose"]), (False, False))

    def test_both_verdicts_are_always_reported(self):
        r = score.score("the answer is 17", "17")
        self.assertFalse(r["match_strict"])
        self.assertTrue(r["match_loose"])
        self.assertNotIn("match", r)  # nothing picks a headline verdict

    def test_aggregate_separates_wrong_from_ungraded(self):
        rows = [
            score.score("17", "17"),
            score.score("nope", "17"),
            score.score(None, "17"),
            score.score("17", None),
        ]
        agg = score.aggregate(rows)
        self.assertEqual(agg["n_scored"], 3)
        self.assertEqual(agg["n_unscored"], 1)
        self.assertEqual(agg["auto_match_strict"], 1)
        self.assertEqual(agg["auto_match_loose"], 1)


class TestResolve(unittest.TestCase):
    def test_an_absent_block_is_ungraded_and_a_present_one_names_the_gold_key(self):
        self.assertIsNone(score.resolve(None))
        self.assertIsNone(score.resolve({}))
        self.assertEqual(score.resolve({"gold_key": "gold"}), "gold")
        self.assertEqual(score.resolve({"gold_key": "expected"}), "expected")

    def test_malformed_block_is_rejected(self):
        with self.assertRaises(ConfigError):
            score.resolve("gold")


if __name__ == "__main__":
    unittest.main()
