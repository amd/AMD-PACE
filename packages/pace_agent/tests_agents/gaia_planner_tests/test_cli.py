# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import unittest
from types import SimpleNamespace

from gaia_planner.cli import _run_task, _select_tasks, _summary
from gaia_planner.models import GaiaTask, PlannerRunResult


class FakePlanner:
    def run(self, task):
        return PlannerRunResult(
            task_id=task.task_id,
            answer="42",
            raw_answer="FINAL ANSWER: 42",
            messages=[],
            tool_events=[],
        )


class TestCli(unittest.TestCase):
    def test_selection_result_and_summary(self):
        tasks = [
            GaiaTask(task_id="one", question="q1"),
            GaiaTask(task_id="two", question="q2", expected="42"),
        ]

        self.assertEqual(_select_tasks(tasks, "two", None), [tasks[1]])
        record = _run_task(FakePlanner(), tasks[1])
        summary = _summary([record])

        self.assertTrue(record["match"])
        self.assertTrue(record["match_strict"])
        self.assertTrue(record["match_loose"])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["correct_loose"], 1)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["unscored"], 0)
        self.assertIn("pace_agent.score", summary["metric"])

    def test_a_task_without_a_reference_answer_is_reported_as_ungraded(self):
        record = _run_task(FakePlanner(), GaiaTask(task_id="one", question="q1"))
        summary = _summary([record])

        self.assertIsNone(record["match_strict"])
        self.assertEqual((summary["scored"], summary["unscored"]), (0, 1))

    def test_task_error_is_persistable(self):
        planner = SimpleNamespace(
            run=lambda _task: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        record = _run_task(planner, GaiaTask(task_id="bad", question="q"))

        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"], "RuntimeError: boom")


if __name__ == "__main__":
    unittest.main()
